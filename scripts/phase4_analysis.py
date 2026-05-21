"""Phase 4 analysis: depth × diameter heatmap + H3 test.

Inputs:  results/phase4/d{1-4}__L{1-4}__seed{0-2}/episodes.csv
Outputs:
  - results/phase4/figures/heatmap_final_return.{pdf,png}
  - results/phase4/figures/depth_diameter_curves.{pdf,png}
  - results/phase4/summary_grid.csv
  - results/phase4/h3_test.csv  -- per diameter, is argmax at L≈d?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnmarl.utils.plotting import heatmap, _save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


DIAMETERS = (1, 2, 3)
DEPTHS = (1, 2, 3, 4)
DIAM_LABELS = {1: "complete (N=4)", 2: "ring (N=4)", 3: "line (N=4)"}


def load_final(run_dir: Path, tail_frac: float = 0.1) -> float:
    df = pd.read_csv(run_dir / "episodes.csv")
    ret = df["return"].to_numpy(dtype=float)
    if len(ret) == 0:
        return float("nan")
    tail = max(1, int(len(ret) * tail_frac))
    return float(np.mean(ret[-tail:]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results" / "phase4")
    args = parser.parse_args()

    # Collect per (d, L, seed).
    rows = []
    for child in sorted(args.results.iterdir()):
        if not child.is_dir() or not (child / "episodes.csv").exists():
            continue
        parts = child.name.split("__")
        try:
            d = int(parts[0].lstrip("d"))
            L = int(parts[1].lstrip("L"))
            seed = int(parts[2].replace("seed", ""))
        except (IndexError, ValueError):
            continue
        rows.append({
            "diameter": d,
            "depth": L,
            "seed": seed,
            "final_return": load_final(child),
        })
    if not rows:
        print(f"no runs found in {args.results}", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(args.results / "summary_grid.csv", index=False)
    print(f"[phase4] {len(df)} runs loaded")

    # Aggregate: mean per (d, L).
    grid = df.groupby(["diameter", "depth"])["final_return"].agg(["mean", "std", "count"]).reset_index()
    grid_mean = grid.pivot(index="diameter", columns="depth", values="mean").reindex(
        index=DIAMETERS, columns=DEPTHS
    )

    fig_dir = args.results / "figures"
    heatmap(
        grid_mean.to_numpy(),
        xlabels=[f"L={L}" for L in DEPTHS],
        ylabels=[f"d={d} ({DIAM_LABELS[d]})" for d in DIAMETERS],
        out_stem=fig_dir / "heatmap_final_return",
        title="GNN-QMIX final return by depth x diameter",
        xlabel="GNN depth L",
        ylabel="Graph diameter d",
        cbar_label="mean final-window return",
    )

    # Curves: one line per diameter, x-axis is depth.
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for d in DIAMETERS:
        sub = grid[grid["diameter"] == d].sort_values("depth")
        ax.errorbar(
            sub["depth"], sub["mean"], yerr=sub["std"].fillna(0.0),
            marker="o", lw=1.5, capsize=2,
            label=f"d={d} ({DIAM_LABELS[d]})",
        )
    ax.set_xlabel("GNN depth L", fontsize=9)
    ax.set_ylabel("Mean final-window return", fontsize=9)
    ax.set_title("H3: depth-diameter alignment", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7, loc="best")
    _save(fig, fig_dir / "depth_diameter_curves")

    # H3 test (pre-registered criterion, see paper Sec. 4):
    #   For each diameter d, declare a *match* if:
    #     (a) the mean curve over depths is non-flat (relative range
    #         (max - min) / max >= flatness_floor), AND
    #     (b) |argmax_L - d| <= 1.
    #   Declare H3 *confirmed at the family level* if at least
    #   ceil(n_diameters * 2/3) per-diameter rows match.
    # The flatness floor prevents a noisy flat curve from accidentally
    # passing the argmax test.
    FLATNESS_FLOOR = 0.10  # 10% range relative to peak; pre-registered.
    DELTA_L = 1            # pre-registered ± window.

    h3_rows = []
    for d in DIAMETERS:
        row = grid_mean.loc[d]
        if row.isna().all():
            continue
        argmax_L = int(row.idxmax())
        peak_val = float(row.max())
        floor_val = float(row.min())
        rel_range = (peak_val - floor_val) / max(abs(peak_val), 1e-9)
        non_flat = bool(rel_range >= FLATNESS_FLOOR)
        within_window = bool(abs(argmax_L - d) <= DELTA_L)
        h3_rows.append({
            "diameter": d,
            "argmax_depth": argmax_L,
            "peak_mean_return": peak_val,
            "floor_mean_return": floor_val,
            "relative_range": rel_range,
            "non_flat": int(non_flat),
            "within_window": int(within_window),
            "h3_match": int(non_flat and within_window),
        })
    pd.DataFrame(h3_rows).to_csv(args.results / "h3_test.csv", index=False)

    print(f"[phase4] H3 per-diameter test "
          f"(flatness floor {FLATNESS_FLOOR}, window ±{DELTA_L}):")
    for r in h3_rows:
        flat = "flat" if not r["non_flat"] else "non-flat"
        ok = "PASS" if r["h3_match"] else "FAIL"
        print(f"   d={r['diameter']}: argmax L={r['argmax_depth']} "
              f"({flat}, rel-range={r['relative_range']:.2f})  [{ok}]")
    n_match = sum(r["h3_match"] for r in h3_rows)
    n_total = len(h3_rows)
    # Family-level criterion: at least ceil(2/3 * n_total).
    threshold = -(-n_total * 2 // 3)
    confirmed = n_match >= threshold
    print(f"[phase4] family-level H3: {n_match}/{n_total} pass "
          f"(need >= {threshold}) -> {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")
    # Write a one-line family-level summary too.
    pd.DataFrame([{
        "n_diameters": n_total,
        "n_match": n_match,
        "threshold": threshold,
        "h3_family_confirmed": int(confirmed),
        "flatness_floor": FLATNESS_FLOOR,
        "delta_L": DELTA_L,
    }]).to_csv(args.results / "h3_family.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
