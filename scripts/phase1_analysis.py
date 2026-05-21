"""Phase 1 — analysis: figures + summary table from results/phase1/.

Produces:

- ``results/phase1/figures/learning_curves.{pdf,png}``
- ``results/phase1/summary.csv`` — per (algo, seed) final-window mean and
  episodes-to-80%-of-best.
- ``results/phase1/summary_aggregate.csv`` — per algo, mean ± 95% t-CI.
- ``results/phase1/pairwise.csv`` — Welch's t with Holm correction.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnmarl.utils.plotting import ALGO_LABELS, learning_curves  # noqa: E402
from gnnmarl.utils.stats import (  # noqa: E402
    episodes_to_threshold,
    mean_with_ci,
    pairwise_compare,
    self_threshold,
)


ALGOS = ("iql", "vdn", "qmix", "gnn_qmix")


def load_returns(run_dir: Path) -> np.ndarray:
    df = pd.read_csv(run_dir / "episodes.csv")
    return df["return"].to_numpy(dtype=float)


def collect(results_dir: Path) -> dict[str, list[tuple[int, np.ndarray]]]:
    """{algo: [(seed, returns), ...]} discovered by directory name."""
    out: dict[str, list[tuple[int, np.ndarray]]] = {a: [] for a in ALGOS}
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or not (child / "episodes.csv").exists():
            continue
        # Run ids look like: "gnn_qmix__ring__N4__seed0".
        parts = child.name.split("__")
        algo = parts[0]
        if algo not in out:
            continue
        seed = int(parts[-1].replace("seed", ""))
        out[algo].append((seed, load_returns(child)))
    for a in out:
        out[a].sort(key=lambda t: t[0])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results" / "phase1")
    parser.add_argument("--window", type=int, default=25)
    args = parser.parse_args()

    runs = collect(args.results)
    counts = {a: len(v) for a, v in runs.items()}
    if not any(counts.values()):
        print(f"no runs found in {args.results}", file=sys.stderr)
        return 1
    print(f"[analysis] loaded runs: {counts}")

    # --- learning-curve figure -------------------------------------------------
    returns_by_algo = {a: [r for _, r in v] for a, v in runs.items() if v}
    fig_dir = args.results / "figures"
    pdf, png = learning_curves(
        returns_by_algo,
        out_stem=fig_dir / "learning_curves",
        window=args.window,
        title=f"Phase 1 pilot — CoordGrid (ring, N=4), 3 seeds",
        ylabel=f"Episode return ({args.window}-ep rolling mean)",
    )
    print(f"[analysis] wrote {pdf.name} / {png.name}")

    # --- per-(algo, seed) summary ----------------------------------------------
    # Per-algorithm self-threshold: episodes-to-reach-80%-of-OWN final-window
    # mean. This metric is invariant to the panel of algorithms (so adding
    # or removing one does not change any other algorithm's number).
    summary_rows = []
    for algo, items in runs.items():
        for seed, returns in items:
            tail = max(1, len(returns) // 10)
            final = float(np.mean(returns[-tail:]))
            self_thr = self_threshold(returns, fraction=0.8)
            ep80 = episodes_to_threshold(returns, threshold=self_thr,
                                         window=args.window)
            summary_rows.append({
                "algo": algo,
                "seed": seed,
                "final_window_mean": final,
                "self_threshold_80pct": self_thr,
                "episodes_to_80pct_of_own": ep80 if ep80 is not None else "",
                "n_episodes": len(returns),
            })
    pd.DataFrame(summary_rows).to_csv(args.results / "summary.csv", index=False)
    print(f"[analysis] wrote summary.csv ({len(summary_rows)} rows)")

    # --- per-algo aggregate ----------------------------------------------------
    agg_rows = []
    for algo in ALGOS:
        items = runs.get(algo, [])
        if not items:
            continue
        finals = [
            float(np.mean(r[-max(1, len(r) // 10):])) for _, r in items
        ]
        ep80s = [
            episodes_to_threshold(
                r,
                threshold=self_threshold(r, fraction=0.8),
                window=args.window,
            )
            for _, r in items
        ]
        m_f, lo_f, hi_f = mean_with_ci(finals)
        ep_valid = [e for e in ep80s if e is not None]
        if ep_valid:
            m_e, lo_e, hi_e = mean_with_ci(ep_valid)
        else:
            m_e = lo_e = hi_e = float("nan")
        agg_rows.append({
            "algo": algo,
            "n_seeds": len(items),
            "final_mean": m_f,
            "final_ci_lo": lo_f,
            "final_ci_hi": hi_f,
            "ep80_mean": m_e,
            "ep80_ci_lo": lo_e,
            "ep80_ci_hi": hi_e,
            "ep80_n_reaching_threshold": len(ep_valid),
        })
    pd.DataFrame(agg_rows).to_csv(args.results / "summary_aggregate.csv", index=False)
    print(f"[analysis] wrote summary_aggregate.csv")

    # --- pairwise tests on final-window mean -----------------------------------
    seed_scores = {
        algo: [
            float(np.mean(r[-max(1, len(r) // 10):])) for _, r in items
        ]
        for algo, items in runs.items() if items
    }
    if sum(len(v) for v in seed_scores.values()) >= 2 and len(seed_scores) >= 2:
        rows = pairwise_compare(seed_scores, test="welch")
        with (args.results / "pairwise.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["a", "b", "mean_a", "mean_b", "stat", "p_raw", "p_holm", "test"])
            for r in rows:
                w.writerow([r.a, r.b, r.metric_a, r.metric_b, r.stat,
                            r.p_raw, r.p_holm, r.test])
        print(f"[analysis] wrote pairwise.csv ({len(rows)} pairs)")

    # --- console table ---------------------------------------------------------
    print()
    print(f"{'algo':>10} | {'final mean':>10} | {'95% CI':>20} | {'ep->80%':>8}")
    print("-" * 60)
    for row in agg_rows:
        ci = f"[{row['final_ci_lo']:.2f}, {row['final_ci_hi']:.2f}]"
        ep80 = (
            f"{row['ep80_mean']:.0f}"
            if not np.isnan(row['ep80_mean'])
            else "n/a"
        )
        print(f"{ALGO_LABELS[row['algo']]:>10} | {row['final_mean']:>10.3f} | "
              f"{ci:>20} | {ep80:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
