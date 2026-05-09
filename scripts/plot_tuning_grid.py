"""Phase 2A — visualize the tuning grid.

Reads results/phase_2_tune/{algo}/*.csv and produces a 4-panel figure:
2 algos (QMIX, GNN-QMIX) × 2 metrics (peak success, late q_tot).

Each panel is a 2×2×2 grid (`embed_dim` × `mixer_lr` × `mixer_init`)
flattened to one bar chart per panel. This makes it visually obvious
which axis matters and which configurations win.

Run after `scripts/run_tuning.py` (and ideally after `lock_tuning.py`
so the chosen config can be highlighted, but the script works on raw
CSVs alone).

Output: results/phase_2_tune/tuning_grid.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TUNE = ROOT / "results" / "phase_2_tune"


def parse_cfg_id(cfg_id: str) -> dict:
    out: dict = {}
    for part in cfg_id.split("__"):
        m = re.match(r"^([a-z_]+)=(.+)$", part)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        if k == "embed_dim":
            out[k] = int(v)
        elif k in {"mixer_lr", "init_scale"}:
            out[k] = float(v)
        else:
            out[k] = v
    return out


def collect(algo: str) -> pd.DataFrame:
    algo_dir = TUNE / algo
    if not algo_dir.exists():
        return pd.DataFrame()
    rows = []
    for f in sorted(algo_dir.glob("*.csv")):
        name = f.stem
        m = re.match(r"^(.+?)_seed(\d+)$", name)
        if not m:
            continue
        cfg_id = m.group(1)
        seed = int(m.group(2))
        params = parse_cfg_id(cfg_id)
        df = pd.read_csv(f)
        ev = df[df["kind"] == "eval"].copy()
        ev["eval_success_rate"] = pd.to_numeric(ev["eval_success_rate"])
        ev = ev.sort_values("episode")
        train = df[df["kind"] == "train"].copy()
        late_q = pd.to_numeric(
            train["q_tot_mean"].tail(50), errors="coerce"
        ).mean()
        peak = float(ev["eval_success_rate"].max()) if not ev.empty else float("nan")
        final = (
            float(ev.tail(max(1, len(ev) // 4))["eval_success_rate"].mean())
            if not ev.empty
            else float("nan")
        )
        rows.append(
            {"algo": algo, "seed": seed, "peak": peak, "final": final,
             "late_q_tot": late_q, **params}
        )
    return pd.DataFrame(rows)


def label_for(row: pd.Series) -> str:
    return (
        f"emb={int(row['embed_dim'])}\n"
        f"lr={row['mixer_lr']:g}\n"
        f"{row['mixer_init'][:4]}"
    )


def panel(ax, df: pd.DataFrame, metric: str, title: str) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    agg = (
        df.groupby(["embed_dim", "mixer_lr", "mixer_init"])[metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["embed_dim", "mixer_lr", "mixer_init"])
    )
    x = np.arange(len(agg))
    bars = ax.bar(
        x,
        agg["mean"],
        yerr=agg["std"].fillna(0),
        capsize=3,
        color=["#1f77b4" if init == "default" else "#d62728" for init in agg["mixer_init"]],
    )
    # Highlight the locked-config winner (by mean final-window success;
    # see scripts/lock_tuning.py for the rationale).
    if metric == "peak":
        # Recompute final-window mean to find the locker's pick
        final_agg = (
            df.groupby(["embed_dim", "mixer_lr", "mixer_init"])["final"]
            .mean()
            .reset_index()
            .sort_values(["embed_dim", "mixer_lr", "mixer_init"])
        )
        win_pos = int(final_agg["final"].values.argmax())
        bars[win_pos].set_edgecolor("gold")
        bars[win_pos].set_linewidth(3)
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(r) for _, r in agg.iterrows()], fontsize=6.5)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


def main() -> int:
    if not TUNE.exists():
        print(f"!! {TUNE} not found.")
        return 2
    qmix = collect("qmix")
    gnn = collect("gnn_qmix")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=130)
    panel(axes[0, 0], qmix, "peak", "QMIX — peak eval success rate")
    panel(axes[0, 1], gnn, "peak", "GNN-QMIX — peak eval success rate")
    panel(axes[1, 0], qmix, "late_q_tot", "QMIX — late-training q_tot")
    panel(axes[1, 1], gnn, "late_q_tot", "GNN-QMIX — late-training q_tot")
    fig.suptitle(
        "Phase 2A tuning grid — gold border = LOCKED config (best mean "
        "final-window success). Blue = default init, red = orthogonal(0.1)."
    )
    fig.tight_layout()
    out = TUNE / "tuning_grid.png"
    fig.savefig(out)
    print(f"  -> {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
