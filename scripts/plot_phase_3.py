"""Phase 3 analysis + plotting.

Reads results/phase_3_mpe/{algo}/n{N}_seed{S}.csv files. Produces:

1. results/phase_3_mpe/learning_curves_n{N}.png      - per-N eval-return curves
2. results/phase_3_mpe/summary.csv                    - aggregate stats
3. results/phase_3_mpe/SUMMARY.md                     - human-readable

simple_spread doesn't have an explicit success metric; we use eval
return mean (less negative = better).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PHASE_3 = ROOT / "results" / "phase_3_mpe"

ALGOS = ["iql", "vdn", "qmix", "gnn_qmix"]
N_AGENTS = [3, 6]
COLORS = {
    "iql": "#888888",
    "vdn": "#1f77b4",
    "qmix": "#ff7f0e",
    "gnn_qmix": "#d62728",
}
LABELS = {"iql": "IQL", "vdn": "VDN", "qmix": "QMIX", "gnn_qmix": "GNN-QMIX"}


def load_curves(algo: str, n: int) -> pd.DataFrame:
    rows = []
    for f in sorted((PHASE_3 / algo).glob(f"n{n}_seed*.csv")):
        seed = int(f.stem.split("seed")[-1])
        df = pd.read_csv(f)
        ev = df[df["kind"] == "eval"].copy()
        ev["eval_return_mean"] = pd.to_numeric(ev["eval_return_mean"], errors="coerce")
        ev["episode"] = pd.to_numeric(ev["episode"])
        ev["seed"] = seed
        ev["algo"] = algo
        ev["n_agents"] = n
        rows.append(ev)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_per_n(n: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    for algo in ALGOS:
        df = load_curves(algo, n)
        if df.empty:
            continue
        agg = (
            df.groupby("episode")["eval_return_mean"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        x = agg["episode"].values
        y = agg["mean"].values
        sd = agg["std"].fillna(0.0).values
        sem = sd / np.sqrt(np.maximum(agg["count"].values, 1))
        ax.plot(x, y, label=LABELS[algo], color=COLORS[algo], lw=2.0)
        ax.fill_between(x, y - sem, y + sem, color=COLORS[algo], alpha=0.18)
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Eval return (mean; less negative = better)")
    ax.set_title(f"simple_spread N={n}: eval return vs episodes (mean ± SEM)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    print(f"  -> {out}")
    plt.close(fig)


def aggregate() -> pd.DataFrame:
    rows = []
    for algo in ALGOS:
        for n in N_AGENTS:
            df = load_curves(algo, n)
            if df.empty:
                continue
            for seed, sd in df.groupby("seed"):
                sd = sd.sort_values("episode")
                final = float(
                    sd.tail(max(1, len(sd) // 4))["eval_return_mean"].mean()
                )
                best = float(sd["eval_return_mean"].max())
                rows.append(
                    {"algo": algo, "n_agents": n, "seed": seed,
                     "final_return": final, "best_return": best}
                )
    return pd.DataFrame(rows)


def main() -> int:
    if not PHASE_3.exists():
        print(f"!! {PHASE_3} not found — run scripts/run_phase_3.py first.")
        return 2
    for n in N_AGENTS:
        plot_per_n(n, PHASE_3 / f"learning_curves_n{n}.png")
    per_seed = aggregate()
    if per_seed.empty:
        print("No data found.")
        return 1
    per_seed.to_csv(PHASE_3 / "per_seed.csv", index=False)
    summary = (
        per_seed.groupby(["algo", "n_agents"])
        .agg(
            n=("seed", "count"),
            final_mean=("final_return", "mean"),
            final_std=("final_return", "std"),
            best_mean=("best_return", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(PHASE_3 / "summary.csv", index=False)
    print(f"  -> {PHASE_3 / 'summary.csv'}")
    print()
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
