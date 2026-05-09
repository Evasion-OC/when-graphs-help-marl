"""Phase 2C analysis + plotting.

Reads results/phase_2_main/{algo}/{scale}_seed{N}.csv files. Outputs:

1. results/phase_2_main/learning_curves_{scale}.png  - per-scale curves
2. results/phase_2_main/summary.csv                  - per-algo per-scale stats
3. results/phase_2_main/SUMMARY.md                   - human-readable
4. results/phase_2_main/forest_plot.png              - peak success per algo
                                                       per scale, with 95% CIs
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "results" / "phase_2_main"

ALGOS = ["iql", "vdn", "qmix", "gnn_qmix"]
SCALES = ["small", "medium", "large"]
COLORS = {
    "iql": "#888888",
    "vdn": "#1f77b4",
    "qmix": "#ff7f0e",
    "gnn_qmix": "#d62728",
}
LABELS = {"iql": "IQL", "vdn": "VDN", "qmix": "QMIX", "gnn_qmix": "GNN-QMIX"}


def load_curves(algo: str, scale: str) -> pd.DataFrame:
    rows = []
    for f in sorted((MAIN / algo).glob(f"{scale}_seed*.csv")):
        seed = int(f.stem.split("seed")[-1])
        df = pd.read_csv(f)
        ev = df[df["kind"] == "eval"].copy()
        for c in ["eval_return_mean", "eval_success_rate", "eval_length_mean"]:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")
        ev["episode"] = pd.to_numeric(ev["episode"])
        ev["seed"] = seed
        ev["algo"] = algo
        ev["scale"] = scale
        rows.append(ev)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_per_scale(scale: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    for algo in ALGOS:
        df = load_curves(algo, scale)
        if df.empty:
            continue
        agg = df.groupby("episode")["eval_success_rate"].agg(["mean", "std", "count"]).reset_index()
        x = agg["episode"].values
        y = agg["mean"].values
        sd = agg["std"].fillna(0.0).values
        n = agg["count"].values
        sem = sd / np.sqrt(np.maximum(n, 1))
        ax.plot(x, y, label=LABELS[algo], color=COLORS[algo], lw=2.0)
        ax.fill_between(x, y - sem, y + sem, color=COLORS[algo], alpha=0.18)
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Eval success rate")
    ax.set_title(f"{scale.capitalize()} task: eval success vs episodes (mean ± SEM)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    print(f"  -> {out}")
    plt.close(fig)


def per_seed_summary() -> pd.DataFrame:
    rows = []
    for algo in ALGOS:
        for scale in SCALES:
            df = load_curves(algo, scale)
            if df.empty:
                continue
            for seed, sd in df.groupby("seed"):
                sd = sd.sort_values("episode")
                peak = float(sd["eval_success_rate"].max())
                final = float(sd.tail(max(1, len(sd) // 4))["eval_success_rate"].mean())
                rows.append(
                    {
                        "algo": algo,
                        "scale": scale,
                        "seed": seed,
                        "peak": peak,
                        "final": final,
                    }
                )
    return pd.DataFrame(rows)


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    if per_seed.empty:
        return per_seed
    return (
        per_seed.groupby(["algo", "scale"])
        .agg(
            n=("seed", "count"),
            peak_mean=("peak", "mean"),
            peak_std=("peak", "std"),
            final_mean=("final", "mean"),
            final_std=("final", "std"),
        )
        .reset_index()
    )


def plot_forest(per_seed: pd.DataFrame, out: Path) -> None:
    """Peak success per (algo × scale), points = seeds, bars = 95% CI."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    yticks = []
    yticklabels = []
    y = 0
    for scale in SCALES[::-1]:
        for algo in ALGOS[::-1]:
            sub = per_seed[(per_seed["algo"] == algo) & (per_seed["scale"] == scale)]
            if sub.empty:
                y += 1
                continue
            vals = sub["peak"].values
            mean = vals.mean()
            ci = (
                stats.t.interval(0.95, df=max(len(vals) - 1, 1), loc=mean, scale=stats.sem(vals))
                if len(vals) > 1
                else (mean, mean)
            )
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - ci[0]], [ci[1] - mean]],
                fmt="o",
                color=COLORS[algo],
                markersize=8,
                lw=2,
                capsize=4,
            )
            ax.scatter(vals, [y] * len(vals), color=COLORS[algo], alpha=0.4, s=20)
            yticks.append(y)
            yticklabels.append(f"{scale} – {LABELS[algo]}")
            y += 1
        y += 0.5  # gap between scales
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.set_xlabel("Peak eval success rate (95% CI; dots = seeds)")
    ax.set_xlim(0, 1)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title("Phase 2C: Peak success per algorithm × task scale")
    fig.tight_layout()
    fig.savefig(out)
    print(f"  -> {out}")
    plt.close(fig)


def main() -> int:
    if not MAIN.exists():
        print(f"!! {MAIN} not found — run scripts/run_main_sweep.py first.")
        return 2
    for s in SCALES:
        plot_per_scale(s, MAIN / f"learning_curves_{s}.png")
    per_seed = per_seed_summary()
    if per_seed.empty:
        print("No data found.")
        return 1
    per_seed.to_csv(MAIN / "per_seed.csv", index=False)
    agg = aggregate(per_seed)
    agg.to_csv(MAIN / "summary.csv", index=False)
    print(f"  -> {MAIN / 'summary.csv'}")
    plot_forest(per_seed, MAIN / "forest_plot.png")
    print()
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
