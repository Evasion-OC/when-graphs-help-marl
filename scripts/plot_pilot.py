"""Phase 1 pilot analysis + plotting.

Reads results/pilot/{algo}_seed{N}.csv files and produces:

1. results/pilot/learning_curves.png     - eval success rate vs episode, per algo
2. results/pilot/eval_return_curves.png  - eval return mean vs episode, per algo
3. results/pilot/summary.csv             - per-algo aggregate stats
4. results/pilot/SUMMARY.md              - human-readable summary

Run after the sweep:
    python scripts/plot_pilot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "results" / "pilot"

ALGOS = ["iql", "vdn", "qmix", "gnn_qmix"]
COLORS = {
    "iql": "#888888",
    "vdn": "#1f77b4",
    "qmix": "#ff7f0e",
    "gnn_qmix": "#d62728",
}
LABELS = {"iql": "IQL", "vdn": "VDN", "qmix": "QMIX", "gnn_qmix": "GNN-QMIX"}


def load_eval_curves(algo: str) -> pd.DataFrame:
    """Stack eval rows from every seed into one long-format DataFrame."""
    frames = []
    for f in sorted(PILOT.glob(f"{algo}_seed*.csv")):
        seed = int(f.stem.split("seed")[-1])
        df = pd.read_csv(f)
        ev = df[df["kind"] == "eval"].copy()
        for c in ["eval_return_mean", "eval_success_rate", "eval_length_mean"]:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")
        ev["episode"] = pd.to_numeric(ev["episode"])
        ev["step"] = pd.to_numeric(ev["step"])
        ev["seed"] = seed
        ev["algo"] = algo
        frames.append(ev[
            ["algo", "seed", "step", "episode",
             "eval_return_mean", "eval_success_rate", "eval_length_mean"]
        ])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_curves(metric: str, ylabel: str, out: Path, smooth: int = 1) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    for algo in ALGOS:
        df = load_eval_curves(algo)
        if df.empty:
            continue
        # group by episode (eval is taken at the same eval_every_episodes
        # across runs, so the episodes line up by construction)
        agg = (
            df.groupby("episode")[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        if smooth > 1:
            agg["mean_sm"] = agg["mean"].rolling(smooth, min_periods=1).mean()
        else:
            agg["mean_sm"] = agg["mean"]
        x = agg["episode"].values
        y = agg["mean_sm"].values
        sd = agg["std"].fillna(0.0).values
        n = agg["count"].values
        sem = sd / np.sqrt(np.maximum(n, 1))
        ax.plot(x, y, label=LABELS[algo], color=COLORS[algo], lw=2.0)
        ax.fill_between(x, y - sem, y + sem, color=COLORS[algo], alpha=0.18)
    ax.set_xlabel("Training episodes")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. episodes (mean ± SEM across 3 seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    print(f"  -> {out}")
    plt.close(fig)


def episodes_to_threshold(df: pd.DataFrame, metric: str, threshold: float) -> float:
    """First episode at which seed-mean of `metric` first reaches `threshold`."""
    if df.empty:
        return float("nan")
    agg = df.groupby("episode")[metric].mean().sort_index()
    idx = agg[agg >= threshold].index
    if len(idx) == 0:
        return float("nan")
    return float(idx.min())


def aggregate_summary() -> pd.DataFrame:
    rows = []
    for algo in ALGOS:
        df = load_eval_curves(algo)
        if df.empty:
            continue
        # Final-window stats: mean over last 25% of eval points
        n_evals = df["episode"].nunique()
        cutoff = df["episode"].quantile(0.75)
        late = df[df["episode"] >= cutoff]
        # Per-seed best success rate, then mean across seeds
        peaks = (
            df.groupby("seed")["eval_success_rate"].max().rename("peak").reset_index()
        )
        # Stability: std of seed-mean trajectory across all eval points (lower=more stable)
        traj = df.groupby("episode")["eval_success_rate"].mean().sort_index()
        stability_std = float(traj.diff().abs().mean())  # mean absolute change between checkpoints
        # Post-peak collapse: drop from peak success to final success per seed
        per_seed = []
        for s in df["seed"].unique():
            sd = df[df["seed"] == s].sort_values("episode")
            if sd.empty:
                continue
            peak_idx = sd["eval_success_rate"].idxmax()
            peak_val = sd.loc[peak_idx, "eval_success_rate"]
            final_val = sd.iloc[-max(1, len(sd) // 4):]["eval_success_rate"].mean()
            per_seed.append(peak_val - final_val)
        post_peak_drop = float(np.mean(per_seed)) if per_seed else float("nan")
        rows.append(
            {
                "algo": algo,
                "seeds": df["seed"].nunique(),
                "final_success_rate_mean": late["eval_success_rate"].mean(),
                "final_success_rate_std": late["eval_success_rate"].std(),
                "peak_success_rate_mean": peaks["peak"].mean(),
                "peak_success_rate_std": peaks["peak"].std(),
                "post_peak_drop_mean": post_peak_drop,
                "stability_step_change": stability_std,
                "final_eval_return_mean": late["eval_return_mean"].mean(),
                "final_episode_length": late["eval_length_mean"].mean(),
                "ep_to_succ_50pct": episodes_to_threshold(
                    df, "eval_success_rate", 0.5
                ),
                "ep_to_succ_70pct": episodes_to_threshold(
                    df, "eval_success_rate", 0.7
                ),
                "n_evals": n_evals,
            }
        )
    return pd.DataFrame(rows)


def write_summary_md(summary: pd.DataFrame, out: Path) -> None:
    lines = [
        "# Phase 1 pilot — summary",
        "",
        "**Task:** 2 agents on a 4×4 grid, max 20 steps/episode, distance-shaping enabled.",
        "**Budget:** 10,000 env steps per run, 3 seeds per algorithm.",
        "**Eval:** 20 greedy episodes every 50 training episodes.",
        "",
        "## Headline metrics",
        "",
    ]
    if summary.empty:
        lines.append("_No results found in `results/pilot/`._")
    else:
        cols_main = [
            ("algo", "Algorithm"),
            ("peak_success_rate_mean", "Peak success (mean across seeds)"),
            ("final_success_rate_mean", "Final-window success (mean ± SD)"),
            ("post_peak_drop_mean", "Post-peak drop (peak→final)"),
            ("ep_to_succ_70pct", "Episodes to 70% success"),
        ]
        display = summary.copy()
        # Compose final±SD column
        display["_final_succ_str"] = display.apply(
            lambda r: (
                f"{r.final_success_rate_mean:.2f} ± {r.final_success_rate_std:.2f}"
                if pd.notna(r.final_success_rate_mean)
                else "—"
            ),
            axis=1,
        )
        header = "| " + " | ".join(c[1] for c in cols_main) + " |"
        sep = "|" + "|".join("---" for _ in cols_main) + "|"
        lines.append(header)
        lines.append(sep)
        for _, row in display.iterrows():
            cells = []
            for key, _label in cols_main:
                if key == "algo":
                    cells.append(str(row[key]))
                elif key == "final_success_rate_mean":
                    cells.append(row["_final_succ_str"])
                else:
                    v = row.get(key, float("nan"))
                    cells.append(f"{v:.3f}" if pd.notna(v) else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("## Interpretation")
        lines.append("")
        lines.append("- **Peak vs. final**: A large drop (peak ≫ final) means the algorithm")
        lines.append("  found a good policy and then collapsed. This is the classic IQL")
        lines.append("  cooperative-MARL instability — independent learners chase a moving")
        lines.append("  target as their teammates' policies update.")
        lines.append("- **Episodes to 70%**: how quickly an algorithm first reaches 70% eval")
        lines.append("  success rate (averaged across seeds). Sample-efficiency proxy.")
        lines.append("- **Final-window**: mean over the last 25% of eval checkpoints; the")
        lines.append("  number we'd report if we trusted late-training behaviour. Pairs with")
        lines.append("  peak to expose collapse.")
    out.write_text("\n".join(lines) + "\n")
    print(f"  -> {out}")


def main() -> None:
    if not PILOT.exists():
        print(f"!! {PILOT} does not exist — run `scripts/run_pilot.py` first.")
        sys.exit(1)
    PILOT.mkdir(parents=True, exist_ok=True)
    plot_curves(
        "eval_success_rate",
        "Eval success rate",
        PILOT / "learning_curves.png",
        smooth=1,
    )
    plot_curves(
        "eval_return_mean",
        "Eval return (mean)",
        PILOT / "eval_return_curves.png",
        smooth=1,
    )
    summary = aggregate_summary()
    summary.to_csv(PILOT / "summary.csv", index=False)
    print(f"  -> {PILOT / 'summary.csv'}")
    write_summary_md(summary, PILOT / "SUMMARY.md")
    print()
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
