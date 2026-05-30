"""Phase 4 analysis — depth × diameter heatmap and the inverted-U test.

For GNN-QMIX, computes mean final-window success rate over seeds at
each (L, N) cell, where N determines ring diameter d = floor(N/2).
Plots a heatmap and overlays the predicted L = d ridge.

Then plots final-window success vs. (L − d) collapsed across N — if H3
is right, this should look like an inverted-U peaking near zero.

QMIX is plotted as a flat reference line per N (no L axis).

Outputs:
    results/phase_4_ablation/heatmap.png
    results/phase_4_ablation/inverted_u.png
    results/phase_4_ablation/summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PHASE_4 = ROOT / "results" / "phase_4_ablation"


def load_gnn_qmix() -> pd.DataFrame:
    rows = []
    for f in sorted((PHASE_4 / "gnn_qmix").glob("L*_N*_seed*.csv")):
        # filename: L{L}_N{N}_seed{S}.csv
        parts = f.stem.split("_")
        L = int(parts[0][1:])
        N = int(parts[1][1:])
        seed = int(parts[2][4:])
        df = pd.read_csv(f)
        ev = df[df["kind"] == "eval"].copy()
        ev["eval_success_rate"] = pd.to_numeric(ev["eval_success_rate"])
        if ev.empty:
            continue
        final = float(ev.tail(max(1, len(ev) // 4))["eval_success_rate"].mean())
        rows.append({"L": L, "N": N, "seed": seed, "final": final})
    return pd.DataFrame(rows)


def load_qmix() -> pd.DataFrame:
    rows = []
    for f in sorted((PHASE_4 / "qmix").glob("N*_seed*.csv")):
        parts = f.stem.split("_")
        N = int(parts[0][1:])
        seed = int(parts[1][4:])
        df = pd.read_csv(f)
        ev = df[df["kind"] == "eval"].copy()
        ev["eval_success_rate"] = pd.to_numeric(ev["eval_success_rate"])
        if ev.empty:
            continue
        final = float(ev.tail(max(1, len(ev) // 4))["eval_success_rate"].mean())
        rows.append({"N": N, "seed": seed, "final": final})
    return pd.DataFrame(rows)


def plot_heatmap(gnn: pd.DataFrame, out: Path) -> None:
    if gnn.empty:
        return
    pivot = gnn.groupby(["L", "N"])["final"].mean().unstack("N")
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=140)
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"N={n}\nd={n//2}" for n in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"L={layer}" for layer in pivot.index])
    ax.set_title("Phase 4: GNN-QMIX final-window success — depth (L) × diameter (d=N/2)")
    fig.colorbar(im, ax=ax, label="Mean final-window success")
    # Predicted L=d ridge
    for col_pos, n in enumerate(pivot.columns):
        d = n // 2
        if d in pivot.index:
            row_pos = list(pivot.index).index(d)
            ax.plot(col_pos, row_pos, marker="*", markersize=14,
                    markerfacecolor="white", markeredgecolor="red")
    ax.text(0.02, 0.98, "★ = predicted L=d ridge",
            transform=ax.transAxes, color="red", fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    fig.tight_layout()
    fig.savefig(out)
    print(f"  -> {out}")
    plt.close(fig)


def plot_inverted_u(gnn: pd.DataFrame, out: Path) -> None:
    if gnn.empty:
        return
    gnn = gnn.copy()
    gnn["d"] = gnn["N"] // 2
    gnn["delta"] = gnn["L"] - gnn["d"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=140)
    agg = (
        gnn.groupby("delta")["final"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("delta")
    )
    sem = agg["std"].fillna(0) / np.sqrt(agg["count"].clip(lower=1))
    ax.errorbar(agg["delta"], agg["mean"], yerr=sem,
                fmt="o-", lw=2, capsize=4, color="#d62728",
                label="GNN-QMIX")
    ax.axvline(0, ls="--", color="gray", alpha=0.5)
    ax.set_xlabel("L − d  (GNN depth minus graph diameter)")
    ax.set_ylabel("Final-window success rate")
    ax.set_title("Phase 4 inverted-U test: GNN-QMIX collapsed across N (mean ± SEM)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    print(f"  -> {out}")
    plt.close(fig)


def main() -> int:
    if not PHASE_4.exists():
        print(f"!! {PHASE_4} not found.")
        return 2
    gnn = load_gnn_qmix()
    qmix = load_qmix()
    if gnn.empty:
        print("No GNN-QMIX data yet.")
        return 1
    plot_heatmap(gnn, PHASE_4 / "heatmap.png")
    plot_inverted_u(gnn, PHASE_4 / "inverted_u.png")

    summary = (
        gnn.groupby(["L", "N"])
        .agg(n_seeds=("seed", "count"),
             final_mean=("final", "mean"),
             final_std=("final", "std"))
        .reset_index()
    )
    summary.to_csv(PHASE_4 / "summary.csv", index=False)
    print(f"  -> {PHASE_4 / 'summary.csv'}")
    print()
    print(summary.to_string(index=False))
    if not qmix.empty:
        print()
        print("QMIX reference (no GNN, locked Phase 2A config):")
        qmix_summary = (
            qmix.groupby("N")["final"].agg(["mean", "std", "count"]).reset_index()
        )
        print(qmix_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
