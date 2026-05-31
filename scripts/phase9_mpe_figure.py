"""Phase 9 MPE capacity-control figure for the paper.

Grouped bars: QMIX / MLP-QMIX / GNN-QMIX at N=3 and N=6 on
``simple_spread``, with 95% Student-t CIs over seeds. Parallel to
``phase8_figure.py`` but on the MPE benchmark. simple_spread returns are
negative (distance penalties), so bars extend downward from 0 and we do
NOT clamp the y-axis at zero; less-negative (taller toward 0) is better.

Reads results/phase9_mpe/<algo>__N<n>__spread__seed*/episodes.csv and
writes paper/figures/phase9_mpe_capacity.pdf. Re-run after the sweep
finishes.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase9_mpe"
OUT = ROOT / "paper" / "figures" / "phase9_mpe_capacity.pdf"

ALGOS = ["qmix", "mlp_qmix", "gnn_qmix"]
LABELS = {"qmix": "QMIX\n(no cap, no graph)",
          "mlp_qmix": "MLP-QMIX\n(cap, no graph)",
          "gnn_qmix": "GNN-QMIX\n(cap + graph)"}
COLORS = {"qmix": "#2ca02c", "mlp_qmix": "#9467bd", "gnn_qmix": "#d62728"}
NS = [3, 6]


def final_window(path: Path, frac: float = 0.10) -> float:
    r = pd.read_csv(path)["return"].to_numpy(float)
    k = max(1, int(len(r) * frac))
    return float(r[-k:].mean())


def cell(algo: str, n: int) -> np.ndarray:
    vals = []
    for d in sorted(glob.glob(str(RES / f"{algo}__N{n}__spread__seed*"))):
        f = Path(d) / "episodes.csv"
        if f.exists():
            vals.append(final_window(f))
    return np.array(vals)


def ci(vals: np.ndarray):
    m = float(vals.mean())
    if len(vals) < 2:
        return m, 0.0
    sem = vals.std(ddof=1) / np.sqrt(len(vals))
    half = sem * stats.t.ppf(0.975, len(vals) - 1)
    return m, float(half)


def main() -> None:
    fig, axes = plt.subplots(1, len(NS), figsize=(8.4, 3.8), dpi=150, sharey=False)
    n_seeds = None
    for ax, n in zip(axes, NS):
        xs = np.arange(len(ALGOS))
        for i, algo in enumerate(ALGOS):
            vals = cell(algo, n)
            if vals.size == 0:
                continue
            n_seeds = len(vals)
            m, half = ci(vals)
            ax.bar(i, m, color=COLORS[algo], width=0.62,
                   yerr=half, capsize=5, edgecolor="black", linewidth=0.6)
            ax.scatter([i] * len(vals), vals, color="black", s=12, zorder=3, alpha=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[a] for a in ALGOS], fontsize=8)
        ax.set_title(f"$N={n}$ (simple\\_spread)", fontsize=11)
        ax.axhline(0, color="black", lw=0.6)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Final-window return\n(higher = better)")
    seedtxt = f"{n_seeds} seeds" if n_seeds else "seeds"
    fig.suptitle(f"External validity (MPE simple\\_spread): the capacity "
                 f"control transfers ({seedtxt}, 95\\% CIs)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}  (n_seeds={n_seeds})")


if __name__ == "__main__":
    main()
