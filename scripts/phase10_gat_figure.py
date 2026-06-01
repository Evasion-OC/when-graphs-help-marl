"""Phase 10 figure — the graph penalty is not GCN-specific.

Grouped bars on CoordGrid ring at N=4 and N=8 for QMIX / MLP-QMIX /
GNN-QMIX / GAT-QMIX. The visual point: GAT-QMIX (learned attention) sits
ABOVE GNN-QMIX (attention helps over fixed GCN aggregation) but still well
BELOW the parameter-matched no-graph control MLP-QMIX --- so even the
attention mechanism the critiqued methods use is a net liability.

Reads results/phase8/<algo>__N<n>__ring__seed* and writes
paper/figures/phase10_gat.pdf.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase8"
OUT = ROOT / "paper" / "figures" / "phase10_gat.pdf"

ALGOS = ["qmix", "mlp_qmix", "gnn_qmix", "gat_qmix"]
LABELS = {"qmix": "QMIX\n(no cap,\nno graph)", "mlp_qmix": "MLP-QMIX\n(cap,\nno graph)",
          "gnn_qmix": "GNN-QMIX\n(GCN)", "gat_qmix": "GAT-QMIX\n(attention)"}
COLORS = {"qmix": "#2ca02c", "mlp_qmix": "#9467bd", "gnn_qmix": "#d62728", "gat_qmix": "#ff7f0e"}
NS = [4, 8]


def final_window(path: Path, frac: float = 0.10) -> float:
    r = pd.read_csv(path)["return"].to_numpy(float)
    k = max(1, int(len(r) * frac))
    return float(r[-k:].mean())


def cell(algo: str, n: int) -> np.ndarray:
    return np.array([final_window(Path(d) / "episodes.csv")
                     for d in sorted(glob.glob(str(RES / f"{algo}__N{n}__ring__seed*")))
                     if (Path(d) / "episodes.csv").exists()])


def ci(v: np.ndarray):
    m = float(v.mean())
    if len(v) < 2:
        return m, 0.0
    return m, float(v.std(ddof=1) / np.sqrt(len(v)) * stats.t.ppf(0.975, len(v) - 1))


def main() -> None:
    fig, axes = plt.subplots(1, len(NS), figsize=(8.6, 3.9), dpi=150)
    n_seeds = None
    for ax, n in zip(axes, NS):
        for i, algo in enumerate(ALGOS):
            v = cell(algo, n)
            if v.size == 0:
                continue
            n_seeds = len(v)
            m, half = ci(v)
            ax.bar(i, m, color=COLORS[algo], width=0.66, yerr=half, capsize=5,
                   edgecolor="black", linewidth=0.6)
            ax.scatter([i] * len(v), v, color="black", s=11, zorder=3, alpha=0.55)
        ax.set_xticks(range(len(ALGOS)))
        ax.set_xticklabels([LABELS[a] for a in ALGOS], fontsize=7.5)
        ax.set_title(f"$N={n}$ (ring)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Final-window return")
    seedtxt = f"{n_seeds} seeds" if n_seeds else "seeds"
    fig.suptitle(f"The graph penalty is not GCN-specific: GAT-QMIX (attention) "
                 f"beats GCN but still loses to no-graph ({seedtxt}, 95\\% CIs)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}  (n_seeds={n_seeds})")
    plt.close(fig)


if __name__ == "__main__":
    main()
