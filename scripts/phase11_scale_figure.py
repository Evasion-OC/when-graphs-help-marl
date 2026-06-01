"""Phase 11 figure — the graph penalty explodes with team size.

Pulls final-window return for qmix / mlp_qmix / gnn_qmix at N in {4,8}
(from results/phase8, the headline ring cells) and N in {12,16} (from
results/phase11_scale), and plots:

  (left)  final-window return vs N for the three algorithms (the GNN curve
          flatlines while the graph-free curves keep climbing);
  (right) the graph penalty (MLP-QMIX minus GNN-QMIX) vs N, growing
          monotonically: 10.6 -> 86.6 -> 153.4 -> 267.4.

Writes paper/figures/phase11_scaling.pdf.
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
OUT = ROOT / "paper" / "figures" / "phase11_scaling.pdf"
# Where each N lives (same env config / protocol throughout).
SOURCES = {4: ROOT / "results" / "phase8", 8: ROOT / "results" / "phase8",
           12: ROOT / "results" / "phase11_scale", 16: ROOT / "results" / "phase11_scale"}
NS = [4, 8, 12, 16]
ALGOS = ["qmix", "mlp_qmix", "gnn_qmix"]
LABELS = {"qmix": "QMIX (no cap, no graph)", "mlp_qmix": "MLP-QMIX (cap, no graph)",
          "gnn_qmix": "GNN-QMIX (cap + graph)"}
COLORS = {"qmix": "#2ca02c", "mlp_qmix": "#9467bd", "gnn_qmix": "#d62728"}


def final_window(path: Path, frac: float = 0.10) -> float:
    r = pd.read_csv(path)["return"].to_numpy(float)
    k = max(1, int(len(r) * frac))
    return float(r[-k:].mean())


def cell(algo: str, n: int) -> np.ndarray:
    base = SOURCES[n]
    vals = []
    for d in sorted(glob.glob(str(base / f"{algo}__N{n}__ring__seed*"))):
        f = Path(d) / "episodes.csv"
        if f.exists():
            vals.append(final_window(f))
    return np.array(vals)


def mean_ci(v: np.ndarray):
    m = float(v.mean())
    if len(v) < 2:
        return m, 0.0
    half = v.std(ddof=1) / np.sqrt(len(v)) * stats.t.ppf(0.975, len(v) - 1)
    return m, float(half)


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=150)

    # Left: return vs N
    for algo in ALGOS:
        ms, hs = [], []
        for n in NS:
            m, h = mean_ci(cell(algo, n))
            ms.append(m); hs.append(h)
        ax1.errorbar(NS, ms, yerr=hs, marker="o", capsize=4, lw=1.8,
                     color=COLORS[algo], label=LABELS[algo])
    ax1.set_xlabel("team size $N$")
    ax1.set_ylabel("final-window return")
    ax1.set_title("Returns diverge as $N$ grows")
    ax1.set_xticks(NS)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax1.grid(alpha=0.3)

    # Right: graph penalty (MLP - GNN) vs N
    pen, penlo, penhi = [], [], []
    for n in NS:
        g, m = cell("gnn_qmix", n), cell("mlp_qmix", n)
        diff = m.mean() - g.mean()
        # CI on the difference (independent samples, Welch-ish half-width)
        sd = np.sqrt(m.var(ddof=1) / len(m) + g.var(ddof=1) / len(g))
        half = sd * 1.96
        pen.append(diff); penlo.append(diff - half); penhi.append(diff + half)
    ax2.plot(NS, pen, marker="s", color="#b22222", lw=2.0)
    ax2.fill_between(NS, penlo, penhi, color="#b22222", alpha=0.15)
    for x, y in zip(NS, pen):
        ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
    ax2.set_xlabel("team size $N$")
    ax2.set_ylabel("graph penalty  (MLP-QMIX $-$ GNN-QMIX)")
    ax2.set_title("The penalty grows monotonically with $N$")
    ax2.set_xticks(NS)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
