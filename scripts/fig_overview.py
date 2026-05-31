"""Figure 1 — design/overview schematic.

Two panels:
  (A) CoordGrid: agents on a grid wired by a tunable coordination graph;
      reward = number of graph edges whose endpoints are co-located.
  (B) The controlled comparison. Three pipelines sharing an identical
      encoder + monotonic mixer, differing only in the block inserted
      before the Q-head:
        QMIX      : (nothing)            ~16k params, no graph
        MLP-QMIX  : L-layer MLP          ~24.5k params, NO graph  (control)
        GNN-QMIX  : L-layer GCN on graph ~24.5k params, +graph
      GNN-QMIX vs MLP-QMIX isolates the graph (matched params);
      MLP-QMIX vs QMIX isolates the capacity.

Output: paper/figures/fig1_overview.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures" / "fig1_overview.png"

C_ENC = "#cfe8ff"      # encoder (shared)
C_MIX = "#d9d9d9"      # mixer (shared)
C_GNN = "#f4a3a3"      # GCN block (graph)
C_MLP = "#c9b3e6"      # MLP block (no graph)
C_NONE = "#f2f2f2"


def box(ax, x, y, w, h, text, fc, fs=8.5, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                                fc=fc, ec="black", lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.0, color="black", zorder=1))


def panel_env(ax):
    ax.set_title("(A) CoordGrid: reward = co-located graph edges", fontsize=10)
    # light grid
    for k in range(6):
        ax.plot([0, 5], [k, k], color="#dddddd", lw=0.6, zorder=0)
        ax.plot([k, k], [0, 5], color="#dddddd", lw=0.6, zorder=0)
    # 4 agents on a ring (positions on grid), ring edges between consecutive
    pos = {0: (1, 4), 1: (4, 3.5), 2: (3.5, 1), 3: (1, 1)}
    ring = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for a, b in ring:
        xa, ya = pos[a]; xb, yb = pos[b]
        ax.plot([xa, xb], [ya, yb], color="#1f77b4", lw=1.6, zorder=1)
    # one co-located pair (agents 2,3 -> reward edge) shown as a star
    for a, (x, y) in pos.items():
        ax.scatter([x], [y], s=420, color="#1f77b4", edgecolor="black", zorder=3)
        ax.text(x, y, f"$a_{a}$", color="white", ha="center", va="center",
                fontsize=8.5, zorder=4)
    ax.text(2.5, -0.7,
            "graph $\\mathcal{G}$ tunable: ring / line / complete / Erdős–Rényi",
            ha="center", fontsize=8)
    ax.set_xlim(-0.4, 5.4); ax.set_ylim(-1.1, 5.4)
    ax.set_aspect("equal"); ax.axis("off")


def panel_arch(ax):
    ax.set_title("(B) One controlled difference: the block before the Q-head",
                 fontsize=10)
    ax.set_xlim(0, 12); ax.set_ylim(0, 9.2); ax.axis("off")

    rows = [
        ("QMIX",     None,             C_NONE, "~16k params · no graph", 6.9),
        ("MLP-QMIX", "MLP $\\times L$\n(no graph)", C_MLP,
         "~24.5k · +capacity, no graph", 3.9),
        ("GNN-QMIX", "GCN $\\times L$\n(over $\\mathcal{G}$)", C_GNN,
         "~24.5k · +capacity, +graph", 0.9),
    ]
    # column x-positions
    x_obs, x_enc, x_blk, x_q, x_mix = 0.2, 2.0, 4.4, 6.8, 9.0
    for name, blk, blkc, note, y in rows:
        ax.text(x_obs + 0.5, y + 1.0, name, fontsize=9.5, fontweight="bold", ha="center")
        box(ax, x_obs, y, 1.0, 0.9, "$o_i$", C_NONE, fs=9)
        box(ax, x_enc, y, 1.7, 0.9, "shared\nencoder", C_ENC)
        arrow(ax, x_obs + 1.0, y + 0.45, x_enc, y + 0.45)
        if blk is None:
            arrow(ax, x_enc + 1.7, y + 0.45, x_q, y + 0.45)
            ax.text((x_enc + 1.7 + x_q) / 2, y + 0.75, "(identity)", fontsize=7,
                    ha="center", color="#777")
        else:
            box(ax, x_blk, y, 1.9, 0.9, blk, blkc)
            arrow(ax, x_enc + 1.7, y + 0.45, x_blk, y + 0.45)
            arrow(ax, x_blk + 1.9, y + 0.45, x_q, y + 0.45)
        box(ax, x_q, y, 1.4, 0.9, "Q-head", C_NONE)
        arrow(ax, x_q + 1.4, y + 0.45, x_mix, y + 0.45)
        box(ax, x_mix, y, 2.6, 0.9, "monotonic\nmixer (shared)", C_MIX)
        ax.text(x_mix + 1.3, y - 0.5, note, fontsize=7.2, ha="center", color="#444")

    # brackets calling out the two contrasts
    ax.annotate("", xy=(11.75, 1.35), xytext=(11.75, 4.35),
                arrowprops=dict(arrowstyle="<->", color="#b22222", lw=1.4))
    ax.text(11.95, 2.85, "graph\n(matched\nparams)", fontsize=7.4, color="#b22222",
            rotation=0, va="center")
    ax.annotate("", xy=(11.75, 4.35), xytext=(11.75, 7.35),
                arrowprops=dict(arrowstyle="<->", color="#1a7d1a", lw=1.4))
    ax.text(11.95, 5.85, "capacity", fontsize=7.4, color="#1a7d1a", va="center")


def main():
    fig = plt.figure(figsize=(11.2, 3.7), dpi=160)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.5], wspace=0.05)
    panel_env(fig.add_subplot(gs[0]))
    panel_arch(fig.add_subplot(gs[1]))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
