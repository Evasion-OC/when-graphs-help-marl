"""Plotting helpers for the empirical study.

All figures are matplotlib-only and write to PDF + PNG side-by-side so the
paper can pull the PDF (vector) while the README can show the PNG. We keep a
single styling pass so figures look consistent across phases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

ALGO_COLORS = {
    "iql": "#7f7f7f",       # grey — no coordination
    "vdn": "#1f77b4",       # blue
    "qmix": "#2ca02c",      # green
    "gnn_qmix": "#d62728",  # red — method under test
}
ALGO_LABELS = {
    "iql": "IQL",
    "vdn": "VDN",
    "qmix": "QMIX",
    "gnn_qmix": "GNN-QMIX",
}


def _save(fig: plt.Figure, out_stem: Path) -> tuple[Path, Path]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = out_stem.with_suffix(".pdf")
    png = out_stem.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------


def learning_curves(
    returns_by_algo_seed: dict[str, list[np.ndarray]],
    out_stem: Path,
    *,
    window: int = 25,
    title: str | None = None,
    xlabel: str = "Episode",
    ylabel: str = "Episode return",
) -> tuple[Path, Path]:
    """Mean ± std learning curves across seeds, one line per algo.

    ``returns_by_algo_seed[algo]`` is a list of per-seed 1-D arrays. We
    truncate to the shortest seed run so the mean is well-defined.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for algo, seed_runs in returns_by_algo_seed.items():
        if not seed_runs:
            continue
        min_len = min(len(r) for r in seed_runs)
        if min_len < window:
            continue
        stacked = np.stack([r[:min_len] for r in seed_runs], axis=0)  # [n_seeds, T]
        # Rolling mean per seed (so the std reflects across-seed variance of
        # the smoothed curve — which is what readers care about).
        csum = np.concatenate(
            [np.zeros((stacked.shape[0], 1)), np.cumsum(stacked, axis=1)], axis=1
        )
        rolling = (csum[:, window:] - csum[:, :-window]) / window  # [n_seeds, T-w+1]
        x = np.arange(window - 1, min_len)
        mean = rolling.mean(axis=0)
        sd = rolling.std(axis=0, ddof=1) if rolling.shape[0] > 1 else np.zeros_like(mean)
        color = ALGO_COLORS.get(algo, None)
        ax.plot(x, mean, label=ALGO_LABELS.get(algo, algo), color=color, lw=1.5)
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    return _save(fig, out_stem)


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------


def forest_plot(
    rows: Sequence[tuple[str, float, float, float]],
    out_stem: Path,
    *,
    title: str | None = None,
    xlabel: str = "Episodes to 80% of best",
) -> tuple[Path, Path]:
    """Forest plot: one row per (algo, condition), each with mean and CI.

    ``rows`` is a list of ``(label, mean, lo, hi)`` tuples; the label is
    shown verbatim on the y-axis.
    """
    n = len(rows)
    fig, ax = plt.subplots(figsize=(5.5, max(2.0, 0.35 * n + 1.0)))
    ys = np.arange(n)[::-1]  # top-to-bottom
    for y, (label, mean, lo, hi) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color="black", lw=1.2)
        ax.plot([mean], [y], marker="o", color="black", markersize=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    return _save(fig, out_stem)


# ---------------------------------------------------------------------------
# Heatmap (Phase 4: depth × diameter)
# ---------------------------------------------------------------------------


def heatmap(
    matrix: np.ndarray,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    out_stem: Path,
    *,
    title: str | None = None,
    xlabel: str = "",
    ylabel: str = "",
    cbar_label: str = "",
    cmap: str = "viridis",
    annotate: bool = True,
) -> tuple[Path, Path]:
    """Annotated heatmap, e.g. final return as a function of (depth, diameter)."""
    if matrix.ndim != 2:
        raise ValueError(f"heatmap expects 2-D matrix; got shape {matrix.shape}")
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    if annotate:
        vmin, vmax = float(np.nanmin(matrix)), float(np.nanmax(matrix))
        midpoint = (vmin + vmax) / 2.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if np.isnan(v):
                    continue
                color = "white" if v < midpoint else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return _save(fig, out_stem)
