"""paper_teaser_figures.py — two standalone publication figures for the TMLR paper.

Writes two vector PDFs to paper/figures/:
  phaseC_tokenmatch.pdf   — standalone 4-arm TokenMatch out-of-harness bar chart
                            (currently only a panel inside phaseC_scaling_replication.pdf)
  teaser_boundary.pdf     — compact two-panel "both endpoints at a glance" teaser:
                            (A) negative endpoint (full-info CoordGrid capacity control,
                                Phase 8) and (B) positive endpoint (pair-routing structure
                                result, Stage C)

Both figures follow the visual conventions established in scripts/phaseC_figures.py
(Okabe-Ito colorblind-safe palette, embedded TrueType fonts, despined axes, CI
whiskers) so that they read as siblings of paper/figures/phaseC_structure.pdf and
paper/figures/phaseC_scaling_replication.pdf.

Input data (read directly; no numbers are re-typed from docs/):
  results/phaseC/summary_token_N6.csv              — Figure 1 (TokenMatch, 12 seeds, 30k steps)
  results/phaseC/summary_pair_N6_ego.csv            — Figure 2 panel B (pair-routing, 12 seeds, 30k steps)
  results/phase8/summary_aggregate.csv              — Figure 2 panel A (CoordGrid ring, 10 seeds, 150k steps)

Every number plotted was independently reconciled against results/phase8/*/episodes.csv
and results/phaseD/token_N6/*/episodes.csv (see task notes); this script itself reads
only the pre-aggregated summary CSVs, matching the "no re-computation from episodes at
plot time" convention of phaseC_figures.py.

Determinism: SOURCE_DATE_EPOCH is fixed before matplotlib is imported so the PDF
CreationDate is constant across runs, and an explicit (empty) metadata dict is passed
to savefig so no author/username is embedded.

Run:
    python scripts/paper_teaser_figures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Fix the embedded PDF CreationDate so repeated runs are byte-for-byte reproducible.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (not Type 3) — required for archival
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.labelsize"] = 9
matplotlib.rcParams["axes.titlesize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 8.5
matplotlib.rcParams["ytick.labelsize"] = 8.5
matplotlib.rcParams["legend.fontsize"] = 8.0
matplotlib.rcParams["figure.dpi"] = 150

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA_C = REPO / "results" / "phaseC"
DATA_8 = REPO / "results" / "phase8"
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# No author/username in the PDF metadata; a blank dict overrides matplotlib's defaults
# (which do not include Author, but we set it explicitly per the anonymity requirement).
PDF_METADATA = {"Author": "", "Title": "", "Subject": "", "Keywords": "", "Creator": "matplotlib"}

# ---------------------------------------------------------------------------
# Okabe-Ito colorblind-safe palette — identical to scripts/phaseC_figures.py
# ---------------------------------------------------------------------------
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Semantic color assignments — consistent with phaseC_structure.pdf / phaseC_scaling_replication.pdf
C_TASK_MATCHED = OI["vermillion"]  # reserved for the task-matched / "structure helps" bars
C_FLOOR_BASE = "#aaaaaa"  # no-graph / wrong-graph / all-to-all "floor" arms
C_NEG_GRAPH = OI["reddish_purple"]  # distinct hue: graph-conditioned arm that UNDERPERFORMS
# (Deliberately NOT vermillion: vermillion is reserved paper-wide for "the graph that
#  helps because it matches the task." GNN-QMIX in the full-info negative panel is the
#  opposite story (graph added where it is not needed hurts), so re-using vermillion
#  there would contradict the established color semantics across figures.)

HATCH_GCN = ""  # solid — GCN aggregator (the only aggregator used in both source studies)


def load(path: Path) -> pd.DataFrame:
    """Load a phaseC-style summary CSV (columns: arm, n, mean, ci_lo, ci_hi)."""
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    for col in ("arm", "n", "mean", "ci_lo", "ci_hi"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in {path.name}")
    return df


def load_phase8(path: Path) -> pd.DataFrame:
    """Load results/phase8/summary_aggregate.csv and normalize to the phaseC schema.

    Native columns: algo, n_agents, n_seeds, final_mean, ci_lo, ci_hi.
    Renamed here to: arm, n, mean, ci_lo, ci_hi (n_agents kept for grouping).
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    for col in ("algo", "n_agents", "n_seeds", "final_mean", "ci_lo", "ci_hi"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in {path.name}")
    return df.rename(columns={"algo": "arm", "n_seeds": "n", "final_mean": "mean"})


def get_row(df: pd.DataFrame, arm: str) -> pd.Series:
    rows = df[df["arm"] == arm]
    if len(rows) == 0:
        raise KeyError(f"arm='{arm}' not found (arms: {list(df['arm'])}) in given DataFrame")
    return rows.iloc[0]


def yerr(row: pd.Series) -> tuple[float, float]:
    return float(row["mean"] - row["ci_lo"]), float(row["ci_hi"] - row["mean"])


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def check(name: str, actual: float, expected: float, tol: float = 0.02) -> None:
    if abs(actual - expected) > tol:
        raise SystemExit(
            f"DISCREPANCY: {name} mean={actual:.4f} but reference says {expected:.2f} "
            f"(diff={actual - expected:+.4f}). Stopping — do not plot doc numbers."
        )
    print(f"  OK {name}: {actual:.4f} ~ {expected:.2f}")


# ---------------------------------------------------------------------------
# Figure 1 — standalone TokenMatch out-of-harness replication
# ---------------------------------------------------------------------------


def fig_tokenmatch() -> Path:
    """Four-arm bar chart: TokenMatch (N=6, K=5 tokens, 12 seeds, 30k steps).

    Data: results/phaseC/summary_token_N6.csv (verified byte-for-byte against
    results/phaseC/manifest_token_N6.csv -> results/phaseD/token_N6/*/episodes.csv,
    final-window fraction 0.2 as used throughout phaseC_analysis.py).
    """
    df = load(DATA_C / "summary_token_N6.csv")

    r_mlp = get_row(df, "mlp")
    r_wrong = get_row(df, "gnn_wrong")
    r_complete = get_row(df, "gnn_complete")
    r_true = get_row(df, "gnn_true")

    # Reference numbers from docs/STAGE_C_FINDINGS.md section 7 — verify, don't trust.
    check("mlp", float(r_mlp["mean"]), 11.90)
    check("gnn_wrong", float(r_wrong["mean"]), 12.03)
    check("gnn_complete", float(r_complete["mean"]), 24.50)
    check("gnn_true", float(r_true["mean"]), 57.59)

    arms = ["No graph\n(MLP)", "Wrong\ngraph", "All-to-all\ngraph", "Task-matched\ngraph"]
    rows = [r_mlp, r_wrong, r_complete, r_true]
    colors = [C_FLOOR_BASE, C_FLOOR_BASE, C_FLOOR_BASE, C_TASK_MATCHED]
    edge_colors = ["#666666", "#666666", "#666666", "#222222"]

    bar_w = 0.55
    grp_gap = 0.5
    # Same group-gap layout as phaseC_structure.pdf; an extra gap sets the
    # task-matched bar visually apart from the three floor controls.
    xs = [0.0, bar_w + grp_gap, 2 * (bar_w + grp_gap), 3 * (bar_w + grp_gap) + grp_gap * 0.6]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    for x, row, col, ec in zip(xs, rows, colors, edge_colors):
        m = float(row["mean"])
        elo, ehi = yerr(row)
        ax.bar(
            x,
            m,
            width=bar_w,
            color=col,
            hatch=HATCH_GCN,
            edgecolor=ec,
            linewidth=0.8,
            yerr=[[elo], [ehi]],
            capsize=3.5,
            error_kw=dict(elinewidth=1.0, ecolor="#333333", capthick=1.0),
            zorder=3,
        )

    # Task maximum reference line (dashed).
    task_max = 60.0
    ax.axhline(task_max, color="#444444", linestyle="--", linewidth=1.0, zorder=2)
    ax.text(
        xs[-1] + bar_w / 2 + 0.05,
        task_max,
        "task maximum (60)",
        fontsize=7.5,
        color="#444444",
        va="center",
        ha="left",
        zorder=4,
        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"),
    )

    # Random floor reference line (dotted), anchored at the no-graph control mean.
    floor_mean = float(r_mlp["mean"])
    ax.axhline(floor_mean, color="#888888", linestyle=":", linewidth=1.0, zorder=2)
    ax.text(
        xs[-1] + bar_w / 2 + 0.05,
        floor_mean,
        "random floor (~12)",
        fontsize=7.5,
        color="#888888",
        va="center",
        ha="left",
        zorder=4,
        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"),
    )

    # Near-solve annotation on the task-matched bar.
    true_mean = float(r_true["mean"])
    frac_of_max = 100.0 * true_mean / task_max
    ax.annotate(
        f"{true_mean:.1f} / {task_max:.0f} $\\approx$ {frac_of_max:.0f}%\n(near-solves)",
        xy=(xs[-1], true_mean),
        xytext=(xs[-1] - 1.35, true_mean - 14.0),
        fontsize=7.5,
        color="#222222",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#222222", lw=0.8),
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(arms, fontsize=8.5)
    ax.set_ylim(0, 68)
    ax.set_xlim(xs[0] - bar_w, xs[-1] + bar_w * 2.6)
    ax.set_ylabel("Final-window mean return (95% CI)", fontsize=9)
    ax.set_title(
        "TokenMatch out-of-harness replication\n($N{=}6$, 12 seeds, 30k steps)", fontsize=9.5
    )

    despine(ax)
    fig.tight_layout()

    out_path = OUT / "phaseC_tokenmatch.pdf"
    fig.savefig(out_path, bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)
    print(f"Wrote {out_path}  (figsize=5.0x3.5)")
    return out_path


# ---------------------------------------------------------------------------
# Figure 2 — compact two-panel teaser: both boundary endpoints at a glance
# ---------------------------------------------------------------------------


def fig_teaser_boundary() -> Path:
    """Panel A: negative endpoint (full-info CoordGrid capacity control, Phase 8).
    Panel B: positive endpoint (pair-routing structure result, Stage C).

    Data:
      Panel A: results/phase8/summary_aggregate.csv   (n=10 seeds, 150k steps)
      Panel B: results/phaseC/summary_pair_N6_ego.csv  (n=12 seeds, 30k steps)
    """
    d8 = load_phase8(DATA_8 / "summary_aggregate.csv")
    dC = load(DATA_C / "summary_pair_N6_ego.csv")

    # ---- Panel A data: qmix / mlp_qmix / gnn_qmix at N=4, N=8 ----
    algos_a = ["qmix", "mlp_qmix", "gnn_qmix"]
    colors_a = [C_FLOOR_BASE, "#888888", C_NEG_GRAPH]
    edge_a = ["#666666", "#555555", "#222222"]

    rows_a = {}
    for n_agents in (4, 8):
        sub = d8[d8["n_agents"] == n_agents]
        rows_a[n_agents] = {a: get_row(sub, a) for a in algos_a}

    # Reference numbers from the task spec (means: N4 83.0/83.4/72.8, N8 164.9/164.9/78.3).
    check("qmix N4", float(rows_a[4]["qmix"]["mean"]), 83.0, tol=0.1)
    check("mlp_qmix N4", float(rows_a[4]["mlp_qmix"]["mean"]), 83.4, tol=0.1)
    check("gnn_qmix N4", float(rows_a[4]["gnn_qmix"]["mean"]), 72.8, tol=0.1)
    check("qmix N8", float(rows_a[8]["qmix"]["mean"]), 164.9, tol=0.1)
    check("mlp_qmix N8", float(rows_a[8]["mlp_qmix"]["mean"]), 164.9, tol=0.1)
    check("gnn_qmix N8", float(rows_a[8]["gnn_qmix"]["mean"]), 78.3, tol=0.1)

    # ---- Panel B data: mlp / gnn_wrong / gnn_complete / gnn_true ----
    arms_b = ["mlp", "gnn_wrong", "gnn_complete", "gnn_true"]
    labels_b = ["No graph\n(MLP)", "Wrong\ngraph", "All-to-all\ngraph", "Task-matched\ngraph"]
    colors_b = [C_FLOOR_BASE, C_FLOOR_BASE, C_FLOOR_BASE, C_TASK_MATCHED]
    edge_b = ["#666666", "#666666", "#666666", "#222222"]
    rows_b = {a: get_row(dC, a) for a in arms_b}

    check("mlp (pair N6 ego)", float(rows_b["mlp"]["mean"]), 49.85)
    check("gnn_wrong (pair N6 ego)", float(rows_b["gnn_wrong"]["mean"]), 50.42)
    check("gnn_complete (pair N6 ego)", float(rows_b["gnn_complete"]["mean"]), 50.94)
    check("gnn_true (pair N6 ego)", float(rows_b["gnn_true"]["mean"]), 66.10)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.5, 2.6))

    # ---------------- Panel A ----------------
    bar_w = 0.24
    group_gap = 0.35
    ns = [4, 8]
    group_centers = []
    x0 = 0.0
    for n_agents in ns:
        xs = [x0 + i * bar_w for i in range(len(algos_a))]
        group_centers.append(xs[1])
        for x, algo, col, ec in zip(xs, algos_a, colors_a, edge_a):
            row = rows_a[n_agents][algo]
            m = float(row["mean"])
            elo, ehi = yerr(row)
            ax_a.bar(
                x,
                m,
                width=bar_w * 0.92,
                color=col,
                edgecolor=ec,
                linewidth=0.7,
                yerr=[[elo], [ehi]],
                capsize=2.5,
                error_kw=dict(elinewidth=0.9, ecolor="#333333", capthick=0.9),
                zorder=3,
            )
        x0 = xs[-1] + bar_w + group_gap

    ax_a.set_xticks(group_centers)
    ax_a.set_xticklabels([f"$N={n}$" for n in ns], fontsize=8.5)
    # Anchored at 0 -- never truncate the axis to exaggerate a gap.
    ax_a.set_ylim(0, 185)
    ax_a.set_ylabel("Final-window\nmean return", fontsize=8)
    ax_a.set_title("(A) Full information: graph hurts", fontsize=9)
    # Budget/metric footnote placed below the x-axis (own space, never overlaps bars/legend).
    ax_a.text(
        0.5,
        -0.34,
        "CoordGrid ring, 10 seeds/arm, 150k steps (Phase 8)",
        transform=ax_a.transAxes,
        fontsize=6.3,
        color="#555555",
        ha="center",
        va="top",
    )
    despine(ax_a)

    from matplotlib.patches import Patch

    # Kept short (bare algorithm names) so the legend box stays narrow and clear of the
    # N=8 bars; the capacity/graph contrast is spelled out in the caption.
    legend_a = [
        Patch(facecolor=C_FLOOR_BASE, edgecolor="#666666", label="QMIX"),
        Patch(facecolor="#888888", edgecolor="#555555", label="MLP-QMIX"),
        Patch(facecolor=C_NEG_GRAPH, edgecolor="#222222", label="GNN-QMIX"),
    ]
    ax_a.legend(
        handles=legend_a,
        loc="upper left",
        fontsize=7.2,
        framealpha=0.9,
        edgecolor="#cccccc",
        handlelength=1.0,
        handleheight=1.0,
        borderpad=0.3,
        labelspacing=0.3,
    )

    # ---------------- Panel B ----------------
    bar_w_b = 0.55
    grp_gap_b = 0.35
    xs_b = [i * (bar_w_b + grp_gap_b) for i in range(len(arms_b))]
    for x, arm, col, ec in zip(xs_b, arms_b, colors_b, edge_b):
        row = rows_b[arm]
        m = float(row["mean"])
        elo, ehi = yerr(row)
        ax_b.bar(
            x,
            m,
            width=bar_w_b,
            color=col,
            edgecolor=ec,
            linewidth=0.7,
            yerr=[[elo], [ehi]],
            capsize=2.5,
            error_kw=dict(elinewidth=0.9, ecolor="#333333", capthick=0.9),
            zorder=3,
        )

    floor_mean_b = float(rows_b["mlp"]["mean"])
    ax_b.axhline(floor_mean_b, color="#444444", linestyle="--", linewidth=0.9, zorder=2)
    ax_b.text(
        xs_b[0] - bar_w_b / 2,
        floor_mean_b + 2.0,
        "random floor (~50)",
        fontsize=6.0,
        color="#444444",
        ha="left",
        va="bottom",
    )

    ax_b.set_xticks(xs_b)
    ax_b.set_xticklabels(labels_b, fontsize=7.0)
    ax_b.set_ylim(0, 78)
    ax_b.set_ylabel("Final-window\nmean return", fontsize=8)
    ax_b.set_title("(B) Private partner goals:\nonly the task-matched graph rises", fontsize=9)
    # Budget/metric footnote placed below the x-axis, matching panel A's convention.
    ax_b.text(
        0.5,
        -0.34,
        "Pair-routing, $N{=}6$, ego, 12 seeds/arm, 30k steps (Stage C)",
        transform=ax_b.transAxes,
        fontsize=6.3,
        color="#555555",
        ha="center",
        va="top",
    )
    despine(ax_b)

    fig.tight_layout()

    out_path = OUT / "teaser_boundary.pdf"
    fig.savefig(out_path, bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)
    print(f"Wrote {out_path}  (figsize=6.5x2.6)")
    return out_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== paper_teaser_figures.py: verifying and plotting ===\n")

    print("--- Figure 1: TokenMatch (standalone) ---")
    p1 = fig_tokenmatch()

    print("\n--- Figure 2: teaser (both boundary endpoints) ---")
    p2 = fig_teaser_boundary()

    print("\n=== Done. Written: ===")
    for p in [p1, p2]:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
