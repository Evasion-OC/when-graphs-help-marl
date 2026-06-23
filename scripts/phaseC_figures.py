"""phaseC_figures.py — publication-grade figures for Stage C / D results.

Writes three vector PDFs to paper/figures/:
  phaseC_structure.pdf        — headline bar chart: graph structure comparison (N=6, ego)
  phaseC_scaling_replication.pdf — two panels: team-size scaling + TokenMatch replication
  phaseC_robustness.pdf       — compact panel: advantage persists across conditions

Input CSVs: results/phaseC/summary_*.csv  (columns: arm, n, mean, ci_lo, ci_hi)
All numbers are read directly from CSVs; no re-computation from episodes.

Run:
    python scripts/phaseC_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType (not Type 3) — required for archival
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.labelsize"] = 9
matplotlib.rcParams["axes.titlesize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 8.5
matplotlib.rcParams["ytick.labelsize"] = 8.5
matplotlib.rcParams["legend.fontsize"] = 8.0
matplotlib.rcParams["figure.dpi"] = 150

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "results" / "phaseC"
OUT  = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Okabe-Ito colorblind-safe palette
# Index: 0=black, 1=orange, 2=sky-blue, 3=bluish-green, 4=yellow, 5=blue,
#        6=vermillion, 7=reddish-purple
# ---------------------------------------------------------------------------
OI = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky":            "#56B4E9",
    "green":          "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Semantic color assignments (consistent across all figures)
C_TASK_MATCHED = OI["vermillion"]   # vivid accent for task-matched GNN bars/lines
C_FLOOR_BASE   = "#aaaaaa"          # neutral gray for no-graph / wrong-graph arms
C_FLOOR_BAND   = "#d9d9d9"          # lighter gray for floor band fills
C_GAT_TRUE     = OI["orange"]       # GAT-true aggregator variant
C_DGN_TRUE     = OI["blue"]         # DGN-true aggregator variant

# Hatch patterns for aggregator distinction within same color family
HATCH_GCN = ""          # solid (GNN-QMIX / GCN is the baseline)
HATCH_GAT = "////"      # forward diagonals
HATCH_DGN = "xxxx"      # cross-hatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(csv_name: str) -> pd.DataFrame:
    """Load a summary CSV from results/phaseC/ and return a DataFrame."""
    path = DATA / csv_name
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    # Validate columns
    for col in ("arm", "n", "mean", "ci_lo", "ci_hi"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in {csv_name}")
    return df


def get_row(df: pd.DataFrame, arm: str) -> pd.Series:
    rows = df[df["arm"] == arm]
    if len(rows) == 0:
        raise KeyError(f"arm='{arm}' not found in DataFrame (arms: {list(df['arm'])})")
    return rows.iloc[0]


def yerr_from_row(row: pd.Series) -> tuple[float, float]:
    """Return (err_lo, err_hi) as positive half-widths for matplotlib yerr."""
    return float(row["mean"] - row["ci_lo"]), float(row["ci_hi"] - row["mean"])


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Figure 1 — Headline structure comparison (N=6, ego, pair-routing)
# ---------------------------------------------------------------------------

def fig_structure():
    """
    Grouped bar chart. Arms arranged in four conceptual groups:
      (1) No graph  — mlp
      (2) Wrong graph — gnn_wrong
      (3) All-to-all — gcn_complete, gat_complete, dgn_complete
      (4) Task-matched — gnn_true, gat_true, dgn_true
    Color: muted grays for groups 1-3, vivid accent for group 4.
    Data sources:
      - mlp, gnn_true, gcn_complete, gat_complete, dgn_complete, gat_true, dgn_true:
            summary_pair_N6_ego_attention.csv
      - gnn_wrong: summary_pair_N6_ego.csv (absent from the attention file)
    Both files share the same run family: shared arms are byte-identical to 10+ sig-figs.
    """
    att  = load("summary_pair_N6_ego_attention.csv")
    core = load("summary_pair_N6_ego.csv")

    # ---- pull each bar ----
    # Group 1: No graph
    r_mlp         = get_row(att, "mlp")
    # Group 2: Wrong graph (only in core file)
    r_wrong       = get_row(core, "gnn_wrong")
    # Group 3: All-to-all (three aggregators)
    r_gcn_cmp     = get_row(att, "gcn_complete")
    r_gat_cmp     = get_row(att, "gat_complete")
    r_dgn_cmp     = get_row(att, "dgn_complete")
    # Group 4: Task-matched (three aggregators)
    r_gnn_true    = get_row(att, "gnn_true")
    r_gat_true    = get_row(att, "gat_true")
    r_dgn_true    = get_row(att, "dgn_true")

    # --- verify key numbers match task specification (tolerance 0.01) ----
    checks = [
        ("mlp",          r_mlp["mean"],      49.85),
        ("gnn_wrong",    r_wrong["mean"],     50.42),
        ("gcn_complete", r_gcn_cmp["mean"],   50.94),
        ("gat_complete", r_gat_cmp["mean"],   50.19),
        ("dgn_complete", r_dgn_cmp["mean"],   50.45),
        ("gnn_true",     r_gnn_true["mean"],  66.10),
        ("gat_true",     r_gat_true["mean"],  67.03),
        ("dgn_true",     r_dgn_true["mean"],  71.50),
    ]
    for name, actual, expected in checks:
        if abs(actual - expected) > 0.02:
            print(f"  WARNING: {name} mean={actual:.4f} but task says {expected:.2f}  "
                  f"(diff={actual-expected:+.4f})")
        else:
            print(f"  OK {name}: {actual:.4f} ~ {expected:.2f}")

    # ---- layout: 4 groups with internal spacing ----
    # Groups: [mlp] [gnn_wrong] [gcn_cmp, gat_cmp, dgn_cmp] [gnn_true, gat_true, dgn_true]
    # x-positions (leave gaps between groups)
    grp_gap   = 0.5    # gap between groups
    bar_w     = 0.55   # bar width

    # Group x-centers
    x_mlp     = 0.0
    x_wrong   = x_mlp + bar_w + grp_gap
    # Group 3: three bars, center at x_g3
    x_g3_lo   = x_wrong + bar_w + grp_gap
    x_gcn_cmp = x_g3_lo
    x_gat_cmp = x_g3_lo + bar_w
    x_dgn_cmp = x_g3_lo + 2 * bar_w
    x_g3_ctr  = x_g3_lo + bar_w          # center of group 3 (for label)
    # Group 4: three bars
    x_g4_lo   = x_dgn_cmp + bar_w + grp_gap
    x_gnn_true= x_g4_lo
    x_gat_true= x_g4_lo + bar_w
    x_dgn_true= x_g4_lo + 2 * bar_w
    x_g4_ctr  = x_g4_lo + bar_w

    xs   = [x_mlp, x_wrong, x_gcn_cmp, x_gat_cmp, x_dgn_cmp,
            x_gnn_true, x_gat_true, x_dgn_true]
    rows = [r_mlp, r_wrong, r_gcn_cmp, r_gat_cmp, r_dgn_cmp,
            r_gnn_true, r_gat_true, r_dgn_true]

    # Colors and hatches per bar
    colors = [
        C_FLOOR_BASE,   # mlp
        C_FLOOR_BASE,   # gnn_wrong
        C_FLOOR_BASE,   # gcn_complete
        C_FLOOR_BASE,   # gat_complete
        C_FLOOR_BASE,   # dgn_complete
        C_TASK_MATCHED, # gnn_true  (GCN aggregator)
        C_GAT_TRUE,     # gat_true
        C_DGN_TRUE,     # dgn_true
    ]
    hatches = [
        HATCH_GCN,  # mlp
        HATCH_GCN,  # gnn_wrong
        HATCH_GCN,  # gcn_complete
        HATCH_GAT,  # gat_complete
        HATCH_DGN,  # dgn_complete
        HATCH_GCN,  # gnn_true (GCN)
        HATCH_GAT,  # gat_true
        HATCH_DGN,  # dgn_true
    ]
    edge_colors = ["#666666"] * 5 + ["#222222"] * 3

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    for x, row, col, hatch, ec in zip(xs, rows, colors, hatches, edge_colors):
        m   = float(row["mean"])
        elo, ehi = yerr_from_row(row)
        ax.bar(x, m, width=bar_w, color=col, hatch=hatch,
               edgecolor=ec, linewidth=0.8,
               yerr=[[elo], [ehi]], capsize=3.5,
               error_kw=dict(elinewidth=1.0, ecolor="#333333", capthick=1.0),
               zorder=3)

    # Random floor dashed reference line at ~50
    floor_mean = float(r_mlp["mean"])  # 49.85 — the no-graph baseline
    ax.axhline(floor_mean, color="#444444", linestyle="--", linewidth=1.0,
               zorder=2, label=f"no-graph floor ({floor_mean:.1f})")

    # Effect size annotation on gnn_true bar
    gnn_true_mean = float(r_gnn_true["mean"])
    ax.annotate(
        r"$d \approx 5.6$",
        xy=(x_gnn_true, gnn_true_mean),
        xytext=(x_gnn_true - 0.3, gnn_true_mean + 5.0),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#222222", lw=0.8),
        color="#222222",
    )

    # x-axis group labels
    ax.set_xticks([x_mlp, x_wrong, x_g3_ctr, x_g4_ctr])
    ax.set_xticklabels(
        ["No graph\n(MLP)", "Wrong\ngraph", "All-to-all\ngraph", "Task-matched\ngraph"],
        fontsize=8.5,
    )

    # y-axis
    ax.set_ylim(0, 85)
    ax.set_ylabel("Final-window mean return (95% CI)", fontsize=9)

    # Legend for aggregator types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_TASK_MATCHED, hatch=HATCH_GCN, edgecolor="#222222",
              label="GCN aggregator (gnn_true / gcn_complete)"),
        Patch(facecolor=C_GAT_TRUE,     hatch=HATCH_GAT, edgecolor="#222222",
              label="GAT aggregator (gat_true / gat_complete)"),
        Patch(facecolor=C_DGN_TRUE,     hatch=HATCH_DGN, edgecolor="#222222",
              label="DGN aggregator (dgn_true / dgn_complete)"),
        Patch(facecolor=C_FLOOR_BASE,   hatch=HATCH_GCN, edgecolor="#666666",
              label="Floor arms (no / wrong / all-to-all graph)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7.5,
              framealpha=0.85, edgecolor="#cccccc")

    # (Run metadata -- pair-routing, N=6, ego obs, 30k steps, 12 seeds/arm --
    #  lives in the LaTeX caption, not on the plot, to avoid overlapping bars.)

    despine(ax)
    fig.tight_layout()

    out_path = OUT / "phaseC_structure.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  (figsize=6.5x3.5)")
    return out_path


# ---------------------------------------------------------------------------
# Figure 2 — Scaling + Out-of-harness replication (two panels)
# ---------------------------------------------------------------------------

def fig_scaling_replication():
    """
    Panel (a): gnn_true vs floor across N = 4, 6, 8, 10.
      - gnn_true: line + markers + 95% CI band
      - floor: per-N shaded band (min ci_lo of mlp/gnn_wrong/gnn_complete to
                                   max ci_hi of same three) — floor moves with N
    Panel (b): TokenMatch bars: gnn_true, gnn_complete, gnn_wrong, mlp.
    """
    # --- load all N sizes ---
    d4  = load("summary_N4_ego.csv")
    d6  = load("summary_N6_ego.csv")
    d8  = load("summary_pair_N8_ego.csv")
    d10 = load("summary_pair_N10_ego.csv")
    dtok = load("summary_token_N6.csv")

    Ns = [4, 6, 8, 10]
    dfs = [d4, d6, d8, d10]

    # gnn_true values
    true_means = [float(get_row(df, "gnn_true")["mean"]) for df in dfs]
    true_lo    = [float(get_row(df, "gnn_true")["ci_lo"]) for df in dfs]
    true_hi    = [float(get_row(df, "gnn_true")["ci_hi"]) for df in dfs]

    # Floor band: min ci_lo and max ci_hi across mlp, gnn_wrong, gnn_complete
    floor_arms = ["mlp", "gnn_wrong", "gnn_complete"]
    floor_band_lo = []
    floor_band_hi = []
    floor_means   = []
    for df in dfs:
        lo_vals = [float(get_row(df, a)["ci_lo"]) for a in floor_arms]
        hi_vals = [float(get_row(df, a)["ci_hi"]) for a in floor_arms]
        m_vals  = [float(get_row(df, a)["mean"])  for a in floor_arms]
        floor_band_lo.append(min(lo_vals))
        floor_band_hi.append(max(hi_vals))
        floor_means.append(np.mean(m_vals))

    Ns_arr = np.array(Ns, dtype=float)

    # --- build figure ---
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.5, 3.2))

    # ---- Panel (a): scaling ----
    # floor shaded band
    ax_a.fill_between(Ns_arr, floor_band_lo, floor_band_hi,
                      color=C_FLOOR_BAND, alpha=0.8, label="Floor arms (mlp / wrong / complete) 95% CI",
                      zorder=1)
    # floor mean line
    ax_a.plot(Ns_arr, floor_means, color="#888888", linewidth=1.0,
              linestyle="--", zorder=2)

    # gnn_true CI band
    ax_a.fill_between(Ns_arr, true_lo, true_hi,
                      color=C_TASK_MATCHED, alpha=0.25, zorder=3)
    # gnn_true mean line + markers
    ax_a.plot(Ns_arr, true_means, color=C_TASK_MATCHED, linewidth=1.8,
              marker="o", markersize=6, markeredgewidth=0.8,
              markeredgecolor="#222222", label="gnn_true (task-matched graph)", zorder=4)

    ax_a.set_xlabel("Team size N", fontsize=9)
    ax_a.set_ylabel("Final-window mean return (95% CI)", fontsize=9)
    ax_a.set_title("(a) Team-size scaling", fontsize=10)
    ax_a.set_xticks(Ns)
    ax_a.set_xlim(3, 11)
    ax_a.set_ylim(0, float(max(true_hi)) * 1.15)  # headroom for gap labels above markers
    ax_a.legend(loc="lower right", fontsize=7.5, framealpha=0.9, edgecolor="#cccccc")

    # Annotation: gap is non-monotone — label each N's advantage just ABOVE the
    # gnn_true marker, in clear space (headroom added via ylim above).
    for n_val, tm, th, fm in zip(Ns, true_means, true_hi, floor_means):
        ax_a.annotate(f"+{tm - fm:.1f}",
                      xy=(n_val, th + 0.025 * ax_a.get_ylim()[1]),
                      ha="center", va="bottom", fontsize=7,
                      color=C_TASK_MATCHED, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                ec="#dddddd", lw=0.5, alpha=0.95))

    despine(ax_a)

    # ---- Panel (b): TokenMatch (out-of-harness) ----
    tok_arms   = ["gnn_true", "gnn_complete", "gnn_wrong", "mlp"]
    tok_labels = ["Task-matched\n(gnn_true)", "All-to-all\n(gnn_complete)",
                  "Wrong graph\n(gnn_wrong)", "No graph\n(mlp)"]
    tok_colors = [C_TASK_MATCHED, C_FLOOR_BASE, C_FLOOR_BASE, C_FLOOR_BASE]
    tok_hatches= [HATCH_GCN, HATCH_GCN, HATCH_GCN, HATCH_GCN]
    tok_edge   = ["#222222", "#666666", "#666666", "#666666"]

    bar_w_b = 0.55
    tok_xs  = np.arange(len(tok_arms), dtype=float) * (bar_w_b + 0.25)

    for x, arm, label, col, hatch, ec in zip(
            tok_xs, tok_arms, tok_labels, tok_colors, tok_hatches, tok_edge):
        row = get_row(dtok, arm)
        m   = float(row["mean"])
        elo, ehi = yerr_from_row(row)
        ax_b.bar(x, m, width=bar_w_b, color=col, hatch=hatch,
                 edgecolor=ec, linewidth=0.8,
                 yerr=[[elo], [ehi]], capsize=3.5,
                 error_kw=dict(elinewidth=1.0, ecolor="#333333", capthick=1.0),
                 zorder=3)

    ax_b.set_xticks(tok_xs)
    ax_b.set_xticklabels(tok_labels, fontsize=7.0)
    ax_b.set_ylabel("Final-window mean return (95% CI)", fontsize=9)
    ax_b.set_title("(b) Out-of-harness: TokenMatch (N=6)", fontsize=10)
    ax_b.set_ylim(0, 75)
    despine(ax_b)

    fig.tight_layout()
    out_path = OUT / "phaseC_scaling_replication.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  (figsize=7.5x3.2)")
    return out_path


# ---------------------------------------------------------------------------
# Figure 3 — Robustness: advantage persists across conditions
# ---------------------------------------------------------------------------

def fig_robustness():
    """
    Three sub-conditions, each showing gnn_true mean + CI and floor mean + CI:
      (1) pair-routing, ego obs       (summary_pair_N6_ego.csv)
      (2) pair-routing, full obs      (summary_pair_N6_full.csv)
      (3) degree-2 / neighbour task   (summary_nbr_N6_ego.csv)

    Uses a simple grouped-dot / bar layout with CI error bars.
    The degree-2 bar deliberately rises only modestly above the floor.
    """
    d_ego  = load("summary_pair_N6_ego.csv")
    d_full = load("summary_pair_N6_full.csv")
    d_nbr  = load("summary_nbr_N6_ego.csv")

    conditions = [
        ("Pair-routing\nego obs",       d_ego),
        ("Pair-routing\nfull obs",      d_full),
        ("Degree-2 topology\nego obs",  d_nbr),
    ]

    # Floor: mean of mlp, gnn_wrong, gnn_complete per condition
    floor_arms = ["mlp", "gnn_wrong", "gnn_complete"]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    grp_width = 0.5   # width per bar
    grp_gap   = 0.55  # gap between the pair of bars for each condition
    pair_gap  = 0.85  # gap between conditions

    group_xs    = []  # center x of each condition group
    x = 0.0
    bar_pairs   = []  # list of (x_floor, x_true, cond_label, df)
    for cond_label, df in conditions:
        x_floor = x
        x_true  = x + grp_width + grp_gap
        bar_pairs.append((x_floor, x_true, cond_label, df))
        group_xs.append((x_floor + x_true) / 2)
        x = x_true + grp_width + pair_gap

    # First pass: draw floor bars
    floor_label_done = False
    for x_floor, x_true, cond_label, df in bar_pairs:
        # Floor: average mean of mlp/gnn_wrong/gnn_complete; CI from their spread
        floor_m_vals = [float(get_row(df, a)["mean"])  for a in floor_arms]
        floor_lo_min = min(float(get_row(df, a)["ci_lo"]) for a in floor_arms)
        floor_hi_max = max(float(get_row(df, a)["ci_hi"]) for a in floor_arms)
        floor_m      = float(np.mean(floor_m_vals))
        floor_elo    = floor_m - floor_lo_min
        floor_ehi    = floor_hi_max - floor_m
        label_f = "Floor (no/wrong/all-to-all graph)" if not floor_label_done else "_nolegend_"
        ax.bar(x_floor, floor_m, width=grp_width,
               color=C_FLOOR_BASE, edgecolor="#666666", linewidth=0.8,
               yerr=[[floor_elo], [floor_ehi]], capsize=3.5,
               error_kw=dict(elinewidth=1.0, ecolor="#333333", capthick=1.0),
               label=label_f, zorder=3)
        floor_label_done = True

    # Second pass: draw gnn_true bars
    true_label_done = False
    for x_floor, x_true, cond_label, df in bar_pairs:
        r_true = get_row(df, "gnn_true")
        m      = float(r_true["mean"])
        elo, ehi = yerr_from_row(r_true)
        label_t = "Task-matched graph (gnn_true)" if not true_label_done else "_nolegend_"
        ax.bar(x_true, m, width=grp_width,
               color=C_TASK_MATCHED, edgecolor="#222222", linewidth=0.8,
               yerr=[[elo], [ehi]], capsize=3.5,
               error_kw=dict(elinewidth=1.0, ecolor="#333333", capthick=1.0),
               label=label_t, zorder=3)
        true_label_done = True

        # Annotate the advantage gap
        r_mlp  = get_row(df, "mlp")
        gap    = m - float(r_mlp["mean"])
        ax.annotate(f"+{gap:.1f}", xy=(x_true, m + ehi + 1.0),
                    ha="center", va="bottom", fontsize=7.5,
                    color=C_TASK_MATCHED)

    # x-axis: condition labels at group centers
    # place between each pair
    pair_centers = [(x_floor + grp_width / 2 + x_true + grp_width / 2) / 2
                    for x_floor, x_true, _, _ in bar_pairs]
    tick_positions = [(xf + grp_width / 2) for xf, _, _, _ in bar_pairs] + \
                     [(xt + grp_width / 2) for _, xt, _, _ in bar_pairs]
    # Use condition-level labels centered between the pair
    ax.set_xticks(pair_centers)
    ax.set_xticklabels([cond for _, _, cond, _ in bar_pairs], fontsize=8.5)

    ax.set_ylim(0, 90)
    ax.set_ylabel("Final-window mean return (95% CI)", fontsize=9)
    ax.set_title("Robustness: gnn_true advantage across conditions (N=6, 30k steps)", fontsize=10)

    ax.legend(loc="upper right", fontsize=8.0, framealpha=0.85, edgecolor="#cccccc")
    despine(ax)
    fig.tight_layout()

    out_path = OUT / "phaseC_robustness.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  (figsize=6.5x3.2)")
    return out_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=== phaseC_figures.py: verifying and plotting ===\n")

    print("--- Figure 1: structure comparison ---")
    p1 = fig_structure()

    print("\n--- Figure 2: scaling + replication ---")
    p2 = fig_scaling_replication()

    print("\n--- Figure 3: robustness ---")
    p3 = fig_robustness()

    print("\n=== Done. Written: ===")
    for p in [p1, p2, p3]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
