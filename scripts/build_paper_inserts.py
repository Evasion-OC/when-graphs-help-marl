"""Generate LaTeX inserts for the paper from results/* CSVs.

For each phase that has data, writes a small ``paper/results/phaseN.tex``
fragment containing the figures + tables the paper imports via ``\\input``.
Falls back to a `\\emph{Results pending.}` line when a phase has not run.

Also copies the figure PDFs into ``paper/figures/`` so the paper directory
is self-contained.

Usage::

    python scripts/build_paper_inserts.py
    pdflatex -interaction=nonstopmode paper/main.tex
    bibtex paper/main
    pdflatex -interaction=nonstopmode paper/main.tex
    pdflatex -interaction=nonstopmode paper/main.tex
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_TEX = REPO_ROOT / "paper" / "results"
PAPER_FIGS = REPO_ROOT / "paper" / "figures"

ALGO_LABEL = {"iql": "IQL", "vdn": "VDN", "qmix": "QMIX", "gnn_qmix": "GNN-QMIX"}


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def _fmt(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    return f"{x:.{digits}f}"


def _copy_figs(src_dir: Path, prefix: str) -> list[str]:
    """Copy ``*.pdf`` files from src_dir into paper/figures/ with a prefix."""
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    if not src_dir.exists():
        return out
    for p in sorted(src_dir.glob("*.pdf")):
        dst = PAPER_FIGS / f"{prefix}__{p.name}"
        shutil.copy2(p, dst)
        out.append(dst.name)
    return out


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[inserts] wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size} B)")


# --------------------------------------------------------------------- Phase 1
def phase1() -> None:
    rdir = REPO_ROOT / "results" / "phase1"
    out_tex = RESULTS_TEX / "phase1.tex"
    summary_path = rdir / "summary_aggregate.csv"
    pair_path = rdir / "pairwise.csv"

    if not summary_path.exists():
        _write(out_tex, "\\noindent\\emph{Phase 1 results pending --- run "
               "\\texttt{scripts/phase1\\_pilot.py} then "
               "\\texttt{scripts/phase1\\_analysis.py}.}\n")
        return

    figs = _copy_figs(rdir / "figures", "phase1")
    fig_name = next((f for f in figs if "learning_curves" in f), None)

    agg = pd.read_csv(summary_path)
    rows = []
    for _, r in agg.iterrows():
        ci = f"[{_fmt(r['final_ci_lo'])}, {_fmt(r['final_ci_hi'])}]"
        ep80 = _fmt(r["ep80_mean"], 0) if pd.notna(r["ep80_mean"]) else "--"
        rows.append(
            f"{ALGO_LABEL.get(r['algo'], r['algo'])} & "
            f"{int(r['n_seeds'])} & "
            f"{_fmt(r['final_mean'])} & {ci} & {ep80} \\\\"
        )

    pairwise_block = ""
    if pair_path.exists():
        pw = pd.read_csv(pair_path)
        # Only the IQL-vs-others row (sanity gate per PHASES.md).
        iql_rows = pw[(pw["a"] == "iql") | (pw["b"] == "iql")]
        if not iql_rows.empty:
            lines = []
            for _, r in iql_rows.iterrows():
                other = r["b"] if r["a"] == "iql" else r["a"]
                gap = r["mean_b"] - r["mean_a"] if r["a"] == "iql" else r["mean_a"] - r["mean_b"]
                lines.append(
                    f"IQL vs.\\ {ALGO_LABEL.get(other, other)} & "
                    f"{_fmt(gap)} & {_fmt(r['p_raw'], 3)} & "
                    f"{_fmt(r['p_holm'], 3)} \\\\"
                )
            pairwise_block = (
                "\n\\begin{table}[t]\n"
                "  \\centering\n"
                "  \\small\n"
                "  \\caption{Phase 1 sanity check: IQL vs.\\ each coordinated "
                "baseline on final-window mean return ($N=4$, ring, 3 seeds). "
                "Welch's $t$, Holm-corrected over the 6-pair family.}\n"
                "  \\label{tab:phase1_pairwise}\n"
                "  \\begin{tabular}{lrrr}\n"
                "    \\toprule\n"
                "    Comparison & $\\Delta$ mean return & $p_{\\text{raw}}$ "
                "& $p_{\\text{Holm}}$ \\\\\n"
                "    \\midrule\n"
                f"    {chr(10).join(lines)}\n"
                "    \\bottomrule\n"
                "  \\end{tabular}\n"
                "\\end{table}\n"
            )

    fig_block = ""
    if fig_name:
        fig_block = (
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=0.85\\linewidth]{{figures/{fig_name}}}\n"
            "  \\caption{Phase 1 learning curves on \\coordgrid\\ ($N=4$, "
            "ring), three seeds. Shaded bands are $\\pm 1$ s.d.\\ across "
            "seeds of the 25-episode rolling mean.}\n"
            "  \\label{fig:phase1_curves}\n"
            "\\end{figure}\n"
        )

    table = (
        "\\begin{table}[t]\n"
        "  \\centering\n"
        "  \\small\n"
        "  \\caption{Phase 1 final-window mean return ($N=4$, ring, 3 seeds). "
        "Confidence intervals are Student-$t$ 95\\% across seeds; "
        "episodes-to-80\\% is reported only for seeds that crossed the "
        "threshold within $5\\times 10^4$ steps.}\n"
        "  \\label{tab:phase1_summary}\n"
        "  \\begin{tabular}{lrrrr}\n"
        "    \\toprule\n"
        "    Algorithm & $n_{\\text{seeds}}$ & Final mean & 95\\% CI & "
        "Episodes to 80\\% \\\\\n"
        "    \\midrule\n"
        f"    {chr(10).join(rows)}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )

    _write(out_tex, fig_block + "\n" + table + pairwise_block)


# --------------------------------------------------------------------- Phase 2
def phase2() -> None:
    rdir = REPO_ROOT / "results" / "phase2"
    out_tex = RESULTS_TEX / "phase2.tex"
    summary_path = rdir / "summary_aggregate.csv"
    h_path = rdir / "h1_summary.csv"
    h2_path = rdir / "h2_did.csv"
    if not summary_path.exists():
        _write(out_tex, "\\noindent\\emph{Phase 2 results pending --- run "
               "\\texttt{scripts/phase2\\_e1\\_sweep.py} then "
               "\\texttt{scripts/phase2\\_analysis.py}.}\n")
        return

    figs = _copy_figs(rdir / "figures", "phase2")
    forest = next((f for f in figs if "forest" in f), None)

    agg = pd.read_csv(summary_path).sort_values(["graph", "n_agents", "algo"])
    rows = []
    for _, r in agg.iterrows():
        ci = f"[{_fmt(r['final_ci_lo'])}, {_fmt(r['final_ci_hi'])}]"
        ep80 = _fmt(r["ep80_mean"], 0) if pd.notna(r["ep80_mean"]) else "--"
        rows.append(
            f"{_tex_escape(r['graph'])} & {int(r['n_agents'])} & "
            f"{ALGO_LABEL.get(r['algo'], r['algo'])} & "
            f"{_fmt(r['final_mean'])} & {ci} & {ep80} \\\\"
        )

    h_block = ""
    if h_path.exists():
        h = pd.read_csv(h_path)
        if not h.empty:
            lines = [
                f"{int(r['n_agents'])} & {_tex_escape(r['graph'])} & {_fmt(r['gap_mean'])} "
                f"& {_fmt(r.get('cohens_d', float('nan')))} & "
                f"{_fmt(r['h1_p_one_sided'], 3)} \\\\"
                for _, r in h.iterrows()
            ]
            h_block = (
                "\n\\begin{table}[t]\n"
                "  \\centering\n"
                "  \\small\n"
                "  \\caption{H1 (within-condition): GNN-QMIX minus QMIX in "
                "mean final-window return, with Cohen's $d$ effect size "
                "(pooled SD, sign convention $d > 0$ favours GNN-QMIX). "
                "$p$-values are one-sided Welch $t$ ($H_a$: \\gnnqmix\\ $>$ "
                "\\qmix), not corrected (single test per condition).}\n"
                "  \\label{tab:phase2_h1}\n"
                "  \\begin{tabular}{rlrrr}\n"
                "    \\toprule\n"
                "    $N$ & graph & $\\Delta = $ GNN $-$ QMIX & "
                "Cohen's $d$ & $p$ (1-sided) \\\\\n"
                "    \\midrule\n"
                f"    {chr(10).join(lines)}\n"
                "    \\bottomrule\n"
                "  \\end{tabular}\n"
                "\\end{table}\n"
            )
    if h2_path.exists():
        h2 = pd.read_csv(h2_path)
        if not h2.empty:
            lines = [
                f"{int(r['n_agents'])} & {_fmt(r['delta_ring_mean'])} & "
                f"{_fmt(r['delta_er_mean'])} & {_fmt(r['did_mean'])} & "
                f"{_fmt(r['h2_p_one_sided'], 3)} \\\\"
                for _, r in h2.iterrows()
            ]
            h_block += (
                "\n\\begin{table}[t]\n"
                "  \\centering\n"
                "  \\small\n"
                "  \\caption{H2 (difference-of-differences): the GNN-QMIX vs.\\ "
                "QMIX gap on ring minus the same gap on Erd\\H{o}s--R\\'enyi at "
                "matched edge density. One-sample $t$ on per-seed DiD against "
                "0 ($H_a$: DiD $> 0$).}\n"
                "  \\label{tab:phase2_h2}\n"
                "  \\begin{tabular}{rrrrr}\n"
                "    \\toprule\n"
                "    $N$ & $\\Delta_{\\text{ring}}$ & "
                "$\\Delta_{\\text{ER}}$ & DiD & $p$ (1-sided) \\\\\n"
                "    \\midrule\n"
                f"    {chr(10).join(lines)}\n"
                "    \\bottomrule\n"
                "  \\end{tabular}\n"
                "\\end{table}\n"
            )

    fig_block = ""
    if forest:
        fig_block = (
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=0.78\\linewidth]{{figures/{forest}}}\n"
            "  \\caption{Phase 2 forest plot: episodes to 80\\% of each "
            "algorithm's \\emph{own} final-window return, per algorithm "
            "$\\times$ condition. Lower is better. Bars are 95\\% Student-$t$ "
            "CIs across seeds; CIs that extend into negative values reflect "
            "the conservative Student-$t$ interval with $n=3$ and should be "
            "interpreted as 'no lower bound resolved.'}\n"
            "  \\label{fig:phase2_forest}\n"
            "\\end{figure}\n"
        )

    table = (
        "\\begin{table}[t]\n"
        "  \\centering\n"
        "  \\small\n"
        "  \\caption{Phase 2 final-window mean return by condition. Edge "
        "density is matched between \\texttt{ring} and \\texttt{erdos\\_renyi} "
        "by construction.}\n"
        "  \\label{tab:phase2_summary}\n"
        "  \\begin{tabular}{llrrrr}\n"
        "    \\toprule\n"
        "    graph & $N$ & algorithm & Final mean & 95\\% CI & "
        "Episodes to 80\\% \\\\\n"
        "    \\midrule\n"
        f"    {chr(10).join(rows)}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )

    _write(out_tex, fig_block + "\n" + table + h_block)


# --------------------------------------------------------------------- Phase 4
def phase4() -> None:
    rdir = REPO_ROOT / "results" / "phase4"
    out_tex = RESULTS_TEX / "phase4.tex"
    grid_path = rdir / "summary_grid.csv"
    h3_path = rdir / "h3_test.csv"
    if not grid_path.exists():
        _write(out_tex, "\\noindent\\emph{Phase 4 results pending --- run "
               "\\texttt{scripts/phase4\\_ablation.py} then "
               "\\texttt{scripts/phase4\\_analysis.py}.}\n")
        return

    figs = _copy_figs(rdir / "figures", "phase4")
    heatmap = next((f for f in figs if "heatmap" in f), None)

    h3_block = ""
    if h3_path.exists():
        h3 = pd.read_csv(h3_path)
        if not h3.empty:
            # Use 3-dp on relative_range so the 9.77% vs 10% boundary is
            # visible (would round to 0.10 at 2dp and look like a pass-by-rounding).
            lines = [
                f"{int(r['diameter'])} & {int(r['argmax_depth'])} & "
                f"{_fmt(r['peak_mean_return'])} & {_fmt(r['relative_range'], 3)} & "
                + (r"\checkmark" if int(r['h3_match']) else r"$\times$") + r" \\"
                for _, r in h3.iterrows()
            ]
            n_match = int(h3["h3_match"].sum())
            n_total = len(h3)
            h3_block = (
                "\n\\begin{table}[t]\n"
                "  \\centering\n"
                "  \\small\n"
                "  \\caption{H3 (depth--diameter alignment) per-diameter "
                "test, pre-registered criteria: (a) the depth curve must be "
                "non-flat (relative range $\\geq 10\\%$ of peak), AND (b) the "
                "argmax depth $L^*$ must satisfy $|L^* - d| \\leq 1$. "
                f"Per-diameter passes: {n_match}/{n_total}.}}\n"
                "  \\label{tab:phase4_h3}\n"
                "  \\begin{tabular}{rrrrc}\n"
                "    \\toprule\n"
                "    $d$ & $L^*$ (argmax) & Peak return & rel.\\ range & "
                "pass \\\\\n"
                "    \\midrule\n"
                f"    {chr(10).join(lines)}\n"
                "    \\bottomrule\n"
                "  \\end{tabular}\n"
                "\\end{table}\n"
            )

    fig_block = ""
    if heatmap:
        fig_block = (
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=0.7\\linewidth]{{figures/{heatmap}}}\n"
            "  \\caption{Phase 4: GNN-QMIX final-window mean return as a "
            "function of GNN depth $L$ and coordination graph diameter $d$. "
            "Cells along the diagonal correspond to depth--diameter alignment "
            "($L \\approx d$), the H3 prediction.}\n"
            "  \\label{fig:phase4_heatmap}\n"
            "\\end{figure}\n"
        )

    _write(out_tex, fig_block + h3_block)


def main() -> int:
    RESULTS_TEX.mkdir(parents=True, exist_ok=True)
    phase1()
    phase2()
    phase4()
    print("[inserts] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
