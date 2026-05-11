"""H1 hypothesis test on Phase 2C main-sweep results.

Hypothesis 1 (paper version): GNN-QMIX outperforms QMIX in tasks with
graph-structured coordination demand; matches QMIX otherwise.

This script runs three analyses on results/phase_2_main/:
1. Per-scale pairwise Welch t-tests with Holm correction (5 seeds → n=5).
2. Effect sizes (Cohen's d) for every pair.
3. Mann-Whitney U (non-parametric) as robustness check, since n=5 is
   modest for parametric assumptions.

Outputs:
    results/phase_2_main/h1_tests.csv  - one row per (scale, pair, metric)
    results/phase_2_main/h1_tests.md   - human-readable summary
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "results" / "phase_2_main"

ALGOS = ["iql", "vdn", "qmix", "gnn_qmix"]
SCALES = ["small", "medium", "large"]
METRICS = ["peak", "final"]


def load_per_seed() -> pd.DataFrame:
    """Reuse the per-seed extraction from plot_main_sweep.py.

    Imports it as a sibling script via explicit path rather than as a
    package, so this works regardless of the working directory.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_main_sweep import per_seed_summary  # type: ignore[import-not-found]  # noqa: PLC0415

    return per_seed_summary()


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s_pooled = np.sqrt(((np.std(a, ddof=1) ** 2) + (np.std(b, ddof=1) ** 2)) / 2)
    if s_pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / s_pooled)


def holm_correct(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros_like(pvals)
    running_max = 0.0
    for rank, idx in enumerate(order):
        v = pvals[idx] * (n - rank)
        running_max = max(running_max, min(v, 1.0))
        adj[idx] = running_max
    return adj


def main() -> int:
    if not MAIN.exists():
        print(f"!! {MAIN} not found — run scripts/run_main_sweep.py first.")
        return 2

    per_seed = load_per_seed()
    if per_seed.empty:
        print("No data found.")
        return 1

    rows = []
    # Per-scale pairwise tests, separately for each metric, with Holm
    # correction across the C(4,2)=6 pairs at each (scale, metric).
    for scale in SCALES:
        sub_scale = per_seed[per_seed["scale"] == scale]
        if sub_scale.empty:
            continue
        for metric in METRICS:
            pairs: list[tuple[str, str, np.ndarray, np.ndarray]] = []
            for i, a in enumerate(ALGOS):
                for b in ALGOS[i + 1:]:
                    xa = sub_scale[sub_scale["algo"] == a][metric].to_numpy()
                    xb = sub_scale[sub_scale["algo"] == b][metric].to_numpy()
                    if len(xa) == 0 or len(xb) == 0:
                        continue
                    pairs.append((a, b, xa, xb))
            if not pairs:
                continue
            pvals_t = np.array(
                [stats.ttest_ind(xa, xb, equal_var=False).pvalue for _, _, xa, xb in pairs]
            )
            pvals_t_adj = holm_correct(pvals_t)
            pvals_u = np.array(
                [stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue
                 for _, _, xa, xb in pairs]
            )
            pvals_u_adj = holm_correct(pvals_u)
            for k, (a, b, xa, xb) in enumerate(pairs):
                rows.append(
                    {
                        "scale": scale,
                        "metric": metric,
                        "algo_A": a,
                        "algo_B": b,
                        "n_A": len(xa),
                        "n_B": len(xb),
                        "mean_A": float(np.mean(xa)),
                        "mean_B": float(np.mean(xb)),
                        "delta": float(np.mean(xa) - np.mean(xb)),
                        "cohens_d": cohens_d(xa, xb),
                        "welch_t": float(stats.ttest_ind(xa, xb, equal_var=False).statistic),
                        "welch_p": float(pvals_t[k]),
                        "welch_p_holm": float(pvals_t_adj[k]),
                        "mwu_p": float(pvals_u[k]),
                        "mwu_p_holm": float(pvals_u_adj[k]),
                    }
                )

    out = pd.DataFrame(rows)
    out_csv = MAIN / "h1_tests.csv"
    out.to_csv(out_csv, index=False)
    print(f"  -> {out_csv}")

    md = ["# H1 statistical tests", "", "Per-scale pairwise comparisons on main-sweep results.",
          "Holm correction across 6 pairs per (scale, metric).", ""]
    for scale in SCALES:
        for metric in METRICS:
            chunk = out[(out["scale"] == scale) & (out["metric"] == metric)]
            if chunk.empty:
                continue
            md.append(f"## scale={scale}, metric={metric}")
            md.append("")
            md.append(
                "| A vs B | mean_A | mean_B | Δ | d | Welch p (Holm) | MWU p (Holm) | sig |"
            )
            md.append("|---|---|---|---|---|---|---|---|")
            for _, r in chunk.iterrows():
                sig = "**yes**" if r["welch_p_holm"] < 0.05 else "no"
                md.append(
                    f"| {r['algo_A']} vs {r['algo_B']} | {r['mean_A']:.3f} | "
                    f"{r['mean_B']:.3f} | {r['delta']:+.3f} | {r['cohens_d']:+.2f} | "
                    f"{r['welch_p_holm']:.4f} | {r['mwu_p_holm']:.4f} | {sig} |"
                )
            md.append("")
    out_md = MAIN / "h1_tests.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"  -> {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
