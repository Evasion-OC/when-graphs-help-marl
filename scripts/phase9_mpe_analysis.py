"""Phase 9 analysis — external-validity replication on MPE simple_spread.

Identical statistical protocol to ``phase8_analysis.py`` (final-window
mean over the last ``--window`` fraction; Welch t-test raw + Holm over the
3 decisive contrasts + Holm over all cell-pairs; Mann-Whitney U), so the
MPE numbers are directly comparable to the CoordGrid numbers. The only
difference is the run-name graph tag (``spread``) and the figure labels.

Inputs:  results/phase9_mpe/<algo>__N<n>__spread__seed<s>/episodes.csv
Outputs (under --results):
  - summary_aggregate.csv     per (algo, N): final-window mean +/- 95% CI
  - pairwise_N<n>.csv          Welch + Holm over the algo set per N
  - confound_verdict.csv       the 3 decisive contrasts per N
  - figures/learning_curves_N<n>.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnmarl.utils.stats import mean_with_ci, pairwise_compare  # noqa: E402

ALGOS = ("iql", "vdn", "qmix", "gnn_qmix", "mlp_qmix")


def parse_run(name: str) -> dict | None:
    parts = name.split("__")
    if len(parts) != 4:
        return None
    try:
        return {
            "algo": parts[0],
            "n_agents": int(parts[1].lstrip("N")),
            "graph": parts[2],
            "seed": int(parts[3].replace("seed", "")),
        }
    except ValueError:
        return None


def final_window_mean(run_dir: Path, frac: float) -> float:
    ret = pd.read_csv(run_dir / "episodes.csv")["return"].to_numpy(dtype=float)
    if ret.size == 0:
        return float("nan")
    k = max(1, int(round(ret.size * frac)))
    return float(ret[-k:].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=REPO_ROOT / "results" / "phase9_mpe")
    ap.add_argument("--window", type=float, default=0.10,
                    help="trailing fraction of episodes for final-window mean")
    args = ap.parse_args()

    rows = []
    for child in sorted(args.results.iterdir()):
        if not child.is_dir() or not (child / "episodes.csv").exists():
            continue
        meta = parse_run(child.name)
        if meta is None or meta["algo"] not in ALGOS:
            continue
        rows.append({**meta, "final": final_window_mean(child, args.window)})
    if not rows:
        print(f"no runs found in {args.results}", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    print(f"[phase9] loaded {len(df)} runs across "
          f"{df['algo'].nunique()} algos, N={sorted(df['n_agents'].unique())}")

    # ---- Per (algo, N) aggregate -----------------------------------------
    agg_rows = []
    for (algo, n), g in df.groupby(["algo", "n_agents"]):
        m, lo, hi = mean_with_ci(g["final"].to_numpy())
        agg_rows.append({
            "algo": algo, "n_agents": n, "n_seeds": len(g),
            "final_mean": m, "ci_lo": lo, "ci_hi": hi,
        })
    agg = pd.DataFrame(agg_rows).sort_values(["n_agents", "algo"])
    agg.to_csv(args.results / "summary_aggregate.csv", index=False)
    print("\n=== final-window return (mean [95% CI]) — MPE simple_spread ===")
    for n in sorted(df["n_agents"].unique()):
        print(f"  N={n}:")
        for _, r in agg[agg["n_agents"] == n].iterrows():
            print(f"    {r['algo']:<10} {r['final_mean']:8.2f}  "
                  f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]  (n={int(r['n_seeds'])})")

    # ---- Pairwise + decisive verdicts ------------------------------------
    verdict_rows = []
    for n in sorted(df["n_agents"].unique()):
        sub = df[df["n_agents"] == n]
        scores = {a: sub[sub["algo"] == a]["final"].to_numpy()
                  for a in ALGOS if (sub["algo"] == a).any()}
        scores = {a: v for a, v in scores.items() if len(v) > 0}
        if len(scores) < 2:
            continue
        results = pairwise_compare(scores, test="welch")
        pd.DataFrame([r.__dict__ for r in results]).to_csv(
            args.results / f"pairwise_N{n}.csv", index=False)

        def gap_allpairs(a, b):
            for r in results:
                if {r.a, r.b} == {a, b}:
                    if r.a == a:
                        return r.metric_a - r.metric_b, r.p_holm
                    return r.metric_b - r.metric_a, r.p_holm
            return float("nan"), float("nan")

        decisive = [
            ("convergence: QMIX - GNN-QMIX", "qmix", "gnn_qmix"),
            ("graph: GNN-QMIX - MLP-QMIX", "gnn_qmix", "mlp_qmix"),
            ("capacity: MLP-QMIX - QMIX", "mlp_qmix", "qmix"),
        ]
        raw_ps = []
        recs = []
        for label, a, b in decisive:
            if a in scores and b in scores:
                xa, xb = scores[a], scores[b]
                delta = float(np.mean(xa) - np.mean(xb))
                raw_p = float(stats.ttest_ind(xa, xb, equal_var=False).pvalue)
                mwu_p = float(stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
                _, allpairs_p = gap_allpairs(a, b)
                recs.append([n, label, delta, raw_p, mwu_p, allpairs_p])
                raw_ps.append(raw_p)
        order = np.argsort(raw_ps)
        holm3 = [0.0] * len(raw_ps)
        run = 0.0
        for rank, idx in enumerate(order):
            run = max(run, min(raw_ps[idx] * (len(raw_ps) - rank), 1.0))
            holm3[idx] = run
        for rec, h3 in zip(recs, holm3):
            n_, label, delta, raw_p, mwu_p, allpairs_p = rec
            verdict_rows.append({
                "n_agents": n_, "comparison": label, "delta": delta,
                "welch_raw_p": raw_p,
                "welch_holm3_p": h3,
                "welch_holm_allpairs_p": allpairs_p,
                "mannwhitney_p": mwu_p,
                "sig_holm3": h3 < 0.05,
                "sig_mwu": mwu_p < 0.05,
            })

    if verdict_rows:
        verdict = pd.DataFrame(verdict_rows)
        verdict.to_csv(args.results / "confound_verdict.csv", index=False)
        print("\n=== decisive comparisons — MPE simple_spread ===")
        print("  (Holm3 = corrected over the 3 decisive tests; MWU = "
              "non-parametric)")
        for _, r in verdict.iterrows():
            flags = ("Holm3*" if r["sig_holm3"] else "      ") + \
                    (" MWU*" if r["sig_mwu"] else "     ")
            print(f"  N={r['n_agents']}  {r['comparison']:<30} "
                  f"delta={r['delta']:+8.2f}  raw={r['welch_raw_p']:.3f}  "
                  f"holm3={r['welch_holm3_p']:.3f}  mwu={r['mannwhitney_p']:.3f}  "
                  f"{flags}")

    # ---- Learning curves -------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType, not Type 3
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt

        fig_dir = args.results / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        colors = {"iql": "#888", "vdn": "#1f77b4", "qmix": "#2ca02c",
                  "gnn_qmix": "#d62728", "mlp_qmix": "#9467bd"}
        for n in sorted(df["n_agents"].unique()):
            fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
            for algo in ALGOS:
                runs = sorted(args.results.glob(f"{algo}__N{n}__*__seed*"))
                curves = []
                for rd in runs:
                    if (rd / "episodes.csv").exists():
                        r = pd.read_csv(rd / "episodes.csv")["return"].to_numpy(float)
                        curves.append(r)
                if not curves:
                    continue
                L = min(len(c) for c in curves)
                arr = np.stack([c[:L] for c in curves])
                w = max(1, L // 50)
                sm = np.array([arr[:, max(0, i - w):i + 1].mean(axis=1)
                               for i in range(L)]).T
                mean = sm.mean(0)
                sem = sm.std(0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros(L)
                ax.plot(mean, color=colors.get(algo, "#333"), label=algo, lw=1.8)
                ax.fill_between(range(L), mean - sem, mean + sem,
                                color=colors.get(algo, "#333"), alpha=0.15)
            ax.set_xlabel("episode")
            ax.set_ylabel("return (smoothed)")
            ax.set_title(f"MPE simple_spread N={n} (150k steps)")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(fig_dir / f"learning_curves_N{n}.pdf")
            plt.close(fig)
        print(f"\n[phase9] figures -> {fig_dir}")
    except Exception as e:  # noqa: BLE001
        print(f"[phase9] plotting skipped: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
