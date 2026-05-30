"""Phase 8 analysis — confound-killer + capacity control.

Inputs:  results/phase8/<algo>__N<n>__ring__seed<s>/episodes.csv
Outputs:
  - results/phase8/summary_aggregate.csv   per (algo, N): final-window mean ± CI
  - results/phase8/pairwise_N<n>.csv        Welch + Holm over the algo set per N
  - results/phase8/confound_verdict.csv     the two decisive comparisons per N:
        * QMIX vs GNN-QMIX   (does the negative ordering survive long training?)
        * GNN-QMIX vs MLP-QMIX (graph vs capacity, matched params)
        * MLP-QMIX vs QMIX     (capacity effect)
  - results/phase8/figures/learning_curves_N<n>.png

Final-window metric = mean episodic return over the last `--window` fraction
of each run's episodes (default 10%), matching the paper's "final-window
return" definition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    ap.add_argument("--results", type=Path, default=REPO_ROOT / "results" / "phase8")
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
    print(f"[phase8] loaded {len(df)} runs across "
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
    print("\n=== final-window return (mean [95% CI]) ===")
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

        def gap(a, b):
            for r in results:
                if {r.a, r.b} == {a, b}:
                    # orient as a - b
                    if r.a == a:
                        return r.metric_a - r.metric_b, r.p_holm
                    return r.metric_b - r.metric_a, r.p_holm
            return float("nan"), float("nan")

        for label, a, b in [
            ("convergence: QMIX - GNN-QMIX", "qmix", "gnn_qmix"),
            ("graph: GNN-QMIX - MLP-QMIX", "gnn_qmix", "mlp_qmix"),
            ("capacity: MLP-QMIX - QMIX", "mlp_qmix", "qmix"),
        ]:
            if a in scores and b in scores:
                d, p = gap(a, b)
                verdict_rows.append({
                    "n_agents": n, "comparison": label,
                    "delta": d, "p_holm": p,
                    "sig_0.05": (p < 0.05) if np.isfinite(p) else False,
                })

    if verdict_rows:
        verdict = pd.DataFrame(verdict_rows)
        verdict.to_csv(args.results / "confound_verdict.csv", index=False)
        print("\n=== decisive comparisons (delta, Holm p) ===")
        for _, r in verdict.iterrows():
            sig = "*" if r["sig_0.05"] else " "
            print(f"  N={r['n_agents']}  {r['comparison']:<32} "
                  f"delta={r['delta']:+8.2f}  p_holm={r['p_holm']:.3f} {sig}")

    # ---- Learning curves (optional, best-effort) -------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = args.results / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        colors = {"iql": "#888", "vdn": "#1f77b4", "qmix": "#2ca02c",
                  "gnn_qmix": "#d62728", "mlp_qmix": "#9467bd"}
        for n in sorted(df["n_agents"].unique()):
            fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
            for algo in ALGOS:
                runs = sorted((args.results).glob(f"{algo}__N{n}__ring__seed*"))
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
            ax.set_title(f"Phase 8 — coord_grid ring N={n} (long horizon)")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(fig_dir / f"learning_curves_N{n}.png")
            plt.close(fig)
        print(f"\n[phase8] figures -> {fig_dir}")
    except Exception as e:  # noqa: BLE001
        print(f"[phase8] plotting skipped: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
