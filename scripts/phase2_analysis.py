"""Phase 2 analysis: H1, H2 tests + forest plot of episodes-to-80%.

Inputs:  results/phase2/<algo>__N<n>__<graph>__seed<s>/episodes.csv
Outputs:
  - results/phase2/figures/forest_ep80.{pdf,png}
  - results/phase2/figures/learning_curves_N{4,8}_{ring,erdos_renyi}.{pdf,png}
  - results/phase2/summary.csv  -- per (algo, N, graph, seed) scalars
  - results/phase2/summary_aggregate.csv -- per (algo, N, graph) mean ± CI
  - results/phase2/pairwise_<N>_<graph>.csv -- Welch + Holm per condition
  - results/phase2/h1_h2_summary.csv -- gap of gnn_qmix vs qmix per condition
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnmarl.utils.plotting import ALGO_LABELS, forest_plot, learning_curves  # noqa: E402
from gnnmarl.utils.stats import (  # noqa: E402
    episodes_to_threshold,
    mean_with_ci,
    pairwise_compare,
    self_threshold,
)

ALGOS = ("iql", "vdn", "qmix", "gnn_qmix")


def load_returns(run_dir: Path) -> np.ndarray:
    return pd.read_csv(run_dir / "episodes.csv")["return"].to_numpy(dtype=float)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results" / "phase2")
    parser.add_argument("--window", type=int, default=25)
    args = parser.parse_args()

    rows = []
    for child in sorted(args.results.iterdir()):
        if not child.is_dir() or not (child / "episodes.csv").exists():
            continue
        meta = parse_run(child.name)
        if meta is None or meta["algo"] not in ALGOS:
            continue
        ret = load_returns(child)
        rows.append({**meta, "returns": ret})
    if not rows:
        print(f"no runs found in {args.results}", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    print(f"[phase2] loaded {len(df)} runs")

    fig_dir = args.results / "figures"

    # Per-condition learning curves.
    for n in sorted(df["n_agents"].unique()):
        for graph in sorted(df["graph"].unique()):
            sub = df[(df["n_agents"] == n) & (df["graph"] == graph)]
            if sub.empty:
                continue
            by_algo = {a: sub[sub["algo"] == a]["returns"].tolist() for a in ALGOS}
            by_algo = {a: v for a, v in by_algo.items() if v}
            if not by_algo:
                continue
            learning_curves(
                by_algo,
                out_stem=fig_dir / f"learning_curves_N{n}_{graph}",
                window=args.window,
                title=f"Phase 2 -- CoordGrid (N={n}, {graph})",
            )

    # Per-(algo, N, graph) scalar summary, with per-run self-threshold
    # (panel-invariant: adding/removing an algorithm does not change any
    # other algorithm's ep80 number).
    summary_rows = []
    for n in sorted(df["n_agents"].unique()):
        for graph in sorted(df["graph"].unique()):
            sub = df[(df["n_agents"] == n) & (df["graph"] == graph)]
            if sub.empty:
                continue
            for _, r in sub.iterrows():
                ret = r["returns"]
                tail = max(1, len(ret) // 10)
                final = float(np.mean(ret[-tail:]))
                self_thr = self_threshold(ret, fraction=0.8)
                ep80 = episodes_to_threshold(ret, threshold=self_thr,
                                             window=args.window)
                summary_rows.append({
                    "algo": r["algo"], "n_agents": int(n), "graph": graph,
                    "seed": int(r["seed"]),
                    "final_window_mean": final,
                    "self_threshold_80pct": self_thr,
                    "episodes_to_80pct_of_own": ep80 if ep80 is not None else "",
                    "n_episodes": len(ret),
                })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.results / "summary.csv", index=False)

    # Per-condition aggregate + pairwise tests.
    agg_rows = []
    h1h2_rows = []
    forest_rows = []
    for n in sorted(df["n_agents"].unique()):
        for graph in sorted(df["graph"].unique()):
            sub = summary_df[(summary_df["n_agents"] == n) & (summary_df["graph"] == graph)]
            if sub.empty:
                continue
            seed_scores: dict[str, list[float]] = {}
            for algo in ALGOS:
                rows_a = sub[sub["algo"] == algo]
                if rows_a.empty:
                    continue
                finals = rows_a["final_window_mean"].astype(float).tolist()
                seed_scores[algo] = finals
                m, lo, hi = mean_with_ci(finals)
                ep80s = pd.to_numeric(rows_a["episodes_to_80pct_of_own"], errors="coerce").dropna().tolist()
                if ep80s:
                    me, loe, hie = mean_with_ci(ep80s)
                else:
                    me = loe = hie = float("nan")
                agg_rows.append({
                    "algo": algo, "n_agents": n, "graph": graph,
                    "final_mean": m, "final_ci_lo": lo, "final_ci_hi": hi,
                    "ep80_mean": me, "ep80_ci_lo": loe, "ep80_ci_hi": hie,
                    "n_seeds": len(finals),
                    "n_reaching_threshold": len(ep80s),
                })
                if not np.isnan(me):
                    forest_rows.append((
                        f"{ALGO_LABELS[algo]} | N={n} {graph}", me, loe, hie,
                    ))

            # Pairwise stats per condition.
            if len(seed_scores) >= 2 and all(len(v) >= 2 for v in seed_scores.values()):
                pairwise = pairwise_compare(seed_scores, test="welch")
                with (args.results / f"pairwise_N{n}_{graph}.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as fh:
                    w = csv.writer(fh)
                    w.writerow(["a", "b", "mean_a", "mean_b", "stat",
                                "p_raw", "p_holm", "test"])
                    for r in pairwise:
                        w.writerow([r.a, r.b, r.metric_a, r.metric_b,
                                    r.stat, r.p_raw, r.p_holm, r.test])

            # H1 per condition: gnn_qmix > qmix gap (one-sided Welch).
            if "gnn_qmix" in seed_scores and "qmix" in seed_scores:
                g = np.array(seed_scores["gnn_qmix"], dtype=float)
                q = np.array(seed_scores["qmix"], dtype=float)
                gap = float(g.mean() - q.mean())
                from scipy import stats as sst
                t_h1 = sst.ttest_ind(g, q, equal_var=False, alternative="greater")
                # Cohen's d using pooled SD (Hedges' correction unnecessary
                # at this magnitude). Sign convention: d > 0 favours GNN-QMIX.
                s_g = float(g.std(ddof=1))
                s_q = float(q.std(ddof=1))
                pooled = float(np.sqrt((s_g ** 2 + s_q ** 2) / 2.0))
                d_cohen = gap / pooled if pooled > 1e-9 else float("nan")
                h1h2_rows.append({
                    "n_agents": n, "graph": graph,
                    "gnn_qmix_mean": float(g.mean()),
                    "qmix_mean": float(q.mean()),
                    "gap_mean": gap,
                    "cohens_d": d_cohen,
                    "h1_p_one_sided": float(t_h1.pvalue),
                })

    pd.DataFrame(agg_rows).to_csv(args.results / "summary_aggregate.csv", index=False)
    pd.DataFrame(h1h2_rows).to_csv(args.results / "h1_summary.csv", index=False)

    # H2: difference-of-differences — is the (GNN-QMIX − QMIX) gap LARGER on
    # the structured (ring) graph than on the matched-density random graph?
    # Implemented as a one-sample t on per-seed paired DiD against zero.
    h2_rows = []
    for n in sorted(df["n_agents"].unique()):
        ring_q  = summary_df[(summary_df["n_agents"] == n) & (summary_df["graph"] == "ring")     & (summary_df["algo"] == "qmix")    ].sort_values("seed")["final_window_mean"].astype(float).to_numpy()
        ring_g  = summary_df[(summary_df["n_agents"] == n) & (summary_df["graph"] == "ring")     & (summary_df["algo"] == "gnn_qmix")].sort_values("seed")["final_window_mean"].astype(float).to_numpy()
        er_q    = summary_df[(summary_df["n_agents"] == n) & (summary_df["graph"] == "erdos_renyi") & (summary_df["algo"] == "qmix")    ].sort_values("seed")["final_window_mean"].astype(float).to_numpy()
        er_g    = summary_df[(summary_df["n_agents"] == n) & (summary_df["graph"] == "erdos_renyi") & (summary_df["algo"] == "gnn_qmix")].sort_values("seed")["final_window_mean"].astype(float).to_numpy()
        if not (len(ring_q) == len(ring_g) == len(er_q) == len(er_g) and len(ring_q) >= 2):
            continue
        delta_ring = ring_g - ring_q
        delta_er = er_g - er_q
        did = delta_ring - delta_er
        from scipy import stats as sst
        t = sst.ttest_1samp(did, popmean=0.0, alternative="greater")
        h2_rows.append({
            "n_agents": n,
            "n_seeds": int(len(did)),
            "delta_ring_mean": float(delta_ring.mean()),
            "delta_er_mean": float(delta_er.mean()),
            "did_mean": float(did.mean()),
            "did_std": float(did.std(ddof=1)) if len(did) > 1 else 0.0,
            "h2_p_one_sided": float(t.pvalue),
        })
    if h2_rows:
        pd.DataFrame(h2_rows).to_csv(args.results / "h2_did.csv", index=False)

    if forest_rows:
        forest_plot(forest_rows, out_stem=fig_dir / "forest_ep80",
                    title="Episodes to 80% of best return")

    # Console table.
    print()
    print(f"{'algo':>10} | {'N':>2} | {'graph':>12} | "
          f"{'final mean':>10} | {'95% CI':>20} | {'ep->80%':>8}")
    print("-" * 80)
    for r in agg_rows:
        ci = f"[{r['final_ci_lo']:.2f}, {r['final_ci_hi']:.2f}]"
        ep80 = (f"{r['ep80_mean']:.0f}" if not np.isnan(r['ep80_mean']) else "n/a")
        print(f"{ALGO_LABELS[r['algo']]:>10} | {r['n_agents']:>2} | "
              f"{r['graph']:>12} | {r['final_mean']:>10.3f} | "
              f"{ci:>20} | {ep80:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
