"""Phase 10 analysis — second GNN family (GAT) robustness check.

Reads a results dir that already contains the qmix / mlp_qmix / gnn_qmix
runs (e.g. results/phase8 for CoordGrid ring, results/phase9_mpe for MPE
simple_spread) plus the newly-added gat_qmix runs, and reports the GAT
contrasts using the SAME protocol as Phase 8/9 (final-window mean; Welch +
Holm over the GAT family + Mann-Whitney; Cohen's d):

    attention-graph : GAT-QMIX - MLP-QMIX  (does learned attention beat the
                                            parameter-matched no-graph control?)
    attention-vs-GCN: GAT-QMIX - GNN-QMIX  (does attention help over fixed
                                            degree-normalized aggregation?)
    net             : GAT-QMIX - QMIX

The headline question is the first contrast: if GAT-QMIX also fails to beat
MLP-QMIX, the graph penalty is not specific to the vanilla GCN -- it holds
for the attention mechanism the critiqued methods (DGN, G2ANet) actually
use. GAT has marginally MORE parameters than MLP-QMIX (the attention
vectors), so a GAT < MLP result is conservative.

Usage:
  python scripts/phase10_gat_analysis.py --results results/phase8 --tag coordgrid_ring
  python scripts/phase10_gat_analysis.py --results results/phase9_mpe --tag mpe_spread
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

from gnnmarl.utils.stats import mean_with_ci  # noqa: E402

ALGOS = ("qmix", "mlp_qmix", "gnn_qmix", "gat_qmix")


def parse_run(name: str) -> dict | None:
    parts = name.split("__")
    if len(parts) != 4:
        return None
    try:
        return {"algo": parts[0], "n_agents": int(parts[1].lstrip("N")),
                "graph": parts[2], "seed": int(parts[3].replace("seed", ""))}
    except ValueError:
        return None


def final_window_mean(run_dir: Path, frac: float) -> float:
    ret = pd.read_csv(run_dir / "episodes.csv")["return"].to_numpy(dtype=float)
    if ret.size == 0:
        return float("nan")
    k = max(1, int(round(ret.size * frac)))
    return float(ret[-k:].mean())


def cohend(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / sp) if sp > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="gat")
    ap.add_argument("--window", type=float, default=0.10)
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
    if "gat_qmix" not in set(df["algo"]):
        print(f"[phase10] no gat_qmix runs in {args.results} yet", file=sys.stderr)
        return 1
    print(f"[phase10:{args.tag}] loaded {len(df)} runs, "
          f"N={sorted(df['n_agents'].unique())}, algos={sorted(df['algo'].unique())}")

    print(f"\n=== final-window return (mean [95% CI]) — {args.tag} ===")
    for n in sorted(df["n_agents"].unique()):
        print(f"  N={n}:")
        for a in ALGOS:
            v = df[(df["algo"] == a) & (df["n_agents"] == n)]["final"].to_numpy()
            if len(v) == 0:
                continue
            m, lo, hi = mean_with_ci(v)
            print(f"    {a:<10} {m:8.2f}  [{lo:.2f}, {hi:.2f}]  (n={len(v)})")

    contrasts = [
        ("attention-graph: GAT-QMIX - MLP-QMIX", "gat_qmix", "mlp_qmix"),
        ("attention-vs-GCN: GAT-QMIX - GNN-QMIX", "gat_qmix", "gnn_qmix"),
        ("net: GAT-QMIX - QMIX", "gat_qmix", "qmix"),
    ]
    out_rows = []
    print(f"\n=== GAT contrasts — {args.tag} ===")
    print("  (Holm over the 3 GAT contrasts per N; MWU = non-parametric)")
    for n in sorted(df["n_agents"].unique()):
        scores = {a: df[(df["algo"] == a) & (df["n_agents"] == n)]["final"].to_numpy()
                  for a in ALGOS}
        raw_ps, recs = [], []
        for label, a, b in contrasts:
            xa, xb = scores.get(a), scores.get(b)
            if xa is None or xb is None or len(xa) == 0 or len(xb) == 0:
                continue
            delta = float(xa.mean() - xb.mean())
            raw_p = float(stats.ttest_ind(xa, xb, equal_var=False).pvalue)
            mwu_p = float(stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
            d = cohend(xa, xb)
            recs.append([n, label, delta, d, raw_p, mwu_p])
            raw_ps.append(raw_p)
        if not recs:
            continue
        order = np.argsort(raw_ps)
        holm = [0.0] * len(raw_ps)
        run = 0.0
        for rank, idx in enumerate(order):
            run = max(run, min(raw_ps[idx] * (len(raw_ps) - rank), 1.0))
            holm[idx] = run
        for rec, h in zip(recs, holm):
            n_, label, delta, d, raw_p, mwu_p = rec
            out_rows.append({"n_agents": n_, "comparison": label, "delta": delta,
                             "cohen_d": d, "welch_raw_p": raw_p, "welch_holm_p": h,
                             "mannwhitney_p": mwu_p, "sig_holm": h < 0.05,
                             "sig_mwu": mwu_p < 0.05})
            flags = ("Holm*" if h < 0.05 else "     ") + (" MWU*" if mwu_p < 0.05 else "")
            print(f"  N={n_}  {label:<38} delta={delta:+8.2f}  d={d:+5.2f}  "
                  f"holm={h:.3f}  mwu={mwu_p:.3f}  {flags}")

    if out_rows:
        outp = args.results / f"gat_verdict__{args.tag}.csv"
        pd.DataFrame(out_rows).to_csv(outp, index=False)
        print(f"\n[phase10:{args.tag}] -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
