"""Phase 13 analysis --- EPyMARL benchmarks (LBF, RWARE).

Same protocol as Phase 8/9 (final-window mean over the last --window
fraction; Welch raw + Holm over the 3 decisive contrasts + Mann-Whitney;
Cohen's d). Groups runs by env tag (parsed from
``<algo>__<tag>__<graph>__seed<s>``).

Inputs:  results/phase13_epymarl/<algo>__<tag>__<graph>__seed<s>/episodes.csv
Outputs: summary_aggregate.csv, confound_verdict.csv (per env tag).
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

ALGOS = ("iql", "vdn", "qmix", "gnn_qmix", "mlp_qmix")


def parse_run(name: str) -> dict | None:
    parts = name.split("__")
    if len(parts) != 4:
        return None
    try:
        return {"algo": parts[0], "tag": parts[1], "graph": parts[2],
                "seed": int(parts[3].replace("seed", ""))}
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
    ap.add_argument("--results", type=Path,
                    default=REPO_ROOT / "results" / "phase13_epymarl")
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
    print(f"[phase13] loaded {len(df)} runs across tags={sorted(df['tag'].unique())}")

    agg_rows, verdict_rows = [], []
    for tag in sorted(df["tag"].unique()):
        sub = df[df["tag"] == tag]
        print(f"\n=== {tag}: final-window return (mean [95% CI]) ===")
        for a in ALGOS:
            v = sub[sub["algo"] == a]["final"].to_numpy()
            if len(v) == 0:
                continue
            m, lo, hi = mean_with_ci(v)
            agg_rows.append({"tag": tag, "algo": a, "n_seeds": len(v),
                             "final_mean": m, "ci_lo": lo, "ci_hi": hi})
            print(f"  {a:<10} {m:8.3f}  [{lo:.3f}, {hi:.3f}]  (n={len(v)})")

        scores = {a: sub[sub["algo"] == a]["final"].to_numpy() for a in ALGOS}
        decisive = [
            ("convergence: QMIX - GNN-QMIX", "qmix", "gnn_qmix"),
            ("graph: GNN-QMIX - MLP-QMIX", "gnn_qmix", "mlp_qmix"),
            ("capacity: MLP-QMIX - QMIX", "mlp_qmix", "qmix"),
        ]
        raw_ps, recs = [], []
        for label, a, b in decisive:
            xa, xb = scores.get(a), scores.get(b)
            if xa is None or xb is None or len(xa) < 2 or len(xb) < 2:
                continue
            delta = float(xa.mean() - xb.mean())
            raw_p = float(stats.ttest_ind(xa, xb, equal_var=False).pvalue)
            mwu_p = float(stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
            recs.append([tag, label, delta, cohend(xa, xb), raw_p, mwu_p])
            raw_ps.append(raw_p)
        if not recs:
            continue
        order = np.argsort(raw_ps)
        holm = [0.0] * len(raw_ps)
        run = 0.0
        for rank, idx in enumerate(order):
            run = max(run, min(raw_ps[idx] * (len(raw_ps) - rank), 1.0))
            holm[idx] = run
        print(f"  --- decisive contrasts ({tag}) ---")
        for rec, h in zip(recs, holm):
            tg, label, delta, d, raw_p, mwu_p = rec
            verdict_rows.append({"tag": tg, "comparison": label, "delta": delta,
                                 "cohen_d": d, "welch_raw_p": raw_p,
                                 "welch_holm_p": h, "mannwhitney_p": mwu_p,
                                 "sig_holm": h < 0.05, "sig_mwu": mwu_p < 0.05})
            flags = ("Holm*" if h < 0.05 else "     ") + (" MWU*" if mwu_p < 0.05 else "")
            print(f"    {label:<32} delta={delta:+7.3f}  d={d:+5.2f}  "
                  f"holm={h:.3f}  mwu={mwu_p:.3f}  {flags}")

    if agg_rows:
        pd.DataFrame(agg_rows).to_csv(args.results / "summary_aggregate.csv", index=False)
    if verdict_rows:
        pd.DataFrame(verdict_rows).to_csv(args.results / "confound_verdict.csv", index=False)
        print(f"\n[phase13] -> {args.results}/confound_verdict.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
