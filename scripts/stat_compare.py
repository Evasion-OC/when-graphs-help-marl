"""Pairwise comparisons across algorithms on the pilot data.

For each pair (algo_A, algo_B):
- Take per-seed peak success rate
- Take per-seed final-window success rate
- Run a Welch t-test (n=3 per arm; underpowered, but reports the direction)
- Report Cohen's d as a magnitude estimate

n=3 is too small for real significance — these stats are *advisory*, not
publication-grade. Phase 2 with 5+ seeds is where we draw conclusions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "results" / "pilot"
ALGOS = ["iql", "vdn", "qmix", "gnn_qmix"]


def per_seed_metrics() -> pd.DataFrame:
    rows = []
    for algo in ALGOS:
        for f in sorted(PILOT.glob(f"{algo}_seed*.csv")):
            seed = int(f.stem.split("seed")[-1])
            df = pd.read_csv(f)
            ev = df[df["kind"] == "eval"].copy()
            ev["eval_success_rate"] = pd.to_numeric(ev["eval_success_rate"])
            ev["eval_return_mean"] = pd.to_numeric(ev["eval_return_mean"])
            ev = ev.sort_values("episode")
            late = ev.iloc[-max(1, len(ev) // 4):]
            rows.append(
                {
                    "algo": algo,
                    "seed": seed,
                    "peak_success": ev["eval_success_rate"].max(),
                    "final_success": late["eval_success_rate"].mean(),
                    "final_return": late["eval_return_mean"].mean(),
                }
            )
    return pd.DataFrame(rows)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s_pooled = np.sqrt(((np.std(a, ddof=1) ** 2) + (np.std(b, ddof=1) ** 2)) / 2)
    if s_pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / s_pooled)


def main() -> int:
    df = per_seed_metrics()
    if df.empty:
        print(f"no results in {PILOT}/")
        return 1

    print(df.to_string(index=False))
    print()
    print("Pairwise Welch t-tests on PEAK success rate (n=3 each arm):")
    print(f"{'A vs B':<22}  {'mean(A)':>8}  {'mean(B)':>8}  {'Δ':>7}  {'t':>7}  {'p':>6}  {'d':>6}")
    for i, a in enumerate(ALGOS):
        for b in ALGOS[i + 1:]:
            xa = df[df["algo"] == a]["peak_success"].values
            xb = df[df["algo"] == b]["peak_success"].values
            if len(xa) == 0 or len(xb) == 0:
                continue
            t, p = stats.ttest_ind(xa, xb, equal_var=False)
            d = cohens_d(xa, xb)
            print(
                f"{a:>9} vs {b:<9}  {xa.mean():>8.3f}  {xb.mean():>8.3f}  "
                f"{xa.mean() - xb.mean():+7.3f}  {t:+7.2f}  {p:6.3f}  {d:+6.2f}"
            )
    print()
    print("Pairwise Welch t-tests on FINAL-WINDOW success rate:")
    print(f"{'A vs B':<22}  {'mean(A)':>8}  {'mean(B)':>8}  {'Δ':>7}  {'t':>7}  {'p':>6}  {'d':>6}")
    for i, a in enumerate(ALGOS):
        for b in ALGOS[i + 1:]:
            xa = df[df["algo"] == a]["final_success"].values
            xb = df[df["algo"] == b]["final_success"].values
            if len(xa) == 0 or len(xb) == 0:
                continue
            t, p = stats.ttest_ind(xa, xb, equal_var=False)
            d = cohens_d(xa, xb)
            print(
                f"{a:>9} vs {b:<9}  {xa.mean():>8.3f}  {xb.mean():>8.3f}  "
                f"{xa.mean() - xb.mean():+7.3f}  {t:+7.2f}  {p:6.3f}  {d:+6.2f}"
            )
    print()
    print("Note: n=3 per arm. p-values are advisory; Phase 2 (5+ seeds) is where")
    print("we draw conclusions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
