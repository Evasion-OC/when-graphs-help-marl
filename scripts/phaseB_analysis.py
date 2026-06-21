"""Stage B analysis -- test the pre-registered H5 from the diameter sweep.

Reads results/phaseB/manifest.csv (written by phaseB_diameter_sweep.py),
recomputes each run's final-window mean return, and evaluates:

H5: A(d) = mean(gnn_repair_Ld) - max(mean(qmix), mean(mlp_qmix)) increases with
    diameter d and is significantly > 0 at the largest d.

Significance uses the same machinery as the rest of the paper: Welch's t with
Holm correction across the per-diameter family, plus a two-sided Mann-Whitney U.
Also reports the matched-depth vs fixed-depth contrast (gnn_repair_Ld vs
gnn_repair_L2) as a secondary check on the routing-depth mechanism.

Writes:
  results/phaseB/summary.csv     per (diameter, arm): final mean +/- 95% CI
  results/phaseB/h5_verdict.csv  per diameter: A, best control, p-values, sig

Usage::

    python scripts/phaseB_analysis.py
    python scripts/phaseB_analysis.py --window 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnmarl.utils.stats import mean_with_ci, pairwise_compare  # noqa: E402

KEY = "gnn_repair_Ld"
CONTROLS = ("qmix", "mlp_qmix")


def final_window_mean(run_dir: str, frac: float, fallback: float) -> float:
    csv = Path(run_dir) / "episodes.csv"
    if not csv.exists():
        return fallback
    ret = pd.read_csv(csv)["return"].to_numpy(dtype=float)
    k = max(1, int(len(ret) * frac))
    return float(ret[-k:].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=REPO_ROOT / "results" / "phaseB")
    ap.add_argument("--window", type=float, default=0.2,
                    help="trailing fraction of episodes for the final-window mean")
    args = ap.parse_args()

    manifest = args.log_dir / "manifest.csv"
    if not manifest.exists():
        print(f"ERROR: {manifest} not found. Run scripts/phaseB_diameter_sweep.py first.")
        return 1
    man = pd.read_csv(manifest)
    man["final"] = [
        final_window_mean(r.run_dir, args.window, r.final) for r in man.itertuples()
    ]

    diameters = sorted(man["diameter"].unique())
    arms = ["qmix", "mlp_qmix", "gnn_repair_L2", "gnn_repair_Ld"]

    # ---- per-(diameter, arm) summary ----
    summ_rows = []
    print("\n=== final-window return: mean [95% CI] over seeds ===")
    for d in diameters:
        print(f"  diameter d={d}")
        for arm in arms:
            vals = man[(man.diameter == d) & (man.arm == arm)]["final"].to_numpy()
            if len(vals) == 0:
                continue
            m, lo, hi = mean_with_ci(vals)
            summ_rows.append({"diameter": d, "arm": arm, "n": len(vals),
                              "final_mean": m, "ci_lo": lo, "ci_hi": hi})
            print(f"    {arm:<14} {m:8.2f}  [{lo:7.2f}, {hi:7.2f}]  (n={len(vals)})")
    pd.DataFrame(summ_rows).to_csv(args.log_dir / "summary.csv", index=False)

    # ---- H5: advantage vs diameter, with significance ----
    print("\n=== H5: graph advantage A(d) = gnn_repair_Ld - best control ===")
    verdict_rows = []
    adv_by_d: dict[int, float] = {}
    for d in diameters:
        sub = man[man.diameter == d]
        scores = {arm: sub[sub.arm == arm]["final"].to_numpy() for arm in arms
                  if len(sub[sub.arm == arm]) > 0}
        if KEY not in scores:
            continue
        key_vals = scores[KEY]
        # Pick the stronger control by mean, then test KEY vs it.
        ctrl = max((c for c in CONTROLS if c in scores), key=lambda c: scores[c].mean())
        adv = float(key_vals.mean() - scores[ctrl].mean())
        adv_by_d[d] = adv

        welch = pairwise_compare(scores, test="welch")
        welch_p = next((r.p_holm for r in welch
                        if {r.a, r.b} == {KEY, ctrl}), float("nan"))
        mwu_p = float(stats.mannwhitneyu(key_vals, scores[ctrl],
                                         alternative="two-sided").pvalue)
        sig = bool(adv > 0 and welch_p < 0.05 and mwu_p < 0.05)
        # Secondary: matched depth vs fixed shallow depth.
        depth_gain = (float(key_vals.mean() - scores["gnn_repair_L2"].mean())
                      if "gnn_repair_L2" in scores else float("nan"))
        verdict_rows.append({"diameter": d, "best_control": ctrl, "A": adv,
                             "welch_p_holm": welch_p, "mwu_p": mwu_p,
                             "significant_positive": sig,
                             "matched_minus_fixed_depth": depth_gain})
        print(f"  d={d}: A={adv:+7.2f}  vs {ctrl:<8} "
              f"Welch p_holm={welch_p:.3f}  MWU p={mwu_p:.3f}  "
              f"{'<<< SIGNIFICANT POSITIVE' if sig else ''}")
    pd.DataFrame(verdict_rows).to_csv(args.log_dir / "h5_verdict.csv", index=False)

    # ---- overall H5 call ----
    ds = sorted(adv_by_d)
    monotone = all(adv_by_d[ds[i]] <= adv_by_d[ds[i + 1]] for i in range(len(ds) - 1))
    top_sig = next((row["significant_positive"] for row in verdict_rows
                    if row["diameter"] == ds[-1]), False) if ds else False
    print("\n=== H5 VERDICT ===")
    print("  advantage by diameter: " + ", ".join(f"d{d}={adv_by_d[d]:+.2f}" for d in ds))
    print(f"  monotonically increasing in d: {monotone}")
    print(f"  significantly positive at largest d (d={ds[-1] if ds else '?'}): {top_sig}")
    if monotone and top_sig:
        print("  => H5 SUPPORTED. Genuine positive: the graph routes non-local\n"
              "     information and the benefit grows with coordination distance.")
    elif top_sig:
        print("  => Partial: graph wins at high diameter but the trend is not\n"
              "     cleanly monotone. Still a positive, scoped to high-diameter routing.")
    else:
        print("  => H5 NOT supported. The Stage-A signal did not survive a proper\n"
              "     multi-seed, fixed-N test. Lock in the strengthened negative paper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
