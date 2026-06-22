"""Stage C analysis -- test the pre-registered H-struct (graph STRUCTURE helps).

Reads results/phaseC/manifest_<label>.csv (written by phaseC_structure_sweep.py)
and evaluates the pre-registered decision rule:

  H-struct: gnn_true beats EACH of {gnn_wrong, gnn_complete, mlp}, significantly
            (Holm-corrected Welch t AND Mann-Whitney U, two-sided, p<0.05).
  Discriminator: does gnn_complete beat mlp? (comm without structure)

Writes results/phaseC/{summary,struct_verdict}_<label>.csv and prints the verdict.

Usage::
    python scripts/phaseC_analysis.py                  # N6, ego (default)
    python scripts/phaseC_analysis.py --label N6_full
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

from gnnmarl.utils.stats import holm_correct, mean_with_ci  # noqa: E402

ARMS = ["mlp", "gnn_true", "gnn_wrong", "gnn_complete"]
KEY = "gnn_true"
RIVALS = ("gnn_wrong", "gnn_complete", "mlp")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def final_window_mean(run_dir: str, frac: float, fallback: float) -> float:
    csv = Path(run_dir) / "episodes.csv"
    if not csv.exists():
        return fallback
    ret = pd.read_csv(csv)["return"].to_numpy(dtype=float)
    k = max(1, int(len(ret) * frac))
    return float(ret[-k:].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=REPO_ROOT / "results" / "phaseC")
    ap.add_argument("--label", type=str, default="N6_ego")
    ap.add_argument("--window", type=float, default=0.2)
    args = ap.parse_args()

    manifest = args.log_dir / f"manifest_{args.label}.csv"
    if not manifest.exists():
        print(f"ERROR: {manifest} not found. Run phaseC_structure_sweep.py first.")
        return 1
    man = pd.read_csv(manifest)
    man["final"] = [
        final_window_mean(r.run_dir, args.window, r.final) for r in man.itertuples()
    ]

    scores = {arm: man[man.arm == arm]["final"].to_numpy() for arm in ARMS
              if len(man[man.arm == arm]) > 0}

    # ---- summary ----
    print(f"\n=== Stage C [{args.label}]: final-window return mean [95% CI] ===")
    summ = []
    for arm in ARMS:
        if arm not in scores:
            continue
        m, lo, hi = mean_with_ci(scores[arm])
        summ.append({"arm": arm, "n": len(scores[arm]), "mean": m, "ci_lo": lo, "ci_hi": hi})
        print(f"  {arm:<13} {m:8.2f}  [{lo:7.2f}, {hi:7.2f}]  (n={len(scores[arm])})")
    pd.DataFrame(summ).to_csv(args.log_dir / f"summary_{args.label}.csv", index=False)

    if KEY not in scores:
        print(f"ERROR: {KEY} missing.")
        return 1

    # ---- H-struct: gnn_true vs each rival; Holm across EXACTLY this 3-comparison
    # family (the pre-registered family in docs/STAGE_C.md), not all 6 arm-pairs. ----
    rivals = [r for r in RIVALS if r in scores]
    raw_welch = [
        float(stats.ttest_ind(scores[KEY], scores[r], equal_var=False,
                              alternative="two-sided").pvalue)
        for r in rivals
    ]
    welch_holm = dict(zip(rivals, holm_correct(raw_welch)))
    print(f"\n=== H-struct: {KEY} vs each rival (Welch Holm[3-family] + MWU) ===")
    verdict_rows = []
    all_sig = len(rivals) == len(RIVALS)
    for rival in rivals:
        key_v, riv_v = scores[KEY], scores[rival]
        adv = float(key_v.mean() - riv_v.mean())
        welch_p = welch_holm[rival]
        mwu_p = float(stats.mannwhitneyu(key_v, riv_v, alternative="two-sided").pvalue)
        d = cohens_d(key_v, riv_v)
        sig = bool(adv > 0 and welch_p < 0.05 and mwu_p < 0.05)
        all_sig = all_sig and sig
        verdict_rows.append({"comparison": f"{KEY}_vs_{rival}", "advantage": adv,
                             "cohens_d": d, "welch_p_holm": welch_p, "mwu_p": mwu_p,
                             "significant": sig})
        print(f"  {KEY} - {rival:<13} = {adv:+7.2f}  d={d:5.2f}  "
              f"Welch p_holm={welch_p:.4f}  MWU p={mwu_p:.4f}  "
              f"{'<<< SIG' if sig else ''}")

    # ---- discriminator: comm without structure (gnn_complete vs mlp) ----
    disc = None
    if "gnn_complete" in scores and "mlp" in scores:
        cv, mv = scores["gnn_complete"], scores["mlp"]
        disc_p = float(stats.ttest_ind(cv, mv, equal_var=False,
                                       alternative="two-sided").pvalue)
        disc = float(cv.mean() - mv.mean())
        print(f"\n  discriminator  gnn_complete - mlp = {disc:+7.2f}  "
              f"Welch p={disc_p:.4f}  "
              f"({'comm alone helps' if disc > 0 and disc_p < 0.05 else 'comm alone does NOT help'})")
    pd.DataFrame(verdict_rows).to_csv(args.log_dir / f"struct_verdict_{args.label}.csv", index=False)

    # ---- verdict ----
    print("\n=== STAGE C VERDICT ===")
    if all_sig:
        print("  => H-struct SUPPORTED. gnn_true significantly beats the wrong graph,\n"
              "     the complete graph, AND no-comm -- at matched params/obs. Graph\n"
              "     STRUCTURE genuinely helps: message passing must follow the task's\n"
              "     coordination edges. This is a real, structure-level positive result.")
    elif disc is not None and disc > 0:
        print("  => Partial: communication helps but the CORRECT graph is not clearly\n"
              "     better than all-to-all. Speaks to comm, not structure. Report honestly.")
    else:
        print("  => NOT supported. The graph does not beat its controls here either.\n"
              "     Lock in the strengthened negative paper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
