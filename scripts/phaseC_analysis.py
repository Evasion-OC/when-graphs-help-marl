"""Stage C analysis -- test the pre-registered structure hypotheses.

Generic over the Stage C sweeps (any --label written by phaseC_structure_sweep.py).
Reports per-arm final-window return (mean, 95% CI) and compares gnn_true against
EVERY other arm present, Holm-correcting across exactly that family (the
pre-registered family in docs/STAGE_C.md), with two-sided Welch t + Mann-Whitney U
and Cohen's d.

  core suite (mlp/gnn_true/gnn_wrong/gnn_complete):
      H-struct holds iff gnn_true beats wrong AND complete AND mlp, significantly.
  attention suite (adds gat/dgn on complete):
      reports whether learned attention over all-to-all (gat_complete/dgn_complete)
      reaches gnn_true -- i.e. whether attention can substitute for the right graph.

Usage::
    python scripts/phaseC_analysis.py --label N6_ego          # committed pair result
    python scripts/phaseC_analysis.py --label nbr_N6_ego      # neighbourhood task
    python scripts/phaseC_analysis.py --label nbr_N6_ego_attention
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

KEY = "gnn_true"
# Preferred display order; any extra arms are appended in first-seen order.
ARM_ORDER = ["mlp", "gnn_true", "gnn_wrong", "gnn_complete",
             "gcn_complete", "gat_complete", "dgn_complete", "gat_true", "dgn_true"]


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
    present = list(man["arm"].unique())
    arms_ordered = [a for a in ARM_ORDER if a in present] + \
                   [a for a in present if a not in ARM_ORDER]
    scores = {a: man[man.arm == a]["final"].to_numpy() for a in arms_ordered}

    print(f"\n=== Stage C [{args.label}]: final-window return mean [95% CI] ===")
    summ = []
    for a in arms_ordered:
        m, lo, hi = mean_with_ci(scores[a])
        summ.append({"arm": a, "n": len(scores[a]), "mean": m, "ci_lo": lo, "ci_hi": hi})
        print(f"  {a:<13} {m:8.2f}  [{lo:7.2f}, {hi:7.2f}]  (n={len(scores[a])})")
    pd.DataFrame(summ).to_csv(args.log_dir / f"summary_{args.label}.csv", index=False)

    if KEY not in scores:
        print(f"ERROR: {KEY} not in manifest arms {present}.")
        return 1

    rivals = [a for a in arms_ordered if a != KEY]
    raw_welch = [
        float(stats.ttest_ind(scores[KEY], scores[r], equal_var=False,
                              alternative="two-sided").pvalue)
        for r in rivals
    ]
    welch_holm = dict(zip(rivals, holm_correct(raw_welch)))

    print(f"\n=== {KEY} vs each other arm (Welch Holm[{len(rivals)}-family] + MWU) ===")
    verdict_rows = []
    core_rivals = [r for r in ("gnn_wrong", "gnn_complete", "mlp") if r in scores]
    all_core_sig = len(core_rivals) == 3
    for r in rivals:
        kv, rv = scores[KEY], scores[r]
        adv = float(kv.mean() - rv.mean())
        wp = welch_holm[r]
        mwu = float(stats.mannwhitneyu(kv, rv, alternative="two-sided").pvalue)
        d = cohens_d(kv, rv)
        sig = bool(adv > 0 and wp < 0.05 and mwu < 0.05)
        if r in core_rivals:
            all_core_sig = all_core_sig and sig
        verdict_rows.append({"comparison": f"{KEY}_vs_{r}", "advantage": adv,
                             "cohens_d": d, "welch_p_holm": wp, "mwu_p": mwu,
                             "significant": sig})
        print(f"  {KEY} - {r:<13} = {adv:+7.2f}  d={d:6.2f}  "
              f"Welch p_holm={wp:.4f}  MWU p={mwu:.4f}  {'<<< SIG' if sig else ''}")
    pd.DataFrame(verdict_rows).to_csv(args.log_dir / f"struct_verdict_{args.label}.csv",
                                      index=False)

    # ---- verdicts ----
    print("\n=== VERDICT ===")
    if {"gnn_wrong", "gnn_complete", "mlp"}.issubset(scores):
        if all_core_sig:
            print("  H-struct SUPPORTED: gnn_true beats the wrong graph, the complete\n"
                  "  graph, AND no-comm at matched params/obs. Graph STRUCTURE helps --\n"
                  "  message passing must follow the task's coordination topology.")
        else:
            print("  H-struct NOT fully supported: gnn_true does not significantly beat\n"
                  "  all of {wrong, complete, mlp}. Report honestly.")
    # attention read-out (power-gated: do not assert "recovers" on an underpowered
    # or confounded comparison). If gat_true itself floors, the attention arms are
    # undertrained/unstabilized and the gat_complete comparison is uninformative.
    gat_true_floored = (
        "gat_true" in scores and "mlp" in scores
        and float(scores["gat_true"].mean() - scores["mlp"].mean()) < 3.0
    )
    for att in ("gat_complete", "dgn_complete"):
        if att in scores:
            gap = float(scores[KEY].mean() - scores[att].mean())
            wp = welch_holm[att]
            n = min(len(scores[KEY]), len(scores[att]))
            if n < 10:
                print(f"  attention: {att} gap {gap:+.1f} (p_holm={wp:.3f}) "
                      f"-> INCONCLUSIVE (underpowered, n={n}).")
            elif gat_true_floored:
                print(f"  attention: {att} gap {gap:+.1f} (p_holm={wp:.3f}); but gat_true is "
                      f"at the floor -> attention arms UNDERTRAINED/confounded (lack the GCN's "
                      f"residual+LayerNorm); comparison inconclusive.")
            elif gap > 0 and wp < 0.05:
                print(f"  attention: gnn_true still beats {att} (+{gap:.1f}, p_holm={wp:.3f})"
                      f" -> learned attention does NOT substitute for structure.")
            else:
                print(f"  attention: {att} reaches gnn_true (gap {gap:+.1f}, p_holm={wp:.3f})"
                      f" -> attention can LEARN the structure from all-to-all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
