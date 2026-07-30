# Phase C classical coordination-graph (DCG) baseline -- confirmatory analysis

Frozen pre-registration: results/phaseC_classical/PREREGISTRATION.md (commit d7b8dd3), backed by docs/CLASSICAL_CG_BASELINE_DESIGN.md. Applied mechanically. Metric: final-window (last 20%) mean return per run, 12 seeds (0-11), 150k env steps, every cell. Primary: two-sided Welch t, Holm-corrected within each cell frozen gnn_true-centred contrast family (the scripts/phaseC_analysis.py wiring the pre-reg defers to the statistician). Robustness: two-sided Mann-Whitney U. A significance claim requires BOTH Welch(Holm) AND MWU < 0.05. Cohen d (pooled SD) and Student-t 95% CIs throughout.

Contrast family, stated explicitly (guardrail). The frozen wiring compares gnn_true against every other arm present in a cell manifest, Holm-correcting across exactly that per-cell family: A = 3 contrasts, B = 4, C = 6. All reported regardless of outcome. Contrasts outside a gnn_true-centred family are labelled EXPLORATORY. No verdict depends on the family choice: the load-bearing C crux (gnn_true vs mlp) fails the MWU robustness gate (p=0.078) independently of any Holm family.

## 1. Per-cell descriptive statistics

### Cell A - full-info co-location ring (N=4, obs=full). Random floor 4.02.

| arm | n | mean | median | sd | 95% CI |
|---|---:|---:|---:|---:|---|
| dcg_canonical | 12 | 74.873 | 75.965 | 6.477 | [70.758, 78.988] |
| dcg_jointobs | 12 | 83.572 | 83.847 | 0.930 | [82.981, 84.163] |
| gnn_true | 12 | 83.928 | 83.889 | 0.358 | [83.701, 84.156] |
| mlp | 12 | 83.655 | 84.077 | 0.932 | [83.062, 84.247] |

### Cell B - 1-hop PairRouting (N=6, ego). Floor ~50, oracle ~149. gnn_true rows = frozen 150k historical (pre-reg 2c).

| arm | n | mean | median | sd | 95% CI |
|---|---:|---:|---:|---:|---|
| dcg_canonical_true | 12 | 50.093 | 50.122 | 0.443 | [49.812, 50.374] |
| dcg_jointobs_complete | 12 | 50.303 | 50.243 | 0.299 | [50.113, 50.493] |
| dcg_jointobs_wrong | 12 | 49.891 | 49.830 | 0.442 | [49.611, 50.172] |
| dcg_jointobs_true | 12 | 90.497 | 95.422 | 16.488 | [80.021, 100.973] |
| gnn_true | 12 | 94.964 | 96.278 | 5.055 | [91.752, 98.175] |

### Cell C - 2-hop relay (N=6, ego). Random floor 33.37; no-comms MLP reference 61.11; privileged oracle 99.11.

| arm | n | mean | median | sd | 95% CI |
|---|---:|---:|---:|---:|---|
| dcg_canonical_true | 12 | 51.174 | 49.647 | 7.458 | [46.435, 55.912] |
| dcg_jointobs_true | 12 | 56.508 | 56.414 | 5.080 | [53.281, 59.736] |
| dcg_jointobs_wrong | 12 | 56.445 | 58.320 | 6.171 | [52.524, 60.366] |
| dcg_jointobs_complete | 12 | 57.095 | 58.679 | 6.059 | [53.245, 60.944] |
| gat_true | 12 | 63.156 | 63.421 | 0.837 | [62.624, 63.688] |
| gnn_true | 12 | 64.004 | 63.841 | 1.172 | [63.260, 64.749] |
| mlp | 12 | 61.114 | 63.329 | 3.808 | [58.695, 63.534] |

## 2. Pre-registered confirmatory contrast family (gnn_true-centred, Holm within cell)

### Cell A (Holm across 3 contrasts)

| contrast | adv (gnn_true - arm) | Cohen d | Welch raw | Welch Holm | MWU | verdict |
|---|---:|---:|---:|---:|---:|---|
| gnn_true vs dcg_canonical | +9.056 | +1.97 | 5.13e-04 | 0.002 | 3.66e-05 | SIG (gnn>arm) |
| gnn_true vs dcg_jointobs | +0.356 | +0.51 | 0.235 | 0.471 | 0.507 | ns / tie |
| gnn_true vs mlp | +0.274 | +0.39 | 0.358 | 0.471 | 0.840 | ns / tie |

### Cell B (Holm across 4 contrasts)

| contrast | adv (gnn_true - arm) | Cohen d | Welch raw | Welch Holm | MWU | verdict |
|---|---:|---:|---:|---:|---:|---|
| gnn_true vs dcg_canonical_true | +44.871 | +12.51 | 3.94e-12 | 1.50e-11 | 3.66e-05 | SIG (gnn>arm) |
| gnn_true vs dcg_jointobs_complete | +44.661 | +12.47 | 4.77e-12 | 1.50e-11 | 3.66e-05 | SIG (gnn>arm) |
| gnn_true vs dcg_jointobs_wrong | +45.072 | +12.56 | 3.75e-12 | 1.50e-11 | 3.66e-05 | SIG (gnn>arm) |
| gnn_true vs dcg_jointobs_true | +4.467 | +0.37 | 0.386 | 0.386 | 0.707 | ns / tie |

### Cell C (Holm across 6 contrasts)

| contrast | adv (gnn_true - arm) | Cohen d | Welch raw | Welch Holm | MWU | verdict |
|---|---:|---:|---:|---:|---:|---|
| gnn_true vs dcg_canonical_true | +12.831 | +2.40 | 8.65e-05 | 5.19e-04 | 3.66e-05 | SIG (gnn>arm) |
| gnn_true vs dcg_jointobs_true | +7.496 | +2.03 | 3.06e-04 | 0.002 | 3.66e-05 | SIG (gnn>arm) |
| gnn_true vs dcg_jointobs_wrong | +7.560 | +1.70 | 0.001 | 0.005 | 3.66e-05 | SIG (gnn>arm) |
| gnn_true vs dcg_jointobs_complete | +6.910 | +1.58 | 0.002 | 0.007 | 0.001 | SIG (gnn>arm) |
| gnn_true vs gat_true | +0.849 | +0.83 | 0.055 | 0.055 | 0.157 | ns / tie |
| gnn_true vs mlp | +2.890 | +1.03 | 0.026 | 0.052 | 0.078 | ns / tie |

## 3. Validity-gate checklist

| gate | status | evidence |
|---|---|---|
| Max-plus unit tests (exact argmax on trees, message-norm convergence) | PASS (pre-reg) | tests green at freeze; not re-run here |
| Reference points reproduce (relay floor 33.37, oracle 99.11, A floor 4.02) | PASS | reference_points.csv |
| Smoke gates (C-liveness, B-concession, A-validity) | PASS (pre-reg) | passed pre-freeze per handoff |
| DCG competent on full-info A (wins/ties before relay trusted) | PASS | dcg_jointobs ties MLP (adv -0.08, Welch 0.830, MWU 0.436) and GNN (Welch_holm 0.471); floor-clear p=4e-23 |
| gnn_true 150k reproduces ~95 on PairRouting-ego (moot - frozen anchor rows) | PASS | gnn_true B mean 94.96 (frozen phaseC_150k rows, pre-reg 2c) |
| DCG implementation validated end-to-end | PASS | matches gnn_true on 1-hop B (90.50 vs 94.96, ns); relay failure structural, not a bug |
| Rule-4 discard trigger (classical fails BOTH A and C) | NOT triggered | classical competent on A; not discarded |

## 4. Frozen decision-rule application

"succeed" (frozen) = final-window mean significantly above the cell floor and, on C, significantly above the MLP reference (pre-reg 2b/4), on BOTH Welch-Holm and MWU.

**Rule 3 (cell B) -- SATISFIED.** dcg_jointobs_true matches gnn_true (gnn adv +4.47, d=0.37, Welch p_holm=0.386, MWU p=0.707 -> not significant = matches), while dcg_canonical_true, dcg_jointobs_wrong, dcg_jointobs_complete all sit at the ~50 floor (~45 below gnn_true, d>12). Outcome: honest partial concession + mechanistic decomposition (1-hop benefit is observation-routing, reproducible by a deep CG), exactly as pre-registered. Not a win.

**Rule 1 (headline DIFFERENTIATION) -- NOT satisfied.** Conjunction: (a) dcg_jointobs fails relay C -- TRUE (56.51, below the 61.11 MLP reference); (b) gnn_true succeeds relay C -- FALSE (gnn_true 64.00 vs MLP 61.11: adv +2.89, d=1.03, Welch p_holm=0.052, MWU p=0.078; fails both required tests); (c) classical wins/ties A -- TRUE for steelman dcg_jointobs. Because precondition (b) fails, the pre-registered headline is NOT claimable.

**Rule 2 (DCG also succeeds C -> subsumed) -- NOT satisfied.** dcg_jointobs_true (56.51) is BELOW the MLP reference; it does not succeed.

**Between-outcomes statement (exact language).** On C, NONE of rules 1/2 fires because their common precondition -- that some method final-window mean is significantly above the no-comms MLP reference -- fails for BOTH gnn_true (p_holm=0.052, MWU=0.078) and every DCG arm (all below MLP). The gnn_true > dcg_jointobs_true head-to-head IS significant (+7.50, d=2.03, Welch p_holm=1.5e-3, MWU=3.7e-5), but it reflects DCG falling BELOW the no-comms baseline, not GNN routing above it. This matches the design pre-registered Backfire-2 (GNN cannot relay above the honest baseline), surfacing at confirmatory scale though the 30k smoke had passed.

## 5. Claimable statements (verbatim for the manuscript)

**[A -- validity + negative endpoint, CLAIMABLE].** On the full-information co-location ring (cell A, N=4, 150k steps, 12 seeds), the steelman DCG dcg_jointobs is fully competent: it ties MLP-QMIX (83.57 vs 83.66; Welch p=0.83, MWU p=0.44, d=-0.09) and GNN-QMIX (83.93; Welch p_holm=0.47, MWU p=0.51), all far above the random floor of 4.02 (p=4e-23). GNN-QMIX shows no advantage over MLP-QMIX on full information (+0.27, Welch p_holm=0.47, MWU p=0.84), consistent with the study no-free-lunch endpoint. The action-coupling-only control dcg_canonical is significantly worse than all three (-9.06 vs GNN, d=1.97, Welch p_holm=1.5e-3, MWU p=3.7e-5; -8.78 vs MLP) yet still far above floor -- dropping cross-agent observation from the pairwise factor costs ~9 points even on DCG home turf.

**[B -- 1-hop concession, CLAIMABLE].** On the 1-hop PairRouting task (cell B, N=6 ego, 150k, 12 seeds), the steelman DCG on the true graph matches GNN-QMIX: dcg_jointobs_true reaches 90.50 [80.02, 100.97] vs gnn_true 94.96 [91.75, 98.18] (advantage +4.47, d=0.37, Welch p_holm=0.386, MWU p=0.707 -- not significant). Every DCG variant lacking correct observation-routing collapses to the ~50 no-routing floor (dcg_canonical_true 50.09, dcg_jointobs_wrong 49.89, dcg_jointobs_complete 50.30; each ~45 points below gnn_true, d>12, p_holm<1e-10). The 1-hop benefit is therefore an observation-routing effect reproducible by a deep coordination graph, not action-coordination. We concede the 1-hop cell, as pre-registered (decision rule 3).

**[C -- 2-hop relay, CLAIMABLE honest version].** On the 2-hop relay (cell C, N=6 ego, 150k, 12 seeds; random floor 33.37, no-comms MLP reference 61.11, privileged oracle 99.11), GNN-QMIX on the true graph does NOT significantly exceed the honest no-comms MLP-QMIX baseline: gnn_true 64.00 [63.26, 64.75] vs mlp 61.11 [58.70, 63.53], advantage +2.89, d=1.03, Welch p_holm=0.052, MWU p=0.078 -- failing both required tests. The mean advantage is a tail artifact of four unstable MLP seeds (53.6-58.2); the medians nearly coincide (gnn_true 63.84 vs mlp 63.33, gap 0.51), and GAT-QMIX on the true graph likewise fails to beat MLP (+2.04, MWU p=0.51). By the pre-registered success criterion, gnn_true does NOT succeed on the relay. GNN-QMIX does significantly beat every DCG variant (vs dcg_jointobs_true +7.50, d=2.03, Welch p_holm=1.5e-3, MWU p=3.7e-5; also vs wrong, complete, canonical), but this reflects the DCG arms falling BELOW the no-comms MLP baseline (dcg_jointobs_true 56.51 = -12% of routing headroom relative to MLP), not GNN-QMIX routing above it. Neither message-passing nor payoff-propagation demonstrates multi-hop information routing above the no-routing baseline at 150k steps.

**NOT claimable (overclaim guardrail):**
- Differentiation demonstrated / double dissociation / the Rule-1 headline -- precondition (gnn_true succeeds on C) fails.
- GNN-QMIX solves / succeeds at 2-hop routing -- it does not exceed the no-comms MLP.
- Message passing routes multi-hop information where payoff propagation cannot (absolute form). The defensible version is the relative ordering gnn_true > DCG on the true graph, with both methods at or below the no-comms baseline.
- Any C claim resting on gnn_true beating the random floor (a-priori true; MLP clears it too) or on gnn_true > DCG (informative for gnn-vs-dcg ordering, uninformative for the routing claim since driven by DCG being below MLP).

## 6. Exploratory (NOT in the confirmatory family)

### 6a. Cell C mlp-centred contrasts (Holm across 6) -- "extra machinery hurts"

| contrast | adv (mlp - arm) | d | Welch raw | Welch Holm | MWU |
|---|---:|---:|---:|---:|---:|
| mlp vs gnn_true | -2.890 | -1.03 | 0.026 | 0.104 | 0.078 |
| mlp vs gat_true | -2.042 | -0.74 | 0.095 | 0.134 | 0.507 |
| mlp vs dcg_jointobs_true | +4.606 | +1.03 | 0.020 | 0.102 | 0.014 |
| mlp vs dcg_jointobs_wrong | +4.669 | +0.91 | 0.038 | 0.115 | 0.017 |
| mlp vs dcg_jointobs_complete | +4.019 | +0.79 | 0.067 | 0.134 | 0.069 |
| mlp vs dcg_canonical_true | +9.941 | +1.68 | 7.81e-04 | 4.68e-03 | 7.32e-04 |

Reading: DCG payoff-propagation machinery underperforms a plain no-comms MLP-QMIX on the relay. dcg_canonical_true is significantly below MLP (-9.94, d=1.68, Welch p_holm=4.7e-3, MWU=7.3e-4). dcg_jointobs_true is directionally below MLP (-4.61, d=1.03, raw Welch=0.020 and MWU=0.014 both <0.05, but Welch p_holm=0.102 in the 6-family, so not confirmatory-significant). gnn_true and gat_true do not significantly exceed MLP. Median ordering: gnn ~ gat ~ mlp > dcg_jointobs ~ dcg_wrong ~ dcg_complete > dcg_canonical.

### 6b. Cell C within-DCG structure and routing headroom

Graph structure does not move DCG on the relay: dcg_jointobs_true vs wrong +0.06 (ns), vs complete -0.59 (ns). The complete graph does NOT rescue DCG via an s-t shortcut here (complete 57.10, still 4.02 below MLP, Welch 0.067) -- the pre-registered Backfire-4 shortcut did not materialise at 150k. Routing headroom captured above the MLP reference, (mean-61.11)/(99.11-61.11): gnn_true +7.6%, gat_true +5.4%, dcg_jointobs_true -12.1%, dcg_jointobs_complete -10.6%.

### 6c. Cell B dcg_jointobs_true dispersion (preview flagged sd~16.5 as possible bimodality)

Seed-level (sorted): [65.3, 67.0, 72.2, 79.5, 86.7, 95.0, 95.8, 96.4, 100.4, 103.5, 106.6, 117.4]. Not bimodal: all 12/12 seeds route (min 65.3, none within 15 of the ~50 floor; 0/12 below 55). The high sd (16.49) is spread among solvers, not a solve-vs-floor split. Exploratory success-count view: 12/12 clearly above floor. The preview suspected bimodality is not borne out; do not report a bimodal split.

### 6d. Cell A dcg-vs-mlp (validity detail)

dcg_jointobs vs mlp: -0.08, d=-0.09, Welch 0.830, MWU 0.436 (tie). dcg_canonical vs mlp: -8.78, d=-1.90, Welch 6.3e-4, MWU 7.7e-5 (canonical significantly worse, but floor-clear at 74.87, p=2.6e-13).

### 6e. Scope note

Cell C has no gnn_wrong/gnn_complete arms, so a within-GNN structure-helps claim is not testable on C even in principle; C speaks only to gnn-vs-dcg and gnn-vs-mlp.
