# Phase D — frozen pre-registered analysis (E1, E0, cell R)

Statistician's report. Applies **only** the analysis frozen in
`results/phaseD_external/PREREGISTRATION.md` (base + AMENDMENT + AMENDMENT 1 +
gate verdict) and the cell-R **ADDENDUM** to
`results/phaseC_classical/PREREGISTRATION.md`. Where this document and any
narrative disagree, the pre-registrations win.

Test wiring reuses `gnnmarl.utils.stats` (`holm_correct`, `mean_with_ci`) and the
`scripts/phaseC_analysis.py` conventions (Welch two-sided + Holm over exactly the
registered family, two-sided Mann–Whitney U as the robustness test, Cohen's d
pooled-SD, Student-t 95% CIs). **Significance requires BOTH tests** (house rule).

---

## 0. Validity checks (run before any result was interpreted)

### 0.1 Commit-order validity — PASS

| commit | date | content |
|---|---|---|
| `568e759` | 2026-07-31 00:46:38 | base Phase-D pre-reg **+** cell-R ADDENDUM |
| `0278112`, `3a2b1f0`, `c1185ff` | 2026-07-31 | smoke amendment, A.6 correction, A.7 plumbing check |
| `9369e45` | 2026-07-31 | AMENDMENT 1 (authorises 150k re-smoke + `oracle_pos`) |
| `b00e26b` | 2026-07-31 02:07:37 | AMENDMENT 1 **gate verdict PASS**; budget frozen at 150k |
| `b5d76f9` | 2026-07-31 03:08:10 | **all confirmatory data** (E1, E0, cell R, re-smoke, references) |

`git merge-base --is-ancestor` confirms both `568e759` and `b00e26b` are ancestors
of `b5d76f9`. Every pre-registration and the budget freeze are strictly earlier in
both wall-clock and ancestry than every datum analysed here. **No forking path via
post-hoc pre-reg editing is possible.**

### 0.2 Metric fidelity — PASS

Recomputed the pre-registered metric (mean return over the last 20% of episodes,
one scalar per run) directly from each run's `episodes.csv` for all 60 E1 runs and
compared against the manifest `final` column:
`max |manifest.final − last-20%-window| = 1.42e-14`. The manifest metric **is** the
pre-registered metric. All runs: 6000 episodes, 150000 env steps.

### 0.3 Protocol-match audit — PASS

Read `config.json` for all 60 E1 + 12 E0 + 24 cell-R runs.

- **One** distinct trainer signature per cell and across all three cells:
  `total_env_steps=150000`, `eps_anneal_steps=120000`, `eps_start=1.0`,
  `eps_end=0.05`, `lr=5e-4`, `buffer_capacity=50000`, `batch_size=32`,
  `target_update_interval=200`, `gamma=0.99`.
- `gnn_true` / `gnn_wrong` / `gnn_complete` have **byte-identical** `algo_kwargs`
  (`gnn_residual=True, gnn_layernorm=True, gnn_layers=2, gnn_hidden=64`), differing
  **only** in the `graph` kwarg — exactly as pre-registered (§1a param audit:
  38534 params each in E1; 23942 each in cell R).
- Seed counts matched **within** every cell (E1: 12 per arm × 5 arms; E0: 6 per arm;
  cell R: 12 per arm). E0's n=6 vs E1's n=12 is pre-registered asymmetry across a
  *non-confirmatory* vs confirmatory cell (§3), not within any one table.

**No protocol asymmetry, no seed-budget forking path.**

### 0.4 Gate 1 (privacy) re-verified at the frozen 150k budget — PASS

Pre-reg §9 requires gates 1–2 re-verified at the confirmatory budget, not only at
smoke. Applying A.3's Trap-1 logic with the committed scripted references:

| reference | value |
|---|---:|
| random floor | −84.81 |
| do-nothing floor | −78.14 |
| `greedy_centroid` (identity-**blind** scripted ceiling) | −53.74 |
| `greedy_target` (position-**omniscient** scripted ceiling) | −31.64 |
| **`mlp` @150k** | mean −73.42, **median −63.58** |
| **`oracle` @150k** | mean −43.36, median −42.81 |

`mlp` sits **below** the blind ceiling (−63.58 < −53.74) and nowhere near the
informed ceiling. It is in the "blind" performance band, exactly as at 30k.
**Privacy confirmed at the confirmatory budget; no leak.**

### 0.5 Gate 2 (oracle headroom) at 150k — PASS (governs all branches)

| contrast | advantage | 95% CI (diff) | d | Welch p | MWU p |
|---|---:|---|---:|---:|---:|
| `oracle` − `mlp` | **+30.06** | [+17.67, +42.45] | **+2.16** | 1.95e-4 | 3.66e-5 |

Median gap +20.77. Both tests < 0.05. **Oracle gate PASSES** — the task is
learnable-when-routed at 150k and the metric has headroom. Per pre-reg §2 the
oracle contrast is a **descriptive anchor / gate**, deliberately **outside** the
Holm family; no correction is applied to it and none is required.

*Answer to AMENDMENT 1's own framing question (A.6):* `oracle` (color injection,
−43.36) now sits clearly **above** the blind scripted ceiling (−53.74) and part-way
toward the position-omniscient ceiling (−31.64). The color→landmark-index
indirection **is** learnable at 150k. A.6's ambiguity resolves in favour of
**(i) budget shortfall**, not (ii) an ungrounded indirection.

---

## 1. Cell E1 — confirmatory (N=6, comm off, 12 seeds, 150k)

### 1.1 Per-arm descriptives (medians mandatory — the arms are tail-heavy)

| arm | n | mean | **median** | sd | 95% CI (Student-t) | range | seeds < random floor |
|---|--:|---:|---:|---:|---|---|--:|
| `oracle` (anchor) | 12 | −43.36 | −42.81 | 4.00 | [−45.90, −40.82] | [−51.28, −38.65] | 0/12 |
| `mlp` | 12 | −73.42 | **−63.58** | 19.28 | [−85.67, −61.16] | [−122.77, −61.23] | 2/12 |
| `gnn_true` | 12 | −109.38 | −108.73 | 22.08 | [−123.41, −95.35] | [−148.30, −73.61] | **11/12** |
| `gnn_wrong` | 12 | −112.16 | −117.94 | 18.10 | [−123.66, −100.67] | [−134.26, −72.33] | **10/12** |
| `gnn_complete` | 12 | −123.99 | −128.68 | 26.55 | [−140.86, −107.13] | [−158.55, −65.51] | **11/12** |

Floor band (pre-reg §4.1): **[−84.81, −78.14]**. All three GNN arms finish
**below the random-policy floor** in the large majority of seeds.

**Headroom fraction** `(arm − mlp)/(oracle − mlp)`, denominator +30.06 (pre-reg §0
mandates reporting this):

| arm | headroom fraction (mean-based) | headroom fraction (median-based) |
|---|---:|---:|
| `gnn_true` | **−1.196** | **−2.17** |
| `gnn_wrong` | **−1.289** | **−2.62** |
| `gnn_complete` | **−1.683** | **−3.13** |

Median-based denominator = +20.77 (−42.81 − (−63.58)). Both are reported because
the rest of this document pairs every E1 mean with a median; the **mean-based
figures are the conservative ones**, for the same reason as §1.2 (`mlp`'s mean is
depressed by its own contaminated seeds). All six are negative — the GNN arms do
not capture headroom, they consume it.

### 1.2 The frozen confirmatory family (exactly three contrasts, `gnn_true`-centred)

Family = {`gnn_true`−`gnn_wrong`, `gnn_true`−`gnn_complete`, `gnn_true`−`mlp`}.
Holm across **exactly these three**, m=3. `oracle` is not in the family (§2).

| contrast | advantage | 95% CI (diff) | Cohen's d | Welch p (raw) | Welch p (Holm) | MWU p | verdict |
|---|---:|---|---:|---:|---:|---:|---|
| `gnn_true` − `gnn_wrong` | +2.78 | [−14.35, +19.91] | +0.14 | 0.7389 | 0.7389 | 0.6236 | **null** |
| `gnn_true` − `gnn_complete` | +14.61 | [−6.10, +35.33] | +0.60 | 0.1572 | 0.3145 | 0.1260 | **null / underpowered** (d=0.60 at n=12) |
| `gnn_true` − `mlp` | **−35.96** | [−53.53, −18.40] | **−1.73** | 3.39e-4 | **1.017e-3** | **7.32e-4** | **significant — in the NEGATIVE direction** |

Median-based gaps (mandated alongside means): `gnn_true`−`mlp` **−45.15**,
`gnn_wrong`−`mlp` −54.37, `gnn_complete`−`mlp` −65.10.

**Note the direction of bias.** `mlp`'s mean (−73.42) is dragged down by its own
3 contaminated seeds (§4.3); its median is −63.58. The median gap (−45.15) is
therefore **worse** than the mean gap (−35.96) — i.e. the pre-registered mean-based
statistic **understates** `gnn_true`'s deficit. No seed exclusion is pre-registered
and none is applied; all 12 seeds are reported.

### 1.3 Mechanical application of the frozen decision branches

Oracle gate PASSES (§0.5), so the branches are evaluated.

| branch | antecedent | obtains? |
|---|---|---|
| **0** — oracle fails | oracle does not beat mlp | **NO** — gate passed |
| **1** — full replication | `gnn_true` beats complete AND wrong AND mlp, both tests, Holm | **NO** — beats none |
| **2a** — ties complete, beats wrong+mlp | — | **NO** — beats neither wrong nor mlp |
| **2b** — beats complete+wrong, weak vs mlp | — | **NO** — beats neither |
| **3** — TRUE NULL (oracle passes, `gnn_true` ties `mlp`) | — | **GOVERNING — see below** |

**Branch 3 is the governing branch, by entailment rather than literal match.**
Branch 3's antecedent as frozen reads "`gnn_true` ties `mlp`". That antecedent is
**not literally satisfied**: `gnn_true` is significantly **worse** than `mlp`
(−35.96, d=−1.73, Welch p_Holm = 1.02e-3, MWU p = 7.3e-4 — both pre-registered
tests significant). Branch 3 governs because its operative consequence follows
**a fortiori**: "`gnn_true` does not exceed the no-graph control" is *entailed* by
"`gnn_true` is significantly below the no-graph control". Branches 0, 1, 2a and 2b
are each excluded on their own antecedents. **No branch is stretched and no new
claim is minted**: the reportable finding stays inside Branch 3's claim
boundary — *does not exceed the no-graph control* — and the excess (significantly
below, and below the random floor) is reported as a **characterised-instability
finding** (§4), not as a claim that graph structure is harmful in general.

Branch 3's frozen consequence therefore applies: **genuine failure to transfer on
this vehicle; the external-validity claim is explicitly rescoped to the
constructed archetypes (CoordGrid + TokenMatch + deep-CG), which the positive
endpoint stands on independently of this vehicle.** The authors pre-committed
(design §10.2, pre-reg §5) to publishing this branch if it landed. It landed.

### 1.4 Robustness of the branch verdict to the instability (EXPLORATORY, labelled)

The obvious referee objection is "`gnn_true` would have beaten `mlp` if it had not
diverged." Answered with a **post-hoc, exploratory, non-pre-registered** sensitivity:
replace the frozen final-window metric with each run's **best** rolling-100-episode
window (peak over the whole run) — a maximally generous readout for a diverging arm.
Single contrast, no Holm, exploratory label mandatory.

| arm | peak-window mean | median | 95% CI |
|---|---:|---:|---|
| `oracle` | −40.24 | −39.73 | [−42.46, −38.03] |
| `mlp` | **−59.64** | −59.67 | [−60.88, −58.39] |
| `gnn_true` | −68.03 | −67.11 | [−69.96, −66.10] |
| `gnn_complete` | −70.11 | −70.66 | [−72.29, −67.92] |
| `gnn_wrong` | −70.38 | −70.53 | [−72.40, −68.37] |

`gnn_true` − `mlp` on the peak metric: **−8.39, d = −3.29, Welch p = 1.67e-7,
MWU p = 3.66e-5.** The peak statistic is **biased in `gnn_true`'s favour** (a max
over a noisier curve is inflated more), and `gnn_true` still loses decisively.
**The Branch-3 verdict is invariant to the divergence:** `gnn_true` never exceeded
`mlp` at any point in training, in any seed-aggregate. This is exploratory support
for the frozen conclusion, not a substitute for it.

---

## 2. Cell E0 — honesty anchor (N=2, comm ON, 6 seeds, 150k) — NON-CONFIRMATORY

No Holm family applies (pre-reg §7: 2 arms, descriptive only). No headline is gated
on E0.

| arm | n | mean | median | sd | 95% CI | range |
|---|--:|---:|---:|---:|---|---|
| `mlp` | 6 | −33.25 | −26.93 | 15.34 | [−49.34, −17.15] | [−53.13, −20.15] |
| `gnn` | 6 | −45.86 | −48.33 | 7.49 | [−53.72, −38.01] | [−53.05, −36.10] |

Contrast (descriptive): `gnn` − `mlp` = **−12.62**, 95% CI [−28.97, +3.74],
**d = −1.05**, Welch p = 0.1116, MWU p = 0.2403.

E0 reference points: random **−28.25**, do-nothing **−26.00**.

**Verdict: the statistical tie holds (neither test significant), but it is
underpowered (n=6, d=−1.05) AND its precondition fails.** Pre-reg §9 bullet 2 lists
as a *validity gate*: "unmodified N=2 `simple_reference` (E0, comm on) produces
non-degenerate learning for `gnn`/`mlp`." At the final window:

- `gnn` mean −45.86 is **17.6 points below the random floor** (−28.25); 6/6 seeds
  show a grad-norm blowup and a peak→final drop of 12.1–28.9 points.
- `mlp` mean −33.25 is **below the random floor**; its median (−26.93) sits *inside*
  the floor band [−28.25, −26.00]. Only 3/6 `mlp` seeds finish clearly above floor
  (−20.15, −20.58, −23.30); 3/6 blow up (seeds 3, 4 drop 30.6 / 30.1 points).
- Both arms **did** learn mid-run: `gnn` peaks −22.9 to −26.2, `mlp` peaks −17.5 to
  −22.5, all above the random floor — then lose it.

**Consequence: E0 cannot carry its frozen honesty-anchor sentence.** The sentence
("the graph is redundant when the env supplies its own channel") presupposes that
both arms converge to a common competent level; a statistical tie between two
*post-divergence* states straddling a random-policy floor is **shared failure, not
redundancy**. The sentence goes on the NOT-claimable list. What E0 *does*
legitimately supply is a second, independent instance of the same instability
(§4.4) — under comm ON, at N=2, in a cell where routing is not even required.

---

## 3. Cell R — relay addendum (governed by the phaseC ADDENDUM)

Cell R's `gnn_wrong`/`gnn_complete` (12 seeds, 150k) are spliced into cell C's
committed manifest. `gnn_true` (64.00) and `mlp` (61.11) are cell C's
**already-committed, unmodified** rows. References: random floor 33.37,
privileged oracle 99.11.

### 3.1 Per-arm descriptives (mean paired with median throughout — standing cell-C discipline)

| arm | n | mean | **median** | sd | 95% CI | range |
|---|--:|---:|---:|---:|---|---|
| `gnn_true` (cell C, committed) | 12 | **64.00** | 63.84 | 1.17 | [63.26, 64.75] | [62.58, 66.34] |
| `gnn_complete` (cell R) | 12 | **63.98** | 63.94 | 0.45 | [63.70, 64.27] | [63.27, 64.97] |
| `gnn_wrong` (cell R) | 12 | **63.28** | 63.15 | 0.55 | [62.93, 63.63] | [62.57, 64.28] |
| `gat_true` | 12 | 63.16 | 63.42 | 0.84 | [62.62, 63.69] | [61.71, 64.35] |
| `mlp` (cell C, committed) | 12 | **61.11** | **63.33** | 3.81 | [58.69, 63.53] | [53.63, 64.05] |
| `dcg_jointobs_complete` | 12 | 57.09 | 58.68 | 6.06 | [53.25, 60.94] | [43.91, 63.54] |
| `dcg_jointobs_true` | 12 | 56.51 | 56.41 | 5.08 | [53.28, 59.74] | [46.47, 61.95] |
| `dcg_jointobs_wrong` | 12 | 56.44 | 58.32 | 6.17 | [52.52, 60.37] | [42.45, 61.98] |
| `dcg_canonical_true` | 12 | 51.17 | 49.65 | 7.46 | [46.44, 55.91] | [38.44, 62.30] |

**The mean-vs-median tension, stated explicitly.** `mlp`'s mean 61.11 is depressed
by **four** unstable seeds (53.63, 55.80, 57.04, 58.17 — the next value is 63.10, a
4.93-point gap; the other eight seeds average 63.59). This matches the committed
manuscript, which already describes the cell-C mean advantage as "a tail artifact
of four unstable \mlpqmix\ seeds" (`paper/main.tex` L1922). Its **median is 63.33**, sitting squarely
inside the GNN cluster (63.28–64.00). The three GNN arms cluster within **0.72
points of each other** with very small dispersion (sd 0.45–1.17).

### 3.2 Contrast family (the ADDENDUM defines none of its own)

The ADDENDUM is explicit: cell R's rows "simply become additional arms in cell C's
existing `gnn_true`-vs-every-arm comparison." Family = `gnn_true` vs every merged
arm = **8 contrasts**, Holm m=8.

| contrast | advantage | 95% CI (diff) | d | Welch p (raw) | Welch p (Holm, m=8) | MWU p | verdict |
|---|---:|---|---:|---:|---:|---:|---|
| `gnn_true` − `gnn_wrong` | +0.73 | [−0.07, +1.52] | +0.79 | 0.0698 | 0.1641 | 0.0690 | **null** |
| `gnn_true` − `gnn_complete` | +0.02 | [−0.75, +0.80] | +0.02 | 0.9535 | 0.9535 | 0.6236 | **null** |
| `gnn_true` − `mlp` | +2.89 | [+0.41, +5.37] | +1.03 | 0.0259 | 0.1035 | 0.0783 | **null** (fails both) |
| `gnn_true` − `gat_true` | +0.85 | [−0.02, +1.72] | +0.83 | 0.0547 | 0.1641 | 0.1572 | **null** |
| `gnn_true` − `dcg_jointobs_true` | +7.50 | [+4.22, +10.77] | +2.03 | 3.06e-4 | **2.14e-3** | **3.66e-5** | **significant** |
| `gnn_true` − `dcg_jointobs_wrong` | +7.56 | [+3.60, +11.52] | +1.70 | 1.35e-3 | **8.10e-3** | **3.66e-5** | **significant** |
| `gnn_true` − `dcg_jointobs_complete` | +6.91 | [+3.02, +10.80] | +1.58 | 2.25e-3 | **1.13e-2** | **1.11e-3** | **significant** |
| `gnn_true` − `dcg_canonical_true` | +12.83 | [+8.06, +17.60] | +2.40 | 8.65e-5 | **6.92e-4** | **3.66e-5** | **significant** |

**Family-size invariance check (important — the family grew *after* cell C's data
were committed).** Re-run on cell C's *original* 6-rival family (m=6, without cell R):
`gnn_true`−`mlp` p_Holm = **0.0517** (vs 0.1035 at m=8), MWU p = 0.0783 unchanged.
Because the **uncorrected MWU already fails** (0.0783 > 0.05), the "`gnn_true` ≈
`mlp`" conclusion is **invariant to family size** under the both-tests house rule.
All four DCG contrasts remain significant in both families (m=6 p_Holm 5.2e-4 to
6.8e-3). **Cell C's committed headline is untouched by cell R.**

**Toolchain-validity check (free, and worth stating).** The m=6 recomputation
returns `gnn_true`−`mlp` d = **1.03**, Welch p_Holm = **0.0517**, MWU p = **0.0783**.
The committed manuscript publishes exactly these three values —
"$d{=}1.03$, Welch $p_{	ext{Holm}}{=}0.052$, MWU $p{=}0.078$"
(`paper/main.tex` L1921–1923). **This analysis pipeline independently reproduces
the already-published cell-C statistics to the printed precision**, which
substantiates the rest of the numbers in this report.

### 3.3 Which frozen expectation pattern obtains

| pattern | test | obtains? |
|---|---|---|
| **1** — all GNN arms at/below the ~61 `mlp` reference → **within-GNN null** | no significant within-GNN differences; none clears `mlp` | **YES, on substance** |
| **2** — `gnn_true` > wrong ≈ complete but ≈ `mlp` ("topology-sensitive but sub-baseline") | requires `gnn_true` > `gnn_wrong` | **NO** (+0.73, p_Holm 0.164, MWU 0.069 — ns) |
| **3** — `gnn_true` suddenly clears `mlp` → drift anomaly, flag | requires `gnn_true` > `mlp` on both tests | **NO** (p_Holm 0.104, MWU 0.078 — fails both) |

**Pattern 1 obtains.** The discriminator is the three contrasts, all null — not a
mean-vs-mean eyeball against the numeral "~61". Pattern 1's literal wording ("all
GNN arms at/below the ~61 `mlp` reference") is not satisfied on *means* (63.28–64.00
> 61.11), but that numeral is a **mean artifact**: `mlp`'s median is 63.33, level
with the GNN cluster, and the GNN arms do not significantly clear `mlp` under the
frozen both-tests rule. Pattern 1's substantive content — a characterised
**within-GNN null**, no multi-hop routing claim, cell C's headline unchanged — is
exactly what the data show. **No drift anomaly is flagged** (pattern 3 does not
obtain); no re-run of `gnn_true` is triggered.

---

## 4. E1/E0 instability characterisation — EXPLORATORY, LABELLED

Not pre-registered. Descriptive characterisation only. Mechanism is a **labelled
hypothesis**, not a finding: the read-only diagnostic could not ablate it.

### 4.1 Instability onset per seed (grad-norm criterion)

Onset = first episode at which the rolling-50-episode median `grad_norm` exceeds
**5× its own baseline** (median over episodes 1000–2000). `eps_anneal_steps=120000`
= episode 4800 = **80% through the run**.

| arm | seeds with blowup | onset BEFORE 120k anneal-completion | median onset (step) | onset as % of budget | median peak→final drop | median max loss | max loss |
|---|--:|--:|--:|--:|--:|--:|--:|
| `gnn_complete` | **10/12** | **10/10** | 62 350 | **42%** | 59.5 | 2.51e4 | **3.41e6** |
| `gnn_wrong` | **9/12** | **9/9** | 83 250 | **56%** | 45.9 | 692 | 1.17e4 |
| `gnn_true` | **8/12** | 5/8 | 98 025 | **65%** | 36.5 | 72.3 | 5.58e3 |
| `mlp` | **2/12** | 0/2 | 140 750 | 94% | 4.7 | 12.8 | 70.2 |
| `oracle` | **0/12** | — | — | — | 3.0 | 8.07 | 10.2 |

Per-seed `gnn_true` return-curve onsets (rolling-100 return falling 10 below peak),
seeds 0–11: steps 95 525 / 128 175 / 100 475 / 78 500 / 85 550 / 62 650 / 78 675 /
86 775 / 105 025 / 46 600 / 64 600 / 43 925 — i.e. **11 of 12 seeds turn over before
the 120 000-step anneal completion**. No `NaN` or `Inf` in any return series in any
arm.

### 4.2 Three of the diagnostic's *descriptive* claims are contradicted by the curves

The read-only diagnostic's **no-implementation-defect verdict is NOT contradicted**
by anything in the data (§5.1). But three of its descriptive characterisations are:

1. **"Divergence begins around eps-anneal completion (~120k)" — CONTRADICTED.**
   **24 of 27** GNN-arm grad-norm blowups begin *before* 120k; median onset is
   42–65% of budget (62k–98k steps), while epsilon is still ≈0.3–0.5. The decline is
   not triggered by exploration collapse; it precedes it substantially.
2. **"`mlp` stable" — CONTRADICTED (partially).** 2/12 `mlp` seeds show the same
   grad-norm blowup signature (seeds 8, 11), and `mlp` seed 7 loses 37.3 points from
   peak without crossing the grad-norm threshold. The failure mode is **not
   GNN-exclusive** — it is rarer, later and milder in `mlp`.
3. **"All GCN topologies equally affected" — CONTRADICTED.** Severity is strongly
   and monotonically ordered: `gnn_complete` (degree 5) 10/12 seeds, onset 42%,
   max loss 3.4e6 ≫ `gnn_wrong` (degree 1) 9/12, 56%, 1.2e4 ≳ `gnn_true` (degree 1)
   8/12, 65%, 5.6e3. Severity tracks **aggregation density**, not topology
   correctness alone.

### 4.3 E1 `mlp`'s own spread (−122.8 … −61.2) — characterised

`mlp` is bimodal, and the mechanism is the same instability:

| seeds | final | peak→final drop | grad-norm blowup | character |
|---|---|---|---|---|
| 0,1,2,3,4,5,6,9,10 (9 seeds) | −61.2 … −71.5 | 3.4 – 10.7 | none | **clean, converged** |
| 7 | −99.44 | 37.3 | sub-threshold | partial late collapse |
| 11 | −83.27 | 23.4 | step 143 525 | late collapse |
| 8 | **−122.77** | **59.4** | step 137 975 | full late collapse |

`mlp`'s clean subgroup plateaus at ≈ −63, which is what its **median (−63.58)**
reports. Its **mean (−73.42)** is a 3-seed tail artifact. This is why the pre-reg
mandates medians alongside means — and note the direction: it makes the reported
`gnn_true` deficit **conservative** (§1.2).

### 4.4 The instability is present in E0 too (comm ON, N=2)

| arm | seeds with blowup | peak | final | drop |
|---|--:|---|---|---|
| `gnn` | **6/6** | −22.9 … −26.2 | −36.1 … −53.1 | 12.1 – 28.9 |
| `mlp` | **3/6** | −17.5 … −22.5 | −20.1 … −53.1 | 2.6 – 30.6 |

E0 `gnn` max loss reaches 2.85e5. The instability is therefore **not** specific to
comm-off, to N=6, or to a routing requirement — it appears wherever this stack meets
this MPE family. It does **not** appear in `oracle` (0/12) nor, per the diagnostic,
on CoordGrid at identical hyperparameters.

### 4.5 Mechanism — LABELLED HYPOTHESIS, not a finding

The residual + LayerNorm stabilised-GCN path is a **hypothesised** contributor. It
is consistent with the density ordering (§4.2 item 3) and with `mlp` — which lacks
residual/LayerNorm — being least affected. It is **not established**: the read-only
diagnostic could not ablate it, `mlp` and E0 show the failure without the GCN path
being the sole candidate, and no controlled ablation has been run. **Do not report
any mechanism as a finding.** A resolving experiment would be an authorised
ablation (residual/LayerNorm on/off × gradient clipping × lr) — a new decision, not
implied by any pre-registration.

---

## 5. Anomaly audit

### 5.1 Does anything contradict the diagnostic's no-defect verdict? — NO

Checked and clean: no `NaN`/`Inf` in any return series (60 + 12 + 24 runs); the
metric reproduces to 1.4e-14 from raw episodes; `config.json` audit shows one
trainer signature and byte-identical GNN `algo_kwargs` differing only in `graph`
(§0.3); param audit pre-registered and matched (38534 ×3 in E1; 23942 ×2 in cell R);
oracle observation plumbing verified at runtime in pre-reg A.7; gate 1 re-verified
at 150k (§0.4); loss/grad-norm drift is large but finite with no numerical failure.

**Provenance, stated precisely.** The no-implementation-defect verdict is taken
**as reported** by the read-only diagnostic (relayed via the task brief and the
gate-verdict amendment); no diagnostic artifact was located in the repository and
this analysis does **not** re-derive that verdict from source. What is
independently verified here is the list above: absence of numerical failure,
absence of parameter mismatch, absence of metric error, and gate-1/gate-2
re-verification at the confirmatory budget.

**The §4.2 contradictions concern the diagnostic's *descriptive characterisation*,
not its defect verdict.** The pre-reg's void rule (gate-verdict note, cf.
`phaseC_classical` rule 4) is triggered only by a **confirmed implementation
defect**. None is confirmed. **Therefore no arm is voided, and the confirmatory
result stands and is interpreted under the frozen branches — exactly as the
gate-verdict amendment requires.**

### 5.2 Consistency with the pre-registered calibration note — CONFIRMED

The gate verdict recorded `gnn_true` diverging in the 150k re-smoke (−122.74 mean,
seed spread −85.8 … −148.3, below the random floor). E1's confirmatory `gnn_true`
(−109.38, range −148.30 … −73.61, 11/12 below floor) **reproduces that calibration
note at n=12**. The confirmatory outcome was correctly anticipated and disclosed in
writing *before* the run. No surprise, no post-hoc reinterpretation.

### 5.3 Out-of-family nominal significance on the relay — flagged, NOT claimable

`gnn_complete` − `mlp` on `relay_routing` = **+2.87**, d = +1.06, raw Welch
p = 0.0246, MWU p = 0.0061 — *both nominally significant uncorrected*. This is
**outside** the frozen `gnn_true`-centred family, carries no Holm correction, and is
a **mean artifact**: against `mlp`'s median (63.33), `gnn_complete`'s 63.98 is level.
(`gnn_wrong` − `mlp` = +2.16, Welch p = 0.076, MWU p = 0.751 — not even nominally
significant, and the MWU/Welch disagreement is itself the signature of a
tail-driven mean.) **Must not be claimed.** Reported here only so it cannot be
discovered later and mistaken for a suppressed positive.

### 5.4 Other material observations

- E1's non-significant within-GNN orderings differ between means and medians in
  places; none is significant and none may be read as a structure effect.
- E0's `mlp` reaching −20 in its 3 clean seeds confirms the harness *can* train
  unmodified `simple_reference`; the failure in §2 is a stability failure, not an
  inability to learn.
- No seed was dropped, re-run, or excluded anywhere in this analysis.

---

## 6. Claimable statements (verbatim-ready for the manuscript)

Each statement below is supported by the frozen protocol applied to the committed
data. Numbers are exact.

**C1 — E1, the Branch-3 result.**
> On a composed third-party MPE vehicle (`simple_reference` pairs, $K{=}3$, $N{=}6$,
> native communication disabled; 12 seeds, 150k steps, pre-registered before the
> run), the task-matched coordination graph **failed to transfer**. The oracle gate
> passed --- an arm handed the routed information directly reached $-43.36$
> $[-45.90, -40.82]$ versus the no-graph control at $-73.42$ $[-85.67, -61.16]$
> (advantage $+30.06$, $d{=}2.16$, Welch $p{=}1.9{\times}10^{-4}$, MWU
> $p{=}3.7{\times}10^{-5}$), so the task is learnable when routed and the metric has
> headroom. Yet `gnn_true` ($-109.38$ $[-123.41, -95.35]$, median $-108.73$) did not
> exceed the no-graph control: it was significantly **below** it (advantage
> $-35.96$, $d{=}-1.73$, Welch $p_{\text{Holm}}{=}1.0{\times}10^{-3}$, MWU
> $p{=}7.3{\times}10^{-4}$), and did not separate from either the density-matched
> wrong graph ($+2.78$, $d{=}0.14$, $p_{\text{Holm}}{=}0.74$, MWU $p{=}0.62$) or
> all-to-all ($+14.61$, $d{=}0.60$, $p_{\text{Holm}}{=}0.31$, MWU $p{=}0.13$).
> This is the pre-registered Branch 3, published as pre-committed.

**C2 — E1, the instability that accompanies it (must travel with C1).**
> The failure is not a converged null. All three graph arms diverged during
> training: 8/12 (`gnn_true`), 9/12 (`gnn_wrong`) and 10/12 (`gnn_complete`) seeds
> show a sustained gradient-norm blow-up, 24 of those 27 beginning **before**
> exploration annealing completes (median onset 42--65\% of the budget), and 10--11
> of 12 seeds in each graph arm finish **below the random-policy floor**
> ($-84.81$). The no-graph control is affected more mildly and later (2/12 seeds,
> onset $\geq$ 92\% of budget), and the oracle arm not at all (0/12). We report the
> failure to exceed the no-graph control as the pre-registered outcome and the
> divergence as a characterised but **unexplained** training instability of this
> stack on this environment family.

**C3 — E1, the divergence does not explain away the result (label as exploratory).**
> The conclusion is robust to the instability. In a post-hoc, exploratory check we
> re-read every run at its **best** 100-episode window rather than the
> pre-registered final window --- a maximally generous readout for a diverging arm.
> `gnn_true` still falls significantly short of the no-graph control ($-8.39$,
> $d{=}-3.29$, Welch $p{=}1.7{\times}10^{-7}$, MWU $p{=}3.7{\times}10^{-5}$). The
> graph arm never exceeded the no-graph control at any point in training.

**C4 — E1, the rescoping sentence (Branch 3 frozen consequence).**
> We therefore do **not** claim external validity for the structure effect on this
> vehicle. The positive endpoint stands on the constructed archetypes on which it
> was measured --- \coordgrid, \tokenmatch, and the deep-coordination-graph
> comparison --- and we rescope the external-validity claim to those. Whether the
> effect survives on a third-party benchmark remains open; our one pre-registered
> attempt did not settle it in favour of the graph.

**C5 — Cell R, the within-GNN null (replaces the missing-arms limitation).**
> Completing the 2-hop relay cell with the two missing within-GNN controls (12
> seeds, 150k steps, byte-identical parameters, pre-registered as an addendum) the
> three graph arms are statistically indistinguishable from one another ---
> `gnn_true` $64.00$ $[63.26, 64.75]$ (median $63.84$), `gnn_complete` $63.98$
> $[63.70, 64.27]$ (median $63.94$), `gnn_wrong` $63.28$ $[62.93, 63.63]$ (median
> $63.15$); `gnn_true` $-$ `gnn_wrong` $+0.73$ ($p_{\text{Holm}}{=}0.16$, MWU
> $p{=}0.069$), `gnn_true` $-$ `gnn_complete` $+0.02$ ($p_{\text{Holm}}{=}0.95$,
> MWU $p{=}0.62$) --- and none clears the no-comms \mlpqmix\ control under both
> pre-registered tests (`gnn_true` $-$ \mlpqmix\ $+2.89$,
> $p_{\text{Holm}}{=}0.10$, MWU $p{=}0.078$; against the \mlpqmix\ median $63.33$
> the graph arms are level). On the relay, graph **structure does not matter among
> the graph arms, because none of them routes** --- a within-GNN null. This does
> not disturb the committed cell-C result: the \mlpqmix\ comparison fails the
> non-parametric test uncorrected, so the conclusion is invariant to the family
> having grown from six contrasts to eight, and all four DCG contrasts remain
> significant in both families.

**C6 — E0, what it can honestly say (NOT the frozen anchor sentence).**
> On the unmodified $N{=}2$ task with native communication left on (6 seeds,
> non-confirmatory), the graph and no-graph arms do not differ significantly
> ($-12.62$, $d{=}-1.05$, Welch $p{=}0.11$, MWU $p{=}0.24$) --- but this cell fails
> its own validity gate and we do not rest the redundancy argument on it: both arms
> reach above-floor performance mid-run and then lose it (graph arm 6/6 seeds,
> no-graph arm 3/6), so the final-window comparison reads a post-divergence state
> for most seeds, with both arm means at or below the random-policy floor
> ($-28.25$). We report it as a second instance of the same instability rather than
> as evidence that the graph is redundant.

### 6.1 Exact limitation-clause edits the paper should carry

**(a) Replace the cell-C missing-arms clause** in `paper/main.tex` Limitations
(currently: *"it carries no `gnn_wrong`/`gnn_complete` arms, so it speaks only to
GNN-vs-DCG and GNN-vs-MLP and cannot test within-GNN structure-dependence on the
relay"*) — and the matching sentence at the end of the Cell C paragraph
(*"This cell carries no `gnn_wrong`/`gnn_complete` arms, so it speaks only to
GNN-vs-DCG and GNN-vs-MLP, not to within-GNN structure"*) — with:

> the within-GNN controls have since been added (12 seeds, 150k, byte-identical
> parameters): `gnn_true`, `gnn_wrong` and `gnn_complete` are mutually
> indistinguishable and none clears the no-comms \mlpqmix\ control, a within-GNN
> null consistent with none of them routing the relay;

The two remaining sub-limitations (single 150k budget; one constructed relay task)
are **unchanged and must be retained**.

**(b) Replace the external-validity "natural next step" promise.** Four sites carry
it: `main.tex` abstract (~L367), ~L2067, ~L2306, ~L2392. Branch 3 requires the
promise be replaced by the report of the attempt plus the rescoping. Suggested
replacement, adapt per site:

> we attempted this replication on a composed third-party MPE vehicle with a native
> private-partner-goal structure and pre-registered decision rules; the graph arm
> failed to exceed the no-graph control there, amid a characterised training
> instability, so we rescope the claim to the constructed archetypes measured here
> and do not generalise beyond ``task-structured routing under partial
> information.''

---

## 7. NOT claimable

1. **The E0 frozen honesty-anchor sentence** ("the graph is redundant when the env
   supplies its own channel — the topology question only acquires content at
   $N\geq4$ with the channel controlled"). Its precondition (pre-reg section 9
   bullet 2, non-degenerate learning for both arms) **fails**: both arms end at or
   below the random floor after diverging. A tie between two failed runs is not
   redundancy.
2. **"Graph structure does not help / is harmful on MPE."** The graph arms
   *diverged* rather than converged. The supported statement is the narrow one:
   they **do not exceed the no-graph control on this vehicle at this budget**.
3. **"Replicates on an unmodified benchmark" / "on coordination structure we did
   not design."** Permanently barred by pre-reg section 6 regardless of outcome
   (composition + pairing graph + comm-disable are author-imposed).
4. The **Branch 1 / 2a / 2b claimable sentence** of pre-reg section 6 (task-matched
   graph beats wrong/all-to-all/no-graph on third-party physics). Those branches do
   not obtain; the sentence is unavailable.
5. **`gnn_complete` − `mlp` on the relay** (+2.87, raw Welch 0.0246, MWU 0.0061).
   Outside the frozen family, uncorrected, and a mean artifact against the `mlp`
   median. Not a finding.
6. **The density-severity ordering of the instability** (`complete` > `wrong` >
   `true`) as any mechanism claim. Exploratory, unablated.
7. **Residual + LayerNorm as the cause** of the instability. Labelled hypothesis
   only — the read-only diagnostic could not ablate it.
8. **Any E2 (K=5) escalation.** Its frozen trigger is Branch 2a, which did not
   obtain. E2 remains un-launched.
9. **Cell-R pattern 3 (drift anomaly).** Does not obtain; no re-examination of the
   committed cell-C `gnn_true` is triggered.
10. **Any narrowing of `gnn_true` − `gnn_complete` in E1 to a "direction".** d=0.60
    at n=12 is underpowered, not a trend; report as null/underpowered, not as
    "gnn_true was better."

---

## 8. Honesty check

- **Protocol matched?** Yes. One trainer signature across all 96 runs analysed;
  GNN arms byte-identical but for `graph`; seed counts matched within every cell;
  the smaller E0 n is pre-registered and E0 is non-confirmatory.
- **Family correct?** Yes. E1 = exactly the three frozen `gnn_true`-centred
  contrasts (m=3), oracle excluded as a pre-registered anchor/gate. Cell R = the
  existing cell-C `gnn_true`-vs-every-arm family grown to m=8 per the ADDENDUM,
  with a reported m=6 invariance check.
- **Claim no stronger than the evidence?** Yes. Branch 3 governs by entailment; the
  excess over its wording is reported as statistics plus a characterised
  instability, not as a new claim. The E0 anchor sentence is withdrawn. Every
  out-of-family nominal significance is disclosed and marked unclaimable.
- **Garden of forking paths?** Controlled: pre-registration and budget freeze both
  precede all data in git ancestry; the confirmatory outcome was anticipated in
  writing (the divergence calibration note) before the run; no seed excluded, no
  metric substituted (the peak-window analysis is labelled exploratory and reported
  because it argues *against* an easy excuse, not for a claim).
- **Power.** E1 n=12 is adequate for the effects observed (|d| = 1.73 and 2.16 on
  the decisive contrasts). `gnn_true`−`gnn_complete` (d=0.60) is **underpowered,
  not null-with-confidence**; raising seeds could resolve it, but would not change
  the branch, since the branch turns on the `mlp` contrast. E0 (n=6, d=−1.05) is
  underpowered by construction and non-confirmatory.
