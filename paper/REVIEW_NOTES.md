# Pre-submission review notes

## Acceptance blockers (must fix)

- **IQL and VDN are literally the same algorithm in this codebase.**
  `src/gnnmarl/algos/iql.py:45-46` and `src/gnnmarl/algos/vdn.py:39-40` both
  implement `_total_q` as `chosen.sum(dim=-1)`, both use the same
  `AgentQNet` encoder, the same `BaseAlgo._td_step` (single Huber on the
  *summed* team Q with Double-DQN targets), and the same Adam optimizer.
  The IQL docstring (`iql.py:8-15`) admits "numerically equivalent" to a
  per-agent formulation, but the paper (`main.tex:182-184` and 295) sells
  IQL as the *no-coordination* baseline ("ignores agent interaction and
  trains independent Q-learners"). A reviewer who reads the code will see
  two algorithms with the same code path and conclude the paper either
  mis-describes IQL or duplicates a baseline to inflate the comparison
  table. The existing pilot CSV (`results/phase1/summary_aggregate.csv`)
  is already inconsistent (IQL 65.04 vs VDN 32.07 from 1 seed), but that's
  variance, not a true algorithmic difference. **Fix:** make IQL a true
  per-agent TD (one Huber per agent, no team sum in the loss) by overriding
  `_td_step`, *or* drop the IQL label entirely and rename to "VDN" with a
  note that classical IQL collapses to this on a fully shared reward.

- **GNN-QMIX is not capacity-controlled against QMIX, contrary to the
  paper's claim.** The "Why this matters" paragraph (`main.tex:306-307`)
  asserts the per-agent encoder is "byte-identical" between QMIX and
  GNN-QMIX, so any gap is attributable to the GCN. The code disagrees:
  `gnn_qmix.py:54-61` defines a *new* encoder
  (`Linear(obs+id, 64) -> ReLU -> Linear(64, 64) -> ReLU`) followed by
  `GCNStack(64, 64, L)` and a fresh `Linear(64, n_actions)`. Each GCN layer
  adds 64×64 + 64 ≈ 4.2k parameters; L=2 adds ~8.4k that QMIX does not have.
  At depth L=4 (Phase 4) the gap is ~17k extra params. This is the *exact*
  confound the paper says it controls for (Sec.~1 contribution gap 2,
  `main.tex:113-116`). **Fix:** either (i) widen the QMIX agent net so the
  parameter count matches GNN-QMIX at each L tested, *or* run an MLP-depth
  control that replaces the GCN with an identical-shape MLP stack of the
  same width, and report it as the "MLP-depth" ablation alongside H3. The
  latter is much more honest because that ablation isolates "graph
  structure" from "extra depth."

- **The 80%-of-best threshold leaks across the algorithms in the
  comparison.** `src/gnnmarl/utils/stats.py:60-75`
  (`threshold_from_best`) sets the bar to 0.8 × max(final-window mean) over
  *the algorithms present*. The paper (`main.tex:351-355`) describes it
  identically. This means adding or removing an algorithm changes every
  other algorithm's "episodes-to-80%" number. Worse, an algorithm that
  reaches a substantially higher ceiling forces all weaker algorithms to
  appear to take longer (or never reach threshold), even on conditions
  where they have always been competitive. The metric is therefore not a
  property of an algorithm-condition pair but of the panel. **Fix:** define
  the threshold as a *fixed*, condition-independent fraction of an absolute
  performance ceiling (e.g. the theoretical max edge-count reward per
  episode, easily computable for CoordGrid: `|E| * episode_steps` if every
  step is co-located), or per-algorithm self-thresholding ("episodes to
  reach 80% of *that algorithm's own* final-window mean") and report both.
  As-is, "GNN-QMIX is more sample-efficient" claims read off the forest
  plot are not interpretable.

- **H3 has no falsifiable success criterion in the paper text.** Sec. 4
  states the qualitative claim (`main.tex:242-247`): peak near $L \approx d$,
  degradation away from it. The H3 test in
  `scripts/phase4_analysis.py:108-122` operationalizes "match" as
  `|argmax_L - d| ≤ 1` and counts per-diameter passes. Neither the
  threshold of 1 nor the global pass criterion (4/4? 3/4?) appears in the
  paper. With only 3 seeds × 4 depths the per-diameter argmax is extremely
  noisy — flat curves will look like H3 confirmation by accident.
  **Fix:** state the rule in the paper ("we declare H3 confirmed if the
  group-level mean-curve peaks within ±1 of $d$ on at least 3 of 4
  diameters AND the L=$d$ vs L=1 / L=$L_{\max}$ contrasts are Holm-
  significant"), report a per-curve flatness statistic (e.g. relative range
  of the L-curve), and pre-commit a non-flatness floor before declaring
  inverted-U.

## Major issues (should fix)

- **The depth × diameter design confounds N with d.** `main.tex:277-279`
  and `scripts/phase4_ablation.py:37-42` realize d=4 by using a *line on
  N=5* — i.e. the team size changes between conditions (N=4 for d∈{1,2,3},
  N=5 for d=4). Observed degradation at d=4 could be a team-size effect,
  not a depth-vs-diameter effect. **Fix:** either (a) hold N=6 fixed across
  all four diameters by picking `ring(N=6, d=3)`, `line(N=5, d=4)`, etc.,
  with care, or (b) explicitly report and statistically test a separate
  N-vs-return slope so the d=4 effect can be decomposed. Mention the
  confound in the paper.

- **H2 is stated qualitatively, not quantitatively.** `main.tex:238-241`
  says "The advantage of GNN-QMIX over QMIX is larger on a structured
  graph (ring) than on Erdős–Rényi" but the analysis script
  (`phase2_analysis.py:173-185`) computes only the per-condition gap and a
  two-sided Welch t — it does not test the **difference of differences**,
  which is the actual H2 statement. **Fix:** add an interaction-term test
  (e.g. a 2-way ANOVA on `final_mean ~ algo * graph_regime` restricted to
  {qmix, gnn_qmix}, or a contrast on `(gap_ring − gap_er)`) and pre-register
  its α.

- **3 seeds is below what the cited evaluation-methodology papers
  recommend.** `main.tex:210-216` cites Henderson 2018 and Agarwal 2021,
  both of which recommend ≥5 seeds (often more, plus CIs that are
  bootstrap-stratified). The paper runs 3 seeds throughout and uses
  Student-t CIs (`utils/stats.py:172-193`) which are extremely wide at n=3
  (t_{0.975, df=2} ≈ 4.30, so the CI is ±2.48 standard errors). Holm-
  corrected Welch t-tests on n=3 vs n=3 have approximately *zero* power for
  realistic effect sizes. **Fix:** either bump to 5 seeds (Phase 2 at 30k
  steps × 5 seeds × 16 cells = doable on CPU in a day), or replace the t-CI
  with a stratified bootstrap percentile interval and acknowledge the
  power limit explicitly. State realised power for the H1/H2 contrasts.

- **The pairwise family is not defined.** `main.tex:359-363` says "Holm-
  Bonferroni step-down correction across the pairwise family" but does not
  say what the family is. `phase2_analysis.py:161-171` corrects only within
  a single (N, graph) condition's 6 pairwise tests, *not* across
  (N, graph) cells. This is defensible but undeclared — and the H1/H2 file
  (`h1_h2_summary.csv`) reports a raw uncorrected p-value with no Holm
  protection at all (`phase2_analysis.py:179-185`). **Fix:** state explicitly
  "Holm is applied per condition over the 6 algorithm pairs; H1/H2 contrasts
  are corrected separately over the (N × graph)=4-cell family."

- **GNN residual ordering is non-standard and undocumented.**
  `networks.py:153-160` computes `new = ReLU(Linear(A_hat @ h))` then adds
  the *pre-layer* `h` after the ReLU, only from layer 2 onward. This is
  neither pre-activation nor post-activation residual; the paper does not
  describe it. The implication for the L=1 vs L≥2 ablation is that L=1 has
  **no residual at all** and depths ≥2 have residual: any apparent benefit
  of L=2 vs L=1 is partly the residual, not the depth. **Fix:** either
  remove the residual (cleanest for the depth ablation), make it consistent
  across all layers (`new = ReLU(Linear(A_hat @ h)) + h` when dims match,
  including the first layer when in=out, OR with a learned skip
  projection), or document the choice in the paper and re-run the depth
  ablation with and without residual to attribute the depth effect.

- **The threshold-from-best is computed differently in Phase 1 vs Phase
  2.** `phase1_analysis.py:97-100` passes per-algo *concatenated* arrays
  (which makes "best final-window mean" the mean over the last 10% of
  every concatenated run); `phase2_analysis.py:104-109` does the same but
  per (N, graph) condition. Meanwhile the *paper definition*
  (`main.tex:351-355`) says "the best final-window return across algorithms
  in the condition" — singular. Concatenation across seeds before taking
  the tail is not equivalent to "best mean across runs." **Fix:** compute
  threshold = 0.8 × max_algo mean_seed(final_window(run)) and document it
  consistently.

- **Discount factor sensitivity not discussed.** γ=0.99 is fixed across all
  experiments (`base.py:111`, `main.tex:289`), but CoordGrid episodes are 25
  steps (`scripts/phase1_pilot.py:38`), so the effective horizon (~100
  steps) far exceeds the episode length. γ=0.95 or even 0.9 would be a
  natural alternative and the paper does not say it considered this. A
  hostile reviewer will ask whether QMIX's relative performance changes
  under shorter discount. **Fix:** add a one-line discount sweep
  (γ ∈ {0.9, 0.95, 0.99}) at the pilot scale as a robustness check in the
  appendix.

- **MPE replication (Phase 3) is essentially absent.** `main.tex:393-401`
  promises a PettingZoo adapter and "restricted replication or deferred to
  future work." The pre-registered hypotheses H1–H3 reference CoordGrid
  only, but the *external-validity* claim in the conclusion
  (`main.tex:482-497`, "graph structure pays for itself only under specific
  conditions") will be read as a general MARL claim. **Fix:** either (a)
  drop Phase 3 prose and explicitly scope all claims to CoordGrid, or
  (b) run even a 3-seed simple_spread comparison at 200k steps; without
  *something* outside CoordGrid the headline generality won't survive
  review.

## Minor issues / polish

- **Abstract sentence 4** (`main.tex:78-79`) is currently a placeholder
  inside flowing prose: "We find that [PLACEHOLDER FOR RESULTS …]" — this
  must be replaced before submission, and the abstract should distinguish
  H1/H2/H3 outcomes (the current single sentence conflates them).

- **Algorithm 1 box** (`main.tex:309-339`) shows `f_mix(q_i, a; s)` taking
  actions as an argument, but `QMixerHypernet.forward` (`networks.py:221`)
  takes only `(q_per_agent, state)` — the action gather happens *before*
  the mixer is called (`gnn_qmix.py:121`). The pseudocode hides this and
  also omits the adjacency argument used by GNN-QMIX. **Fix:** align the
  pseudocode shapes to the code.

- **The bibliography is missing** several papers a reviewer will expect:
  *Wang et al. 2020 QPLEX* (the obvious extension of QMIX beyond QTRAN);
  *Foerster et al. 2017 RIAL/DIAL* (predecessors to coordination-graph
  ideas); *Yu et al. 2022 MAPPO* (current strong baseline that everyone
  asks about); *Sukhbaatar et al. 2016 CommNet* and *Hoshen 2017 VAIN*
  (early MARL-with-communication that the related-work paragraph treats as
  if graph-MARL is a fresh idea). DGN's claim of being applied "in a
  continuous setting" (`main.tex:192-193`) is a mis-summary — DGN is
  evaluated on discrete cooperative tasks too. **Fix:** add 3-5 of the
  above and tighten the DGN sentence.

- **The episode boundary in the replay treats next_state under done
  correctly** (`base.py:337`: `(1 - done) * next_q_tot`) but
  CoordGrid never sets done before the fixed `episode_steps`
  (`coord_grid.py:164`). That makes every episode a fixed-length truncation
  rather than a Markovian termination — i.e. the implicit "value of being
  in the terminal state" is biased to 0 even though the rollout was cut
  artificially. **Fix:** either set `done=False` and `next_state` to the
  true next reset state's value bootstrapped from the target net (proper
  truncation handling), or document the fixed-horizon assumption in the
  paper's Background section.

- **Prose padding to trim.** Sec. 1 ("The pitch for graph-based MARL is
  intuitive ... The pitch is also commonly believed", `main.tex:100-105`):
  drop the "pitch / pitch" double — say once. Sec. 6 ("Where graph
  structure helps", `main.tex:417-426`) and Sec. 8 ("The headline finding",
  `main.tex:490-495`) repeat the same sentence stem three times. The
  Limitations item 1 (`main.tex:464-467`) is hand-wavy ("tabular-flavoured
  gridworld") — either back up "tabular-flavoured" or drop the qualifier.

- **The `name = "gnn_qmix"` literal** appears in two places and the env
  factory string is `"coord_grid"` — paper uses `\gnnqmix` and `\coordgrid`
  consistently, but the appendix repro reference (`main.tex:519-521`)
  should literally name the script entrypoints (`scripts/phase2_e1_sweep.py`
  etc.) so reviewers can match prose to code.

- **`AgentQNet` carries a `_agent_eye` buffer that is also duplicated in
  `_GNNAgentNet`** (`networks.py:60-65`, `gnn_qmix.py:63-67`). Not a
  correctness bug but worth deduplicating before the supplementary code is
  released.

## Things the paper does well (preserve)

- The pre-registered three-hypothesis structure (`main.tex:226-247`) and the
  Limitations section (`main.tex:458-479`) are unusually honest for a
  workshop-flavoured submission; keep both.

- Tying CoordGrid's reward to the graph edges *directly*
  (`main.tex:262-269`, `coord_grid.py:269-275`) is the cleanest possible
  operationalisation of "graph = coordination structure" and the right
  call for this study — leverage it more prominently in the abstract.

- The compute-scaling table (`main.tex:526-545`) and the docs/PHASES.md
  trail show a pre-registered budget that was scaled down for compute and
  *reported as such*, not silently. Reviewers will not penalise this if you
  state expected statistical power under the reduced budget.

- The single shared training loop (`base.py:299-346`) is genuinely a clean
  ablation harness; the IGM/monotonicity QMIX mixer
  (`networks.py:221-255`) matches the Rashid et al. reference
  implementation closely.
