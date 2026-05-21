# Final pre-submission review

Scope: second-pass methodology and writing review against the now-populated
draft (`paper/main.tex`, `paper/results/phase{1,2,4}.tex`,
`results/phase{1,2,4}/*`). The big structural issues from the first pass
(IQL/VDN identity, GCN residual confound, Holm family, threshold leakage)
have been fixed at the code level. The remaining issues are mostly about
calibration of claims to what 3-seed CPU data can actually support, plus a
handful of inconsistencies between the design section and the executed
runs.

## Acceptance blockers (must fix)

- **The Phase 3 (MPE) section is still a literal placeholder.**
  `main.tex:525-527` reads "`[PLACEHOLDER for either: results of restricted
  runs, OR a statement that this is deferred to future work, ...]`". This
  cannot ship. Pick one: delete the subsection and re-scope all claims to
  CoordGrid, or replace it with a single sentence ("Phase 3 is deferred;
  the PettingZoo adapter is included in the released code at
  `src/gnnmarl/envs/mpe_adapter.py`"). Whatever you choose, propagate the
  scope to the abstract and conclusion — at present the
  Introduction/Discussion frame the negative finding as a finding about
  graph-based MARL in general, while every empirical claim lives on
  CoordGrid.

- **Phase 4 design section claims a $d=4$ diameter that was never run.**
  `main.tex:312-317` advertises "diameter range $\{1,2,3,4\}$ ... combining
  complete ($d=1$), ring on $N=4$ ($d=2$), line on $N=4$ ($d=3$), and
  line on $N=5$ ($d=4$)." The actual sweep (`scripts/phase4_ablation.py:40-44`
  and `results/phase4/h3_test.csv`) tests only $d \in \{1,2,3\}$ — see
  `results/phase4/summary_grid.csv` and the heatmap, which shows three rows
  all at $N=4$. Section~\ref{sec:results:ablation} (`main.tex:532-534`) is
  internally consistent ($d \in \{1,2,3\}$, $N=4$), so the design section is
  wrong, not the results. **Fix:** drop the $d=4$ / line-on-$N=5$
  description from `main.tex:312-317`. While there, drop "$N$ is held
  constant across diameters so the variable under study is unconfounded"
  (`main.tex:537-538`) — that is now correct but the prior sentence
  contradicted it.

- **Phase 4 table renders LaTeX-escape errors.** `paper/results/phase4.tex:17-19`
  emits literal `$\\times$` and `\\checkmark` (double-backslash) in the
  table body. Compiled output will show `\times` / `\checkmark` rather than
  the symbols, or fail outright on some toolchains. Fix the generator
  (likely a `f"{...}"` with `\\` that should be a single `\`) and
  regenerate.

- **Phase 4 row $d=1$ is mis-classified as "non-flat" in the prose.**
  `results/phase4/h3_test.csv` has `relative_range = 0.09769`, i.e.\
  strictly *below* the pre-registered 10% floor, so the pre-registered
  non-flat criterion fails. The .tex table at `phase4.tex:17` rounds it to
  `0.10`, which a reader will reasonably interpret as exactly meeting the
  threshold. `main.tex:559-561` then describes $d=1$ as "essentially flat
  ... (relative range $10\%$, at the pre-registered floor)" — but the
  underlying number is 9.77%, which is *below* the floor. Either report
  the unrounded value (`0.098`) in the table and explicitly say "below the
  10% non-flat floor", or restate the rule with a $\le$/$<$ that matches
  the data. The current ambiguity reads like rule-bending.

## Major issues (should fix)

- **"Conclusively falsified" overclaims given $n=3$ seeds.** `main.tex:491`
  says "H1 is conclusively falsified." H1 is the *direction* GNN-QMIX > QMIX,
  and the one-sided $p \in [0.983, 0.999]$ for that direction means we
  *cannot reject* QMIX $\ge$ GNN-QMIX — it does not by itself license a
  positive claim that GNN-QMIX is worse with Holm-corrected power. The
  effect sizes ($-60$ to $-147$) and the consistency across 4 cells *do*
  support a strong negative claim, but the right word is "falsified" (no
  "conclusively") or "rejected across all tested conditions." Same fix at
  `main.tex:78` and `main.tex:711-714`. The data are unambiguous; just
  stop using the booster.

- **The CI on GNN-QMIX $N=8$ ring crosses zero and the abstract still
  reports it as a robust gap.** `phase2.tex:29` reports GNN-QMIX
  $N=8$ ring final mean $21.1$, CI $[-11.5, 53.7]$. The
  $\Delta = -146.6$ headline gap is real, but the CI for the GNN-QMIX
  point estimate alone spans most of the distance from "broken" to "random
  play," which a reviewer will notice. Recommendation: add one sentence in
  Section~\ref{sec:results:scale} (around `main.tex:512-517`) acknowledging
  that the GNN-QMIX point estimate at $N=8$ ring sits on top of a CI that
  includes zero, and that the QMIX-vs-GNN-QMIX gap is the well-identified
  quantity, not the GNN-QMIX absolute return. This is also a fair place to
  flag the compute counter-argument (see next item).

- **Phase 1 GNN-QMIX got 50k steps and reached 58.5; Phase 2 got 30k
  steps and reached 6-23. The paper does not address this.** The most
  natural reviewer counter-argument is "GNN-QMIX is a bigger model and
  you cut its compute by 40% — of course it lost." The numbers support
  this concern: at Phase 1 (50k, ring N=4) GNN-QMIX = 58.5; at Phase 2
  (30k, ring N=4) GNN-QMIX = 23.1, while QMIX *barely changed* (84.3 vs
  83.5). The differential collapse is huge and the most plausible
  alternative explanation is sample efficiency, not "the graph doesn't
  help." The paper acknowledges the step cut at `main.tex:481-486` but
  never connects it to the Phase-1-vs-Phase-2 GNN-QMIX collapse.
  **Fix (no new experiments needed):** add a paragraph at the end of
  Section~\ref{sec:results:scale} or in the Limitations explicitly
  conceding: (i) Phase 1 (50k) shows GNN-QMIX reaches ~58 on ring $N=4$;
  Phase 2 (30k) shows ~23 on the same condition; (ii) the most natural
  alternative explanation is undertraining of the larger model; (iii) the
  fact that QMIX is essentially unchanged across the step cut (84.3 → 83.5)
  argues that the regime is not generically under-budgeted, but the GNN
  variant may need more; (iv) under-training thus cannot be ruled out for
  the negative claim and is listed as the leading alternative explanation
  in Limitations. Without this concession the paper is one chart away from
  being undone by a reviewer who notices the Phase 1 vs Phase 2 delta.

- **"Where graph structure helps in our experiments: nowhere" overreaches
  given a single GNN family.** `main.tex:586` is going to be the most
  quoted sentence in this paper, and Limitations item 3 explicitly says
  "We test a vanilla GCN. Whether attention (GAT) or learned message
  passing (e.g. GIN) changes the depth-diameter trade-off is open." The
  Discussion heading should match: rename to "Where a vanilla GCN over the
  coordination graph helps in our experiments: nowhere", or drop "nowhere"
  and phrase as "we find no condition in which GNN-QMIX (a vanilla GCN
  inserted into QMIX) improves on QMIX." Same fix in the abstract
  (`main.tex:80-83`) — "GNN-QMIX is worse than ... QMIX in every condition"
  becomes "this specific GCN-augmented QMIX variant is worse than ... QMIX
  in every condition." This is not hedging; it is matching the headline to
  the experiment that was actually run.

- **"GNN-QMIX is a fair re-creation of GraphMIX" is not actually claimed
  but a reviewer will ask.** `main.tex:228-231` says GraphMIX "interleaves
  a GNN with a QMIX-style monotonic mixer, closest in spirit to the
  GNN-QMIX baseline we evaluate here." A reader will read this as "we
  reimplemented GraphMIX." We did not — our GNN-QMIX is a plain GCN block
  inserted before the Q head of a vanilla QMIX, not the full GraphMIX
  pipeline (which uses agent-level and team-level GNN heads and an
  attention-based pooling). **Fix:** insert one sentence in
  Section~\ref{sec:algos} (around `main.tex:337-341`) clarifying that
  GNN-QMIX is a controlled GNN-augmented QMIX, not a re-implementation of
  any specific named method, and that the GraphMIX comparison is purely
  spiritual. Otherwise a reviewer will say "your GraphMIX is a strawman"
  and the paper does not have a clean answer.

- **Reward design favours the sum mixer by construction.** The reward is
  $\sum_{(i,j) \in E} \mathbf{1}[p_i = p_j]$ (`main.tex:299-301`). This is
  literally a sum of edge indicators, and QMIX's monotonic mixer can
  approximate it cleanly without any need to message-pass: the per-agent
  Q already captures "am I near my graph neighbours". A reviewer will
  argue the reward design is biased *toward* sum-mixer architectures and
  *against* needing a GCN, so the negative claim is engineering-driven,
  not architecture-driven. The paper currently does not address this.
  **Fix (no new experiments):** add this counter-argument explicitly in
  Limitations (item 2, `main.tex:660-663` already touches it but does not
  state the sum-decomposition argument), say "edge-sum reward is by
  construction monotonically decomposable across agents up to the graph
  structure, so it favours QMIX-shaped mixers; tasks where the team
  reward is a *non-separable* function of joint co-location are out of
  scope and would be the cleanest follow-up." This pre-empts the most
  technically substantive negative review.

## Logical / honesty audit

- `main.tex:78-79` (abstract): "GNN-QMIX is worse than the structurally
  simpler QMIX in *every* condition we tested" — the italics on "every"
  read as triumphant given the CI on $N=8$ ring includes zero. Drop the
  italics; keep the claim.

- `main.tex:148-156` (contributions item 4): claims "the prevailing positive
  picture of graph-based MARL does not survive a controlled comparison."
  This is fine as a paper contribution, but the abstract sentence at
  `main.tex:80-82` reads "one-sided $p \in [0.98, 1.00]$ that GNN-QMIX
  improves on QMIX." That phrasing inverts the test direction in a way
  that is *technically* correct ($H_a$: GNN-QMIX $>$ QMIX) but reads as
  hostile statistics-speak. Recommend rewriting as "one-sided Welch $t$,
  $H_a$: GNN-QMIX $>$ QMIX, $p \ge 0.98$ — i.e.\ no evidence in favour of
  GNN-QMIX."

- `main.tex:586` "Where graph structure helps in our experiments:
  nowhere" — see Major item 4. Universal-quantifier framing on a single
  GNN family is the single sentence most likely to be quoted out of
  context in a hostile review.

- `main.tex:737-741` (conclusion): "the answer in some practically-relevant
  regimes is 'not by default'". This is actually well-calibrated; keep it.

## Robustness audit (strongest counter-arguments)

1. **Compute fairness.** GNN-QMIX has ~8.4k extra parameters at $L=2$
   and was given 30k steps in Phase 2 vs 50k in Phase 1. The Phase-1-to-
   Phase-2 GNN-QMIX collapse (58.5 → 23.1 on the same ring $N=4$
   condition) is the smoking gun for under-training. Must be conceded
   in-text (see Major item 3 above). The QMIX number is rock-stable
   across the step cut, which strengthens the paper's hand somewhat, but
   not enough to ignore the asymmetry.

2. **Hyperparameter unfairness.** $\eta = 5 \times 10^{-4}$ for both
   QMIX and GNN-QMIX; learning rate not retuned for the GCN. A reviewer
   will ask whether GNN-QMIX needs lower LR or longer warmup. The
   paper's "no per-algo tuning" stance is defensible
   (`main.tex:758-760`) — but the Limitations should explicitly say
   "we did not retune learning rate or warmup for GNN-QMIX; the
   identical-recipe choice is by design but may understate GNN-QMIX
   under its best hyperparameters."

3. **Implementation soundness / GraphMIX strawman concern.** See Major
   item 5. Our GNN-QMIX is a clean control on QMIX + GCN, not a
   re-implementation of any prior named method. Make that explicit.

4. **Reward-design bias toward sum mixer.** See Major item 6. The
   edge-sum reward is by construction monotonically decomposable in a
   way that benefits QMIX. State this in Limitations.

5. **Erdős–Rényi resampled per episode is much harder than a fixed
   graph.** `main.tex:513-516` notes the GCN encoder fails on ER. A
   reviewer will ask whether this is the architectural failure or the
   *non-stationarity* of the input. The Discussion does not separate
   these; one sentence acknowledging that ER per-episode resampling
   destroys the stationary graph assumption a GCN normally relies on
   would close the gap.

## Polish (worth doing if time)

- `main.tex:455` references "Phase-0 acceptance criterion" but the paper
  itself never defines Phase 0 — that name only lives in
  `docs/PHASES.md`. Rewrite as "the Phase 1 sanity-check criterion (see
  Appendix~\ref{app:budget} and the harness's `docs/PHASES.md`)" or just
  "the harness sanity-check criterion is met."

- Abstract sentence stem "GNN-QMIX is worse ..." (`main.tex:79`) and
  Conclusion sentence stem "GNN-QMIX does not beat QMIX ..."
  (`main.tex:710-712`) repeat the same claim in three places (abstract,
  Section 6 lead, conclusion). Vary the phrasing.

- `main.tex:557` "$p=$ pass" reads like an unfilled placeholder. Replace
  with the actual flatness statistic ("relative range $0.78$, argmax
  $L^*=2$, both pre-registered criteria met") or remove the parenthetical.

- `main.tex:565-570` (Phase 4 finding 2) reports "Mean returns at
  $L \in \{3,4\}$ are 4.4-8.1" — verify against the heatmap: $L=3$ at
  $d=1$ is 7.5 (which is within 4.4-8.1) but $L=3$ at $d=2$ is 5.1 and
  $L=4$ at $d=2$ is 4.6, so the range 4.4-8.1 is correct. Good.

- `main.tex:312` lists graph families in the design section but the
  Phase 2 sweep only used ring + erdos_renyi and Phase 4 only used
  complete/ring/line. Either tighten the design-section list to what was
  actually swept, or note that the others (grid2d) ship in the harness
  for future use.

- Caption of Fig.~\ref{fig:phase2_forest}: "episodes to 80\% of best
  final-window return per algorithm" still says "of best" in the
  generated .tex (`phase2.tex:4`). Per the previous review's resolution
  this metric is now per-algorithm self-thresholding, not best-across-
  algorithms. Regenerate or hand-edit the caption to "episodes to 80\%
  of each algorithm's own final-window return."

- `phase1.tex:11`: "episodes-to-80\% is reported only for seeds that
  crossed the threshold" — fine, but the column shows three values
  (1372, 1526, 1422, 1553) with no annotation for which / how many
  seeds reached threshold. If all three seeds reached threshold in every
  algorithm, say so. If not, append "($n/3$ seeds)" to each.

- The bibliography has all five papers the first review asked for
  (`qplex`, `mappo`, `commnet`, `rial_dial`, `graphmix`) — good. One
  more worth adding for the depth-cliff / over-squashing narrative is
  Alon \& Yahav 2021 ("On the Bottleneck of Graph Neural Networks and
  its Practical Implications"). The over-depth-cliff finding in Phase 4
  is essentially over-squashing, and citing it would lend the
  practitioner rule external support.

## Things to preserve (good and important)

- The pre-registered H3 falsifiability criterion (`main.tex:280-285`) and
  the honest "1/3 diameters pass" reporting are exactly what
  Agarwal-precipice style review wants. Keep.

- The capacity-overhead concession at `main.tex:345-355` ("recognise
  that a tighter test would fit a same-shape MLP-depth control... we
  leave that ablation to a future revision") is the right tone — it
  pre-empts the major reviewer complaint without conceding the
  headline finding.

- The pre-vs-executed compute table (`tab:budget`, `main.tex:807-831`)
  and the explicit Holm family definition (`main.tex:419-430`) are
  unusually rigorous for this venue; both will read well.

- The CoordGrid edge-sum reward is the cleanest possible
  operationalisation of "graph = coordination" (see Robustness item 4
  for the *flip side* of this design choice); the paper exploits it
  well in `main.tex:298-309`.

- The IQL implementation has been fixed to a real per-agent TD
  (`src/gnnmarl/algos/iql.py:57-85`); the IQL $<$ VDN $<$ QMIX ordering
  in Phase 1 and Phase 2 is now genuinely informative and supports the
  sanity-check framing.

- The GCN no longer has the L=1-vs-L>=2 residual confound the previous
  review flagged (`networks.py:142-164`); the depth ablation is
  therefore a clean depth-only ablation.
