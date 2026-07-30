# Design: classical coordination-graph baseline (pre-registration draft)

Status: DESIGN — not yet approved, not yet pre-registered, nothing has run.
Author: marl-research-scientist agent, 2026-07-30. Addresses the JAAMAS referee's
subsumption attack: "show the deep isolation delivers something payoff-propagation
over the same factored reward does not."

## 0. The reframe that makes this honest before it makes it decisive

The referee is **right about the 1-hop cell** and the design must concede it up front
rather than dodge it. PairRouting's reward is, edge-by-edge over the matching, a genuine
Guestrin factored payoff:

`r = Σ_edge(i,p) [ prox(pos_i, goal_p) + prox(pos_p, goal_i) ]`, a function of the joint
edge observation `(o_i, o_p)` alone.

So a modern coordination-graph learner (DCG; Böhmer, Kurin & Whiteson 2020 — the neural,
function-approximated descendant of Kok & Vlassis max-plus) whose pairwise factor
`Q_ip(o_i, o_p, a_i, a_j)` conditions on the joint edge observation **sees `goal_{p(i)}`
directly on the true edge**, and max-plus over disjoint matching edges is exact. That
factor is *strictly more expressive* than GNN-QMIX's per-agent routed utility under a
monotonic mix, and our own 1-hop positive captured only ~16% of headroom. **The most
likely 1-hop outcome is that steelman DCG reproduces true>wrong≈complete and
matches/beats `gnn_true`.** If it does, the 1-hop positive is subsumed — and any version
where the classical method "fails" 1-hop is a strawman the referee will correctly call
out as observation-starvation.

The differentiation the referee literally demands therefore **cannot** live in any 1-hop
cell. It lives, mechanistically, in a **multi-hop relay**: max-plus propagates
action-*values* through action-*coupling*; a pure relay has no action coupling, so a
1-hop pairwise factorization cannot carry a non-adjacent source observation across the
relay, whereas a 2-layer GNN composes the feature over two hops. That is the payload
cell, and its liveness is a go/no-go smoke.

## 1. Which classical method, exactly

**Primary baseline: neural DCG = deep sparse cooperative Q-learning with max-plus action
selection**, factored over `env.adjacency()`. This is the correct steelman — not tabular
(tabular loses on *generalization* across the 9^4 joint-state space, a different strawman
the referee would dismiss).

Value model:
`Q_tot(o,a) = (1/|V|) Σ_i U_i(o_i,a_i) + (1/|E|) Σ_{(i,j)∈E} Q_ij(o_i,o_j,a_i,a_j)`
- `U_i`: MLP(o_i + agent-id one-hot) → |A|. Shared params, matched to GNN-QMIX's
  per-agent capacity.
- `Q_ij`: MLP([o_i;o_j]) → |A|×|A|, symmetrized
  `Q_ij(a_i,a_j) = ½(f(o_i,o_j)[a_i,a_j] + f(o_j,o_i)[a_j,a_i])`.

Greedy joint action via **max-plus** (T≈6–8 rounds, mean-subtracted messages for
loopy-graph stability):
`m_{i→j}(a_j) = max_{a_i} [ U_i(a_i) + Σ_{k∈N(i)\j} m_{k→i}(a_i) + Q_ij(a_i,a_j) ] − const`
then `a_i* = argmax_{a_i} [ U_i(a_i) + Σ_{k∈N(i)} m_{k→i}(a_i) ]`.

Training: reuse the harness's Double-DQN TD on `Q_tot` with
`y = r + γ Q_tot^target(o', a'*)`, `a'*` from max-plus at `o'`, Huber loss. This reuses
buffer/target/optimizer plumbing but **must override `act()` and the next-action
selection inside `_td_step`** — DCG does not fit `base.py`'s per-agent-argmax IGM
skeleton (that override is the main build cost).

**Two conditioning variants — the pre-registered structure variable (a control pair, not
a menu):**
- **`dcg_jointobs` (steelman, HEADLINE arm):** `Q_ij` conditions on `(o_i,o_j)` —
  canonical DCG, edge-local, *legal* (uses only what a graph edge carries). Can route
  1-hop.
- **`dcg_canonical` (mechanistic control):** `U_i(o_i,a_i)` and `Q_ij(a_i,a_j)` with
  **no cross-agent obs in the pairwise factor** — the classic Kok–Vlassis
  coordination-game factor that captures action interaction, not observation routing.
  Isolates action-coupling from observation-routing.

**Partial-observability handling / "best legal version" (steelman rule):** the edge
factor gets the two endpoints' full ego observations — the maximal information an edge
legally carries at decentralized execution. A full-*global*-state factor is included, if
at all, only as a clearly-labeled **privileged ceiling arm** (`dcg_oracle_state`), never
as the comparison, because it violates decentralized execution that QMIX/GNN-QMIX obey.
Generosity flows toward DCG (the thesis-threatening arm): tune max-plus iterations,
payoff-net width, and lr only to help DCG win; never tune the GNN arms to widen the gap.

## 2. The three cells and their roles

| cell | env / obs | role | prediction |
|---|---|---|---|
| **(A) full-info edge-reward CoordGrid** | default co-location reward, `graph=ring`, `obs_mode=full` | **validity + honest dissociation** | classical **wins/ties** (its home turf: action coordination), GNN-QMIX ≤ MLP (our negative endpoint) |
| **(B) PairRouting-ego** | `pair_routing=True`, `obs_mode=ego`, N=6, 3×3 | **subsumption check** (referee's literal example) | `dcg_jointobs_true` **matches/beats** `gnn_true` → expect to **concede**; `dcg_canonical_true` fails |
| **(C) 2-hop relay (NEW mode)** | disjoint 3-chains s–r–t, `obs_mode=ego` | **the payload** — only cell that meets the referee's bar | `gnn_true` (2 layers) succeeds; `dcg_*_true` **structurally fails** to relay s→t |

**Cell C spec (new env mode `relay_routing` in `coord_grid.py`):** N=6 = two disjoint
chains (s1–r1–t1, s2–r2–t2). Source `s` privately observes a goal `g_s`; **sink `t` is
rewarded for `prox(pos_t, g_s)`**; **relay `r` is rewarded for its own private goal
`g_r`** — this occupies r's action so it cannot double as an action-signaling channel.
True graph = the chains (edges s–r, r–t; s and t exactly 2 hops apart); wrong = a
degree-matched mis-wiring; complete = all-to-all.

Mechanism (state this in the pre-reg): the g_s-dependence enters DCG only through
`μ_{s→r}(a_r)`. For it to reach `a_t` it must change r's argmax `a_r` (so coupling
`Q_rt(a_r,a_t)` transmits it), but r has no task incentive to condition `a_r` on g_s,
and 5 actions cannot encode a 9-valued goal — so g_s enters `μ_{r→t}` only as a flat
offset and dies. A 2-layer GNN composes `o_s → h_r → h_t` regardless of r's actions.
**Head-to-head `gnn_true` vs `dcg_jointobs_true` on the *same* true sparse graph is THE
contrast.** (Honest secondary note: `dcg_complete` may *succeed* via a shortcut s–t
edge — that is exactly the point: payoff-propagation routes only across direct edges it
is given, message passing routes along multi-hop task paths; adding all-to-all is the
dense graph Kok–Vlassis already deprecates.)

## 3. Fairness / "matched" protocol

- **Environment interface:** byte-identical env, `max_neighbors_override=N-1` so obs_dim
  is constant across graphs; only `env.adjacency()` differs across true/wrong/complete
  arms.
- **Seeds:** 12 (0–11), matching the house standard, for every confirmatory cell.
- **Budget:** **150k env steps** (matches the F1 robustness budget; generous to the
  classical arm — pre-empts "baseline under-trained"). "Matched" = identical env-step
  budget; DCG and GNN both learn by gradient TD, so this is a clean common currency.
- **Capacity:** DCG `U_i` matched to GNN-QMIX per-agent net; `Q_ij` payoff net sized so
  total DCG params ≥ GNN-QMIX params (generosity toward DCG). Report param counts.
- **Metric / stats (house style):** final-window mean return (last 20%); two-sided Welch
  t Holm-corrected across the contrast family + Mann–Whitney U; Cohen's d; 95% CIs.
  Defer exact test wiring to the research-statistician.

## 4. Smokes as go/no-go gates (run BEFORE committing 12-seed sweeps)

1. **Smoke C-liveness (run first, cheapest kill-switch):** `gnn_true` 2-layer on relay,
   3 seeds. Is the regime alive (sink t reaches g_s, return clearly above the relay
   floor)? If GNN floors on relay → **NO-GO: no differentiation exists; concede
   complementarity and do NOT run the DCG relay sweep.**
2. **Smoke B-concession:** `dcg_jointobs_true` on PairRouting-ego, 3 seeds — expect
   match/beat `gnn_true` (also validates the DCG implementation on a task it *should*
   solve).
3. **Smoke A-validity:** `dcg_canonical_true` on full-info edge-reward, 3 seeds — must
   clear the floor and coordinate co-location (home-turf sanity).

## 5. Frozen decision rules (4-outcome set)

Let "succeed" = final-window mean significantly above that cell's random floor and,
where compared, contrasts significant after Holm.

1. **DCG-jointobs fails relay (C) while `gnn_true` succeeds, AND classical wins
   full-info (A)** → **DIFFERENTIATION DEMONSTRATED — the headline.** Message passing
   composes observations across multi-hop task paths; pairwise payoff-propagation over
   the same factored reward cannot. Double dissociation: action-coordination under full
   observability vs multi-hop information routing under partial observability. Neither
   subsumes the other.
2. **DCG-jointobs also succeeds on relay (C)** → **SUBSUMED at the mechanism level →
   rescope.** A deep coordination graph reproduces the routing benefit; the paper's
   contribution narrows to the *empirical when*, not a unique mechanism.
3. **On 1-hop (B): `dcg_jointobs_true` matches/beats `gnn_true` (expected) while
   `dcg_canonical_true` fails** → the 1-hop benefit is observation-routing, not
   action-coordination, and is reproducible by a deep CG. **Report as an honest partial
   concession + mechanistic decomposition — not a win.**
4. **Classical fails BOTH A and C** → **VALIDITY GATE FAILS; implementation suspect,
   discard, do not interpret.**

## 6. Validity gates

- DCG wins/ties on full-info edge-reward (A) before its relay result is trusted.
- Unit test: max-plus recovers the exact brute-force argmax on the disjoint
  matching/chain (where it must be exact) and converges (message-norm plateau) on
  complete/ring.
- Reference points reproduce: relay floor (random policy) and a full-info greedy oracle
  measured as in `phaseC_reference_points.py`.
- `gnn_true` at 150k reproduces its known ~66 on PairRouting-ego (harness sanity).

## 7. Implementation sketch, cost, wall-clock

- **New files:** `src/gnnmarl/algos/maxplus.py` (class `DCG(BaseAlgo)`: overrides `act`,
  `_td_step` next-action, adds `U`/`Q_ij` nets + `_max_plus()`;
  `conditioning ∈ {"jointobs","canonical","oracle_state"}` kwarg); register in
  `algos/__init__.py`. New env mode `relay_routing` + a `relay` graph kind in
  `coord_grid.py`. A `phaseC_classical_sweep.py` mirroring `phaseC_structure_sweep.py`
  (reuse existing `phaseC_analysis.py`).
- **Runs (confirmatory):** ~180 new runs at 150k (C: ~8 arms×12; B: ~4 DCG arms×12
  reusing existing GNN 150k; A: ~3×12). DCG is ~2–4× a GNN update.
- **Cost on a 16-core box:** smokes ~15 min total; full confirmatory ≈ half a day
  wall-clock (comfortably overnight, CPU-only). Kill-switch smoke C-liveness first means
  a dead payload costs ~5 min, not half a day.
- Implementation ownership: rl-systems-engineer; test choice: research-statistician.

## 8. Backfire analysis and honest pre-reg wording

- **Backfire 1 — B concedes (likely).** Do not frame B as a win under any wording.
  Pre-reg fixes `dcg_jointobs` as the headline 1-hop arm and pre-commits outcome (3).
- **Backfire 2 — C-liveness fails (GNN can't relay either).** No differentiation to
  claim; pre-commit to concede complementarity and NOT run the DCG relay sweep.
- **Backfire 3 — "you starved the baseline."** Neutralized structurally: `dcg_jointobs`
  gets full joint-edge obs, ≥GNN params, 150k, tuned only in its own favor, and must win
  full-info A. `dcg_canonical` is labeled a mechanistic control, never the comparison.
- **Backfire 4 — `dcg_complete` succeeds on relay via shortcut.** Pre-registered as
  expected and illuminating: payoff-propagation needs a direct edge; message passing
  uses the sparse multi-hop task path. Head-to-head is same-graph `gnn_true` vs
  `dcg_true`.
- **Scope rule (all outcomes):** "up to 150k steps," never "in principle"; "a deep
  coordination graph (DCG)," not "all payoff propagation."
