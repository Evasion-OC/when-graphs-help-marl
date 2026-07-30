# Pre-registration — classical coordination-graph (DCG) baseline

**Written and git-committed BEFORE any smoke or confirmatory result was
observed.** Freezes the design in `docs/CLASSICAL_CG_BASELINE_DESIGN.md`
(read in full for the mechanism argument) plus the implementation
deviations and reference-point calibration discovered while finalising this
pre-reg. Addresses the subsumption attack: "show the deep isolation
delivers something payoff-propagation over the same factored reward does
not."

## 0. What is frozen here

- The three cells (A, B, C), their arms, env configs, and roles (design
  section 2).
- 12 seeds (0–11), 150k confirmatory env-step budget, matching the house
  standard and the F1 150k-robustness check (`results/phaseC_150k/`).
- Metric: final-window (last 20%) mean return per run.
- Contrast family: two-sided Welch's t with Holm correction across the
  contrast family, plus Mann–Whitney U; Cohen's d; 95% CIs. Test wiring via
  `scripts/phaseC_analysis.py` (deferred exact test choice to the
  research-statistician per design section 3, unchanged here).
- The four decision rules, verbatim from design section 5 (reproduced in
  section 4 below).
- The validity gates from design section 6 (reproduced, with one number
  corrected — see section 2c).
- The smoke go/no-go logic from design section 4 (reproduced in section 3,
  with one calibration extension — see section 3's note on cell C budget).

## 1. The three cells (unchanged from design)

| cell | env / obs | role | prediction |
|---|---|---|---|
| **(A) full-info edge-reward ring** | default co-location reward, `graph=ring`, `obs_mode=full`, N=4, grid=5 | validity + honest dissociation | classical wins/ties (home turf), GNN-QMIX ≤ MLP |
| **(B) PairRouting-ego** | `pair_routing=True`, `obs_mode=ego`, N=6, grid=3 | subsumption check (1-hop) | `dcg_jointobs_true` matches/beats `gnn_true` → expect to concede; `dcg_canonical_true` fails |
| **(C) 2-hop relay** | `relay_routing=True`, `obs_mode=ego`, N=6, grid=3 | the payload — only cell that meets the referee's bar | `gnn_true` (2-layer) succeeds; `dcg_*_true` structurally fails to relay s→t |

Arms per cell (implemented in `scripts/phaseC_classical_sweep.py::arms()`):

- **A:** `mlp`, `gnn_true` (existing GNN-QMIX config, repaired
  residual+LayerNorm), `dcg_jointobs`, `dcg_canonical` — all on `graph=ring`
  (A has no true/wrong/complete structure axis; DCG's own conditioning arm
  IS the structure variable here).
- **B:** `dcg_jointobs_true`, `dcg_jointobs_wrong`, `dcg_jointobs_complete`,
  `dcg_canonical_true` — DCG-only manifest, merged post-hoc with historical
  `gnn_true` rows (section 2c).
- **C:** `mlp`, `gnn_true` (2-layer), `dcg_jointobs_true`,
  `dcg_jointobs_wrong`, `dcg_jointobs_complete`, `dcg_canonical_true`, plus
  optional `gat_true` (on by default, `--no-gat` to drop).

## 2. Implementation deviations from the design doc (folded into this pre-reg)

**(a) `relay_wrong` is not fully edge-disjoint from `relay`.** It shares the
source–relay edges of its own chain; only the relay–sink side is rewired to
the *next* chain's sink (see `coord_grid.py::_build_adjacency`, `relay_wrong`
branch). At N=6 (2 chains of 3), a degree-matched arm that is ALSO fully
edge-disjoint from `relay` cannot exist — the only alternative permutation
recreates a matched (source, sink) pair, which is not "wrong" at all. The
load-bearing property is that `relay_wrong` breaks the source→sink route (no
source is ≤2 hops from its own chain's sink), not edge-disjointness; this is
covered by tests (`test_maxplus.py`, `test_coord_grid.py` adjacency checks).
Accepted as designed; noted here for the record, not treated as a defect.

**(b) `oracle_state` is a labeled optional ceiling arm, excluded from the
confirmatory family.** `DCG(conditioning="oracle_state")` is
act()-state-blind / TD-state-aware: the live rollout policy (`DCG.act`)
has no state argument in the `Algo.act(obs, adj, ...)` contract and falls
back to an all-zero state proxy, so it explores/acts as
canonical-DCG-with-QMIX-style utilities while only the TD target sees the
privileged global state. Because its behaviour policy never actually uses
the privileged information, its results would not honestly support the
"even wired for state privilege, DCG's max-plus routing structurally can't
propagate the value" framing without confusing readers about what "oracle"
means. It is **not included in `arms()`** for any cell in this sweep and is
excluded from the confirmatory family and from all four decision rules.
(Design section 1 always described it as "never the comparison" — this
narrows that further to "not run in the confirmatory grid at all.")

**(c) Cell-B merge source is corrected to the 150k `gnn_true` data, not the
30k file the design/handoff named.** Design section 7 and the handoff both
name `results/phaseC/manifest_pair_N6_ego.csv` as "the historical gnn_true
150k rows." **Verified before freezing: that file is actually a 30k run**
(`config.json` → `total_env_steps: 30000`, `eps_anneal_steps: 24000`), with
`gnn_true` mean **66.10** (n=12, sd 4.4) — this is the number the design
doc's validity-gate prose ("`gnn_true` at 150k reproduces its known ~66")
was echoing, but it is a 30k number. A genuine 150k, 12-seed `gnn_true` run
on the byte-identical PairRouting-ego config (same env_kwargs, same
algo_kwargs: `gnn_layers=2`, residual+LayerNorm, `gnn_hidden=64`, `lr=5e-4`,
`batch=32`, `buffer=50k`, `target=200`) already exists from the F1
150k-robustness check: `results/phaseC_150k/manifest_pair_N6_ego_attention.csv`,
`gnn_true` mean **94.96** (n=12, sd 5.14) — `gnn_true` itself climbs
substantially from 30k to 150k (consistent with F1's own finding that more
budget helps the true-graph arm too, not just the attention arms). Using
the 30k row as the 150k comparison would violate the matched-protocol rule
(design section 3 / house style: identical budget across compared cells).
**Frozen correction: cell B's `--merge-gnn-true-from` source is
`results/phaseC_150k/manifest_pair_N6_ego_attention.csv`, not
`results/phaseC/manifest_pair_N6_ego.csv`.** The validity gate in section 4
below is restated with 94.96, not ~66. Neither historical file is modified;
this is a merge-source correction only, applied via
`--merge-gnn-true-into/--merge-gnn-true-from` after cell B's confirmatory
run, exactly as designed in section 7 (only the source path changes).

## 2b. Reference points (measured before freezing gates, not fit to any run)

Committed to `results/phaseC_classical/reference_points.csv`
(`scripts/phaseC_classical_reference_points.py`, 500 episodes × 12 seeds
per cell/policy; does not touch the frozen `phaseC_reference_points.py` or
its CSV).

| cell | task | policy | mean return | seed sd | n |
|---|---|---|---:|---:|---:|
| C | relay_routing | random | **33.37** | 0.33 | 12 |
| C | relay_routing | oracle (privileged: sink sees source's goal) | **99.11** | 0.02 | 12 (analytic ceiling ≈ 100) |
| A | colocation_ring | random | **4.02** | 0.08 | 12 |
| B | pair_routing (existing, frozen) | random | ~50 | — | — |
| B | pair_routing (existing, frozen) | oracle | ~149 | — | — |

**Important asymmetry for cell C's liveness gate, caught before freezing:**
the relay reward is additive — `relay_prox` (relay agent → its OWN goal,
which it observes directly, zero routing needed) + `sink_prox` (sink →
source's goal, 2 hops, the payload). Because `relay_prox` is trivially
learnable with **no communication at all** (own-goal is in every relay
agent's base observation regardless of graph or obs_mode), a no-comms `mlp`
baseline clears the 33.4 random floor for free just by solving its own
relay-proximity term. **"Above the random floor" is therefore NOT sufficient
evidence of routing on cell C.** The C-liveness gate (section 3.1) is
reframed: `gnn_true` must clear not just the 33.4 random floor but the
**`mlp` arm's mean** (measured in the same smoke, `mlp` is already in cell
C's arm list) — that is the honest no-routing-competent reference. Both
numbers are reported in the smoke table.

## 3. Smokes — go/no-go gates (run BEFORE any 12-seed sweep)

Order fixed by design section 4 (cheapest kill-switch first):

1. **C-liveness (kill-switch).** `--cell C --smoke`, 3 seeds. Runs the full
   cell-C arm list (`mlp`, `gnn_true`, all `dcg_*`, `gat_true`) — `mlp` and
   `gnn_true` are the liveness pair. **GATE:** if `gnn_true`'s final-window
   mean is not clearly above both the 33.4 random floor AND the `mlp` arm's
   mean → relay payload is DEAD → do **NOT** run the cell-C confirmatory
   DCG sweep; record NO-GO and stop at cell C.
   **Budget calibration (frozen extension, per design section 4's own
   license to extend a too-short smoke up to 30k, documented rather than
   silently applied):** 3,000 steps is ~2% of the 150k confirmatory budget
   and very plausibly too short to see a 2-hop composition emerge from
   scratch. This smoke is run at **30,000 steps** (the stated smoke cap),
   not the script's default 3,000, specifically for cell C. The verdict is
   read off the **learning curve**, not only the final-window endpoint: if
   `gnn_true` is flat at/near the `mlp` level for the smoke's duration →
   NO-GO; if `gnn_true` is still visibly climbing above `mlp` at the end of
   the smoke but hasn't separated far → reported as **ambiguous /
   underpowered-smoke**, not a silent kill — the call is surfaced with the
   curve, not auto-decided by an endpoint threshold. A false NO-GO on the
   headline cell is treated as the worse error than a slightly late GO.
2. **B-concession.** `--cell B --smoke`, 3 seeds, default smoke budget
   (3,000 steps — B is a fast 1-hop task, existing `gnn_true` 150k data
   already anchors the comparison). Expect `dcg_jointobs_true` to
   match/beat the `gnn_true` reference (94.96, corrected per section 2c) —
   validates the DCG implementation end-to-end on a task it *should* solve.
3. **A-validity.** `--cell A --smoke`, 3 seeds, default smoke budget.
   `dcg_canonical` and `dcg_jointobs` must clear the 4.02 random floor and
   visibly coordinate co-location (home-turf sanity).

Smoke is a liveness/plumbing check, not a significance test — no
seeds/Holm/CI machinery applied at this stage, per design section 4.

## 4. Frozen decision rules (verbatim from design section 5)

Let "succeed" = final-window mean significantly above that cell's random
floor (cell C: above the `mlp` reference per section 2b's correction) and,
where compared, contrasts significant after Holm.

1. **DCG-jointobs fails relay (C) while `gnn_true` succeeds, AND classical
   wins full-info (A)** → **DIFFERENTIATION DEMONSTRATED — the headline.**
   Message passing composes observations across multi-hop task paths;
   pairwise payoff-propagation over the same factored reward cannot. Double
   dissociation: action-coordination under full observability vs multi-hop
   information routing under partial observability. Neither subsumes the
   other.
2. **DCG-jointobs also succeeds on relay (C)** → **SUBSUMED at the
   mechanism level → rescope.** A deep coordination graph reproduces the
   routing benefit; the paper's contribution narrows to the *empirical
   when*, not a unique mechanism.
3. **On 1-hop (B): `dcg_jointobs_true` matches/beats `gnn_true` (expected)
   while `dcg_canonical_true` fails** → the 1-hop benefit is
   observation-routing, not action-coordination, and is reproducible by a
   deep CG. **Report as an honest partial concession + mechanistic
   decomposition — not a win.**
4. **Classical fails BOTH A and C** → **VALIDITY GATE FAILS; implementation
   suspect, discard, do not interpret.**

## 5. Validity gates (from design section 6, one number corrected)

- DCG wins/ties on full-info edge-reward (A) before its relay result is
  trusted.
- Unit test: max-plus recovers the exact brute-force argmax on the disjoint
  matching/chain (where it must be exact) and converges (message-norm
  plateau) on complete/ring. **Status: implemented and green**
  (`tests/test_maxplus.py::test_max_plus_matches_brute_force_on_trees`,
  `::test_max_plus_matches_brute_force_multiple_seeds`,
  `::test_max_plus_message_norm_converges_on_loopy_graphs`; full suite 160
  tests green at pre-reg time, confirmed via `.venv` interpreter with
  `torch==2.12.1+cpu`).
- Reference points reproduce: relay floor (random policy, 33.37) and a
  full-info greedy oracle (99.11), measured in
  `scripts/phaseC_classical_reference_points.py` (relay-analogue of
  `phaseC_reference_points.py`); PairRouting floor (~50) / oracle (~149)
  already committed in `results/phaseC/reference_points.csv`, unchanged.
- **`gnn_true` at 150k reproduces its known ~95 (corrected from the
  design/handoff's "~66", which was the 30k number — see section 2c) on
  PairRouting-ego** (harness sanity), sourced from
  `results/phaseC_150k/manifest_pair_N6_ego_attention.csv`.

## 6. Fairness / matched protocol (unchanged from design section 3)

- Byte-identical env interface; `max_neighbors_override=N-1` so `obs_dim` is
  constant across true/wrong/complete graphs.
- 12 seeds (0–11) for every confirmatory cell; 150k env-step budget for
  every confirmatory cell (matched across the whole comparison table, house
  rule).
- DCG capacity ≥ GNN-QMIX params (generosity toward DCG); param audit
  printed by `scripts/phaseC_classical_sweep.py --params-only` before each
  sweep.
- Generosity (max-plus iterations, payoff-net width, lr) flows toward DCG
  only; GNN arms are never tuned to widen the gap.

## 7. Backfire analysis (unchanged from design section 8)

Reproduced without edits — B concedes (likely, pre-committed to outcome 3,
never framed as a win), C-liveness failing means concede complementarity
and skip the DCG relay sweep, "you starved the baseline" is neutralized
structurally (section 6 here), `dcg_complete` succeeding on relay via a
direct s–t shortcut is pre-registered as expected and illuminating, not a
failure of the design. Scope rule: "up to 150k steps," never "in
principle"; "a deep coordination graph (DCG)," not "all payoff
propagation."
