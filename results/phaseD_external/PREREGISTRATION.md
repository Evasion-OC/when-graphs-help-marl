# Pre-registration — Phase D external validity (MPE `simple_reference` pairs)

**Written and git-committed BEFORE any smoke or confirmatory result was
observed.** Freezes the design in `docs/EXTERNAL_VALIDITY_DESIGN.md` (read in
full for the mechanism argument, the rejected alternatives, and the honest
claimable/not-claimable pair) plus the implementation facts folded in below.
Implementation: `src/gnnmarl/envs/mpe_reference_pairs.py`
(`gnnmarl.envs.mpe_reference_pairs.MPEReferencePairs`),
`scripts/phaseD_external_sweep.py`, committed at `85dea03`; 180+ tests green
(`tests/test_mpe_reference_pairs.py` = 20 of these) confirmed on `.venv`
(`torch==2.12.1+cpu`) immediately before this pre-reg was written.

## 0. What is frozen here

- The cells run under this pre-reg: **E1** (primary confirmatory) and **E0**
  (non-confirmatory honesty anchor). **E2** is pre-specified but frozen OFF —
  see section 5's escalation trigger. Cell **R** (relay addendum) is governed
  by the ADDENDUM to `results/phaseC_classical/PREREGISTRATION.md` (section 8
  below), not by this document.
- The arms, env configs, and roles (design section 3, reproduced in section 1
  below).
- Metric: final-window (last 20%) mean episode return (native MPE reward);
  also report headroom fraction `(arm − mlp) / (oracle − mlp)` per design
  section 4. `oracle` and `mlp − oracle` are descriptive anchors, not part of
  the confirmatory contrast family.
- The frozen contrast family and test wiring (section 2).
- The five decision branches, verbatim from design section 5 (section 5
  below), including the oracle-gate conditionality that governs all of them
  and the frozen E2 escalation trigger.
- The three-gate smoke kill-switch order (design section 6, reproduced in
  section 4 below) and Trap-1/Trap-2 calibration notes added at pre-reg time
  (section 4.1).
- Budget: **NOT frozen yet.** Design section 4: "calibrated in smoke, then
  frozen in the pre-reg before confirmatory." The smoke cap is 30k-class env
  steps (matching the classical-baseline pre-reg's own extension convention),
  extendable **once**, documented. The frozen number and its rationale will
  be appended as a dated AMENDMENT to this file, committed separately,
  strictly before the E1/E0 confirmatory launch. No confirmatory run may
  precede that amendment's commit.

## 1. Cells and arms (unchanged from design, folded in with implementation facts)

| cell | k (N=2k) | comm | arms | seeds | confirmatory |
|---|---|---|---|---|---|
| **E0** | 1 (N=2) | ON (unmodified `simple_reference`) | `mlp`, `gnn` | 6 | **NO** — honesty anchor (design section 8); expected TIE; no headline gated on it |
| **E1** | 3 (N=6) | OFF (comm slot zeroed) | `gnn_true`, `gnn_wrong`, `gnn_complete`, `mlp`, `oracle` | 12 | **YES** — the vehicle's headline cell |
| **E2** | 5 (N=10) | OFF | same 5 arms | 12 | pre-specified; runs **only** on the frozen Branch-2a trigger (section 5) — NOT launched under this pre-reg |

**"True graph"** = the K native pair edges (block-diagonal 1-regular matching,
`2k <-> 2k+1`) — each agent's edge is exactly the partner who holds its
private target, fixed regardless of the communication graph. `gnn_wrong` is a
1-regular matching sharing no edge with `gnn_true` (undefined at k=1, hence
E0 has no wrong/complete arms). `gnn_complete` is all-to-all over 2k agents.

## 1a. Implementation facts folded into this pre-reg verbatim

**Privacy is structural, not injected.** `mpe2`'s `simple_reference` never
emits an agent's own target in its native observation; obs slice `[8:11]` is
the *partner's* `goal_color` (the payload that must be routed). Comm slice
`[11:21]` (the native `say` broadcast) is zeroed unless `comm_on=True`. With
comm off, `[8:11]` can reach the partner only via the coordination graph.
Oracle mode appends the partner's `[8:11]` block (i.e. this agent's own
target, which only the partner holds) directly into the observation,
`obs_dim` 24 vs 21 for the four confirmatory arms — this is the intended
headroom short-circuit, not a leak into the non-oracle arms.

**Param audit at K=3 (E1), verified via `--params-only` immediately before
this pre-reg:**

| arm | algo | params |
|---|---|---:|
| `mlp` | mlp_qmix | 38278 |
| `gnn_true` | gnn_qmix | 38534 |
| `gnn_wrong` | gnn_qmix | 38534 |
| `gnn_complete` | gnn_qmix | 38534 |
| `oracle` | mlp_qmix (on oracle obs) | 40774 |

`gnn_true`/`wrong`/`complete` are byte-identical (only `adjacency()`
differs). The Δ vs `mlp` (256 params) is the stabilised-GCN's
residual+LayerNorm overhead (`2 layers x 2 x gnn_hidden=64`), matching the
positive-endpoint disclosure convention (`STAGE_C_FINDINGS` section 3) — not
a capacity handicap against `gnn_true`. `oracle`'s Δ vs `mlp` (2496 params)
is the wider encoder input (24 vs 21 obs dims) plus the resulting state_dim
growth (144 vs 126) — `oracle` is a matched-architecture `mlp_qmix` on a
larger observation, not a differently-shaped network.

**Cell R arms** (`gnn_wrong`/`gnn_complete` on `relay_routing`, governed by
the ADDENDUM in section 8) are 23942 params each, identical to cell C's
already-committed `gnn_true` — verified via the same `--params-only` run.

**Centralized-mixer state confound check (disclosed, not a defect).** The
QMIX centralized mixer's state input is the flattened comm-zeroed
observation stack, which therefore contains all K `goal_color` blocks
(every partner's target, for every pair) regardless of which agent's graph
routed which payload to which actor. This state is **identical across the
four confirmatory arms** (`mlp`, `gnn_true`, `gnn_wrong`, `gnn_complete` all
see the same-shaped, same-content mixer state at matched obs_dim=21) — so it
does not confound the frozen contrast family: any arm-to-arm difference in
final return must come from what the *decentralized* per-agent actor
observed and acted on (obs_dim 21, comm-zeroed, graph-routed via the GNN
encoder for the three `gnn_*` arms only), never from privileged mixer-side
information that only some arms receive. `oracle`'s mixer state differs
(144-dim, oracle obs baked in) — expected and consistent with `oracle` being
an anchor outside the contrast family, not a confound inside it.

## 2. Contrast family and test wiring (frozen)

- **Family (exactly three contrasts, `gnn_true`-centred):**
  `gnn_true − gnn_wrong`, `gnn_true − gnn_complete`, `gnn_true − mlp`.
- Two-sided Welch's t with Holm correction across exactly these three
  contrasts, **and** two-sided Mann–Whitney U as a robustness check.
  **Significance requires BOTH tests** (house rule, matching
  `phaseC_classical`'s convention) — Holm-corrected Welch alone or MWU alone
  is not sufficient.
- Report Cohen's d and 95% CIs for each of the three contrasts.
- `oracle` and `mlp − oracle` (the headroom denominator) are **descriptive
  anchors only** — not in the Holm family, not subject to a significance
  gate themselves; they gate the vehicle (section 5's oracle-gate
  conditionality), not the contrast.
- Exact test wiring deferred to the research-statistician at analysis time
  (matching `phaseC_classical`'s deferral convention) — this section freezes
  *which* tests and *which* three contrasts, not the statistician's exact
  script.

## 3. Env / training config (frozen, matches implementation)

- `mpe_reference_pairs`, `k=3` (E1) / `k=1` (E0), `max_cycles=25`,
  `local_ratio=0.5` (mpe2 upstream default).
- Algo: `gnn_qmix` (stabilised: `gnn_residual=True, gnn_layernorm=True`,
  `gnn_layers=2`, `gnn_hidden=64`) for `gnn_true`/`wrong`/`complete`/E0's
  `gnn`; `mlp_qmix` (`gnn_layers=2`, no residual/LayerNorm — matches
  `phaseC_classical_sweep`'s `mlp` control) for `mlp` and `oracle`.
- Shared trainer hyperparameters: `buffer_capacity=50_000`, `batch_size=32`,
  `target_update_interval=200`, `lr=5e-4`, `gamma=0.99`, `eps_start=1.0`,
  `eps_end=0.05`, `eps_anneal_steps=0.8 * total_env_steps`.
- 12 seeds (0–11) for E1; 6 seeds (0–5) for E0 — matched *within* each cell,
  not across E0/E1 (E0 is non-confirmatory by design, cheaper by
  construction).
- Total env steps: **TO BE FROZEN FROM SMOKE** (section 0). Same budget
  applies to every arm within E1 (matched-protocol house rule) and every arm
  within E0.

## 4. Smoke gates (3 seeds, kill-switch order, design section 6)

Run `--cell E1 --smoke` (all 5 E1 arms in one run — the mlp/oracle/gnn_true
liveness triple is read off the same smoke) plus `--cell R --smoke`
(plumbing only, see section 8) and `--cell E0 --smoke` (cheap, non-gating).

1. **Privacy/implementation gate — `mlp` must be at the floor, not clear
   it.** `mlp` cannot observe its own target (structural absence, section
   1a); it should sit at or near the **do-nothing / random floor band**, not
   climb toward `oracle`. If `mlp` clears the floor band and tracks toward
   `oracle`-level return → the privacy wrapper is broken → **NO-GO**, report,
   stop (do not run any other gate or the confirmatory grid).
2. **Oracle-headroom gate.** `oracle` must **clearly** beat `mlp` within the
   smoke budget cap (30k-class, one permitted extension); if not →
   **NO-GO for the whole vehicle** — do not run the E1 confirmatory grid
   (Branch 0, section 5). A weak or ambiguous oracle separation is treated as
   failing this gate, not rounded up to a pass.
3. **`gnn_true` liveness (calibration, not an auto-kill).** Should climb
   toward `oracle` over the smoke curve. If flat, extend once (documented)
   and re-read the curve. Then **freeze the confirmatory budget** at the
   value where `oracle` has clearly solved the task and `gnn_true` has
   visibly begun separating from `mlp`/`gnn_wrong`/`gnn_complete` — not
   necessarily converged, just separating. Amend this pre-reg with that
   number and commit before any confirmatory launch.

Cell R's smoke (3 seeds, `--cell R --smoke`) is plumbing-only per section 8:
liveness = no crashes/NaNs, since it reuses the already-proven
`relay_routing` env and training stack.

### 4.1 Calibration notes added at pre-reg time (not in the original design doc)

- **Floor-band definition (avoids a false NO-GO on gate 1).** Because
  `simple_reference` has continuous physics and a distance-based reward, a
  policy that does not know its own target can still do better than *pure*
  uniform-random by exploiting generic structure (e.g. drifting toward the
  landmark centroid, standing still near the spawn region) — that is still
  "floored," not a privacy leak. The reference-points script therefore
  measures **both** a uniform-random policy **and** a do-nothing (always
  no-op) policy on the exact E1 config before freezing; the floor **band**
  is `[min, max]` of these two, and gate 1's pass condition is `mlp` sitting
  in or near that band **and** far below `oracle` — never "`mlp` above
  uniform-random" in isolation, which would produce spurious NO-GOs for
  benign reasons unrelated to privacy. This mirrors the additive-reward
  caution already on record in `results/phaseC_classical/PREREGISTRATION.md`
  section 2b (cell C's `mlp`-vs-random-floor correction).
- **Reference points committed before any gate is read**, to
  `results/phaseD_external/reference_points.csv` (random + do-nothing on
  the E1 config; oracle-informed expectation noted alongside). Cell R reuses
  the existing `results/phaseC_classical/reference_points.csv` relay
  references — no new reference-point run for cell R.

## 5. Frozen decision branches (verbatim from design section 5)

All branches are conditioned on the **oracle gate**: oracle significantly
beats mlp on both tests (task is learnable-when-routed and the metric has
headroom). No branch below is evaluated unless the oracle gate passes.

- **Branch 0 — oracle fails:** UNINTERPRETABLE, not a null. No paper
  sentence; raise budget and re-smoke, or record the vehicle as intractable
  and flag to authors.
- **Branch 1 — full replication** (`gnn_true` beats `complete` AND `wrong`
  AND `mlp`, both tests, Holm): external validity earned; limitation
  discharged. Paper sentence per design section 5; the limitations edit
  replaces the "natural next step" promise, retaining only the residual
  caveat that the coordination structure remains author-imposed.
- **Branch 2a — ties complete, beats wrong+mlp:** structure-over-wrong-and-
  no-graph earned; "more comm didn't help" NOT earned on MPE (dilution weak
  at K=3); scoped sentence per design. **This is the frozen E2 escalation
  trigger** (design section 10.3): if and only if E1 lands here, K=5 (cell
  E2, 12 seeds, same frozen budget) may be launched as a pre-specified
  follow-up to test whether "more comm doesn't help" strengthens with more
  diluters. E2 is NOT launched under this pre-reg and requires the E1
  analysis to land Branch 2a first.
- **Branch 2b — beats complete+wrong, weak vs mlp:** structure contrasts
  earned; `true − mlp` reported as directional/underpowered.
- **Branch 3 — TRUE NULL** (oracle passes, `gnn_true` ties `mlp`): genuine
  failure to transfer; the paper reports it and explicitly rescopes the
  external-validity claim to CoordGrid + TokenMatch + deep-CG constructed
  archetypes (the positive endpoint stands on those, independent of this
  vehicle). HEADLINE-THREATENING; **authors pre-commit now to publishing
  this branch if it lands** (design section 10.2).

## 6. Overclaim guardrail (frozen, design section 1)

Regardless of which branch lands, the paper may claim (Branch 1/2a/2b only)
"the task-matched graph beats a density-matched wrong graph, all-to-all,
and/or the no-graph control at byte-identical parameters — so the structure
effect is not an artifact of our gridworld or token-matching dynamics: it
survives third-party continuous physics and a third-party reward, on a task
whose private-partner-goal structure is native rather than injected." The
paper may **never** claim "replicates on an unmodified benchmark" or
"replicates on coordination structure we did not design" — composition + the
pairing graph + comm-disable remain author-imposed regardless of outcome.

## 7. Cheap N=2 honesty anchor (E0, non-confirmatory, design section 8)

Unmodified N=2 `simple_reference`, comm ON, `gnn` vs `mlp`, 6 seeds. Expected
TIE, reported as: "the graph is redundant when the env supplies its own
channel — the topology question only acquires content at N ≥ 4 with the
channel controlled." No headline is gated on E0; no Holm family applies to
it (2 arms, descriptive comparison only, reported alongside E1 for honesty).

## 8. Relay-cell addendum (cell R) — governed separately

Cell R (`gnn_wrong` + `gnn_complete` added to the existing `coord_grid`
`relay_routing` cell C) is pre-registered as an **ADDENDUM to
`results/phaseC_classical/PREREGISTRATION.md`**, not as part of this
document's contrast family or branches. See that file's ADDENDUM section for
the frozen budget (12 seeds, 150k, unconditional — does not depend on E1's
budget freeze), the three frozen expectation patterns, and the explicit
statement that cell R cannot un-null cell C's already-committed
`gnn_true ≈ mlp` result.

## 9. Validity gates (summary, expanded in section 4)

- Privacy confirmed by source inspection (section 1a) AND empirically (`mlp`
  floors, gate 1).
- Harness-trains-MPE sanity: unmodified N=2 `simple_reference` (E0, comm on)
  produces non-degenerate learning for `gnn`/`mlp` (not literally required
  to beat any external SOTA — internal sanity only).
- `oracle` solves and `mlp` floors in the confirmatory config (gates 1–2,
  re-verified at the frozen confirmatory budget, not only at smoke budget).
- Reference points measured and committed to
  `results/phaseD_external/reference_points.csv` before any gate is read.
- `tests/test_mpe_reference_pairs.py` (adjacency/obs_dim/privacy-slice
  tests) green — confirmed at pre-reg time, full suite 180+ tests.
