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

## AMENDMENT (2026-07-31) — smoke results and gate verdict

**Written and committed after the smoke, before any confirmatory launch —
this amendment records the outcome and the resulting decision; it does NOT
retroactively change sections 0–9 above.**

### A.1 Reference points (committed, section 4.1's procedure)

`results/phaseD_external/reference_points.csv`
(`scripts/phaseD_reference_points.py`, 500 episodes × 12 seeds), E1 config
(k=3, comm off, `graph="true"`, non-oracle obs):

| policy | mean return | seed sd | n |
|---|---:|---:|---:|
| random | −84.81 | 0.673 | 12 |
| do_nothing | −78.14 | 0.595 | 12 |

Floor band (section 4.1): **[−84.81, −78.14]**.

### A.2 Smoke 1 — 3,000 steps, 3 seeds (default smoke budget)

All 5 E1 arms and cell R's 2 arms run (`--smoke --cell E1,R`). **Uninterpretable
at this budget**: every E1 arm's per-episode return *degrades* monotonically
over the run (e.g. `mlp` mean-by-decile: −87.7 → −89.4 → −90.4 → −149.2 →
−157.0), consistent with `eps_anneal_steps = 0.8 × 3000 = 2400` collapsing
exploration well before the Q-network has learned anything useful — a
known QMIX failure mode at very short budgets, not evidence about any gate.
Per section 4/design section 6, this licenses the one permitted extension.
Cell R's smoke (3 seeds, 3000 steps) is plumbing-only and passed cleanly:
`gnn_wrong` 31.25, `gnn_complete` 31.67 (sd ~1.5, n=3), no crashes/NaNs,
values plausible next to the existing `relay_routing` random floor (33.37).

### A.3 Smoke 2 — 30,000 steps, 3 seeds (the one permitted extension, used)

`--cell E1 --steps 30000 --seeds 3`. Final-window means (n=3 each):

| arm | mean | sd |
|---|---:|---:|
| `mlp` | −58.26 | 0.36 |
| `oracle` | −57.93 | 0.39 |
| `gnn_true` | −63.67 | 1.95 |
| `gnn_wrong` | −64.81 | 0.57 |
| `gnn_complete` | −62.49 | 0.10 |

Per-episode deciles for `mlp`/`oracle`/`gnn_true` (seed 0) show all three
curves **plateau by ~episode 480/1200 (40% through) and stay flat for the
remaining 60%** — this is convergence, not an artifact of reading the
window too early.

**Gate 1 (privacy) — PASS, with the Trap-1 nuance applied.** `mlp`
(−58.26) sits well above the floor band, which on its own would look like a
leak. Resolved by the greedy-policy diagnostic below (A.4): `mlp` sits
*below* the identity-blind `greedy_centroid` scripted ceiling (−53.74) and
nowhere near the informed `greedy_target` ceiling (−31.64) — i.e. `mlp`'s
improvement over the floor is exactly the benign structural exploitation
Trap 1 anticipated (agents can see all 3 landmark *positions*, just not
which one is the target, so a trained policy can beat pure noise without
any privacy leak). `mlp` is in the correct "blind" performance band, not
the "informed" one. **Privacy confirmed both by source (section 1a) and now
empirically.**

**Gate 2 (oracle-headroom) — FAIL at the extension cap.** `oracle`
(−57.93) does **not** clearly beat `mlp` (−58.26); the two track each other
within noise for 6 consecutive deciles. Per section 4/design section 6:
this is the kill-switch. **No E1 confirmatory launch.**

### A.4 Diagnostic — does headroom exist at all? (resolves which failure mode gate 2 is)

A trained-oracle-≈-trained-mlp result is consistent with two different
diagnoses that call for opposite responses (no env headroom → Branch 0,
rescope the vehicle; headroom exists but training didn't reach it →
calibration problem, not a null). Added at amendment time (not a training
run, not a budget extension — a scripted, zero-learning reference in the
same style as `phaseC_classical_reference_points.py`):
`scripts/phaseD_greedy_reference.py`, `results/phaseD_external/greedy_reference.csv`
(200 episodes × 12 seeds), same E1 config:

| policy | mean return | seed sd | n |
|---|---:|---:|---:|
| `greedy_target` (hand-scripted, uses exactly the info `oracle` mode injects) | **−31.64** | 0.426 | 12 |
| `greedy_centroid` (hand-scripted, identity-blind — landmark positions only, no target) | −53.74 | 0.561 | 12 |

**Headroom clearly exists** (a ~22-point gap between a zero-learning
informed script and a zero-learning blind script, dwarfing the noise in
either). The trained `oracle` arm (−57.93) sits at the *blind* level, not
partway toward the informed ceiling it was structurally handed. **Diagnosis:
this is failure mode (b) — a training/calibration problem (30k steps is not
enough for this harness's QMIX to learn to exploit a small privileged slice
of a 24-dim continuous observation), not failure mode (a) — the vehicle is
not intrinsically headroom-free.** This is consistent with, not contradicted
by, the design's own Branch-0 language ("raise budget and re-smoke, or
record the vehicle as intractable") — the evidence here supports "raise
budget," not "intractable."

### A.5 Verdict and decision

**NO-GO for the E1 confirmatory launch under this pre-reg's frozen smoke
protocol.** The oracle-headroom gate is a hard kill-switch (section 4) and
the one permitted budget extension (3,000 → 30,000 steps) has already been
used. Per the task's own instruction ("if a run reveals the regime is
underpowered or unlearnable, say so and propose the budget/calibration
change — do not silently extend budget"), going further (e.g. re-smoking at
150k) requires new authorization, not a unilateral extension past the
pre-registered cap. **The confirmatory budget is NOT frozen. No E1, E2, or
E0 runs are launched under this pre-reg.**

**Cell R is independently governed** (separate ADDENDUM to
`results/phaseC_classical/PREREGISTRATION.md`, own frozen budget, own
plumbing-only smoke gate already passed cleanly, not conditioned on E1's
oracle-headroom finding). It is **held, not launched**, pending an explicit
go-ahead from the authors, in line with the instruction to stop and report
rather than launch when a kill-switch fires anywhere in the protocol.

**Recommendation to authors (not acted on unilaterally):** the vehicle looks
tractable in principle (proven headroom) but the harness's default
hyperparameters (`lr=5e-4`, `buffer_capacity=50k`, `target_update=200`,
`eps_anneal=0.8×steps`, reused verbatim from CoordGrid) may be miscalibrated
for continuous MPE physics with a 21–24-dim observation and non-toroidal
dynamics. Two independent next steps, not mutually exclusive: (1) a fresh,
explicitly-authorized re-smoke at a materially larger budget (e.g. 150k,
matching the house standard used elsewhere in this repo) to see whether
`oracle` eventually separates given enough samples; (2) a short
hyperparameter-only recalibration pass (learning rate / target-update
interval / exploration schedule) on the `oracle` arm alone (never widen an
advantage by tuning `mlp` down or `gnn_true` up — house rule) before
re-spending a full smoke budget. Both are genuinely new decisions for the
authors, not implied by the existing pre-registration.

### A.6 Correction to A.4 (caught in post-hoc review, before any further action)

**`greedy_target` is NOT "exactly the info `oracle` mode injects" as A.4
claimed — it is strictly stronger.** Verified directly:
`_GOAL_SLICE = slice(8, 11)` in `mpe_reference_pairs.py` is the target
landmark's **color** (a fixed 3-vector — `mpe2`'s `reset_world` hardcodes
`landmarks[0].color=[.75,.25,.25]` (red), `[1]=[.25,.75,.25]` (green),
`[2]=[.25,.25,.75]` (blue), **identically on every episode/seed**, only
`state.p_pos` — the landmark *positions* the agent separately observes via
`entity_pos` — is re-randomized per episode). `scripts/phaseD_greedy_reference.py`'s
`greedy_target` policy instead reads `goal_b.state.p_pos` directly from the
`mpe2` world object — the landmark's actual position, bypassing the
color→landmark-index→position lookup the `oracle` arm's network would
actually have to perform (map the injected 3-dim color to one of the three
fixed-order `entity_pos` slots, then steer to that slot's position). So
**−31.64 is a position-omniscient ceiling, not the `oracle` arm's true
achievable ceiling** — A.4 established *information sufficiency* (color
does determine the target unambiguously, under an always-fixed
color↔landmark-index mapping) but did **not** establish *learnability* of
that indirection within 30k steps. The oracle arm plateauing exactly at the
blind `greedy_centroid` level is therefore consistent with EITHER (i) a
budget shortfall (A.5's original reading) OR (ii) the network never having
grounded the static color→slot mapping at all — the diagnostic in A.4
cannot distinguish these, because it skips the indirection entirely.

One fact favours (i) being worth testing rather than dismissed: the
color→landmark-index mapping is **static across every episode** (not
re-randomized), so it is a fixed function learnable in principle by a
shallow net without any per-episode relational binding — a plausibly easy
target for more gradient steps, not a fundamentally hard credit-assignment
problem. But this is a plausibility argument, not evidence; a genuine
learnability test (not attempted here) would need either substantially more
training or a probe specifically isolating the color→slot mapping.

**This does not change the verdict in A.5** — NO-GO stands, nothing is
launched, cell R/E0 remain held. It changes the recommendation's framing:
a re-smoke at a larger budget should be described to authors as **testing
whether the color→landmark-index indirection is learnable**, not as
"confirming known headroom the oracle should reach" (authors must not read
−31.64 as the oracle arm's expected ceiling with more steps). A cleaner
alternative operationalization, worth considering instead of/alongside a
budget increase: redefine `oracle` to inject the target landmark's
*relative position* directly (rather than its color) — that is the more
honest "routing done for free" ceiling A.4 intended, and removes the
indirection-learnability confound entirely. This is a design change, not
implied by the current pre-registration, and would need its own
authorization before implementation.

### A.7 Runtime plumbing check (rules out a third failure mode)

Before treating (i)/(ii) as the exhaustive failure-mode space, verified a
third candidate directly: that `oracle=True`'s injected slot `obs[:, 21:24]`
is dead/zero/wrong at runtime (a `_compose_obs` bug), which would produce
the *identical* observed signature (`oracle ≈ mlp`, since `oracle` would
just be `mlp` plus three inert dims) for a reason unrelated to learnability
or budget. Checked directly by instantiating
`MPEReferencePairs(k=3, graph="true", comm_on=False, oracle=True, seed=0)`
and inspecting the composed observation, both at `reset()` and after one
`step()`: `obs[0, 21:24]` equals `obs[1, 8:11]` (agent 1's own goal_color)
and vice versa, both non-zero and distinct — the injected slot correctly
carries the partner's target color, symmetric across both agents, and
persists correctly across a step. **Plumbing confirmed correct; failure
mode (iii) ruled out.** The (i) budget-shortfall / (ii)
indirection-not-grounded framing in A.6 is the live, exhaustive-as-checked
diagnosis space.

## AMENDMENT 1 (2026-07-31) — authorized 150k re-smoke + position-oracle diagnostic

**Written and git-committed BEFORE the re-smoke or the position-oracle diagnostic
was run.** Does not reopen or revise the `## AMENDMENT (2026-07-31)` section
above (the 30k-cap NO-GO verdict stands as the record of that smoke); this is
a new authorization layered on top of it, granted explicitly by the authors
per that amendment's own "Recommendation to authors" (A.5) and its A.6
follow-up, in response to the report that (i) headroom clearly exists
(`greedy_target` vs `greedy_centroid`, A.4) and (ii) the trained `oracle` arm
plateauing at the blind level is ambiguous between a budget shortfall and an
ungrounded color→landmark-index indirection (A.6).

### A1.1 What is authorized

1. **One 150k-class re-smoke (3 seeds)** of `{oracle, mlp, gnn_true}` — the
   same three arms whose curves the original smoke gate reads. Framing is
   frozen as: **this tests whether the color→landmark-index indirection
   becomes learnable at the 150k-class budget used elsewhere in this paper**
   (matching the house standard e.g. `results/phaseC_150k/`,
   `results/phaseC_classical/`'s 150k cells) — it is explicitly **NOT** framed
   as "confirming a known-reachable ceiling." A.6 already established that
   `greedy_target`'s −31.64 is a position-omniscient ceiling strictly above
   what a network doing the color→slot lookup could achieve even with
   unlimited budget; no number from A.4/A.6 is treated as `oracle`'s expected
   value here.
2. **One diagnostic arm, `oracle_pos`** (position injection instead of color
   injection), run alongside the three above at the same 3 seeds / 150k
   budget. **Labeled DIAGNOSTIC ONLY.** Its outcome cannot gate or enter any
   confirmatory family (the frozen three-contrast family in section 2 is
   unchanged: `gnn_true−gnn_wrong`, `gnn_true−gnn_complete`, `gnn_true−mlp`;
   `oracle_pos` is not in it and never will be — it is not even a candidate
   arm for E1's confirmatory grid, which keeps `oracle` at
   `oracle_mode="color"` exactly as designed). It exists solely to
   disambiguate, in the Branch-0 write-up **if the gate fails again**,
   "the indirection is unlearnable" (color-`oracle` fails, `oracle_pos`
   clearly separates — the position-injected ceiling is reachable, so the
   specific blocker is the color→slot lookup, not the harness/budget more
   generally) from "other harness failure" (neither `oracle` nor `oracle_pos`
   separates from `mlp` — points at a harness/hyperparameter problem
   orthogonal to the indirection).

### A1.2 Gate rule for the re-smoke (frozen now, applied when it lands)

**Unchanged from the original gate 2** (section 4, item 2 / design section 6):
curve-based, not a statistical test. **PASS iff `oracle` clearly separates
above `mlp`** — a visible gap between the two curves with no overlap in their
final (last-20%) windows across the 3 seeds, read the same way the original
30k smoke was read (A.3's decile tables). The confirmatory both-tests
criterion (Welch + MWU, Holm-corrected, on the three-contrast family) is
**deferred to the confirmatory run itself**, exactly as originally frozen —
this re-smoke is still a smoke, not a substitute for the confirmatory grid.
`gnn_true` liveness is read the same way as before: calibration information
(should visibly begin separating from `mlp`), not an independent auto-kill.
`oracle_pos` does not participate in the PASS/FAIL decision at all (A1.1.2).

### A1.3 Frozen consequence if the gate FAILS

**Branch 0, final.** The vehicle is intractable in this harness at the
budgets tested (30k and 150k). **No further budget extensions** — this
authorization exhausts the re-smoke option; a hypothetical 300k+ re-smoke
would require fresh authorization this document does not grant and the
project's compute budget does not obviously support. **No confirmatory run.**
The result does **not** enter the manuscript as evidence for or against
external validity on this vehicle. The limitation (an untested MPE variant)
remains open, with the following honest one-sentence disclosure option
available for the limitations section, to be used verbatim or adapted at
write-up time, **only if this branch is reached**:

> "We attempted an external-validity check on a composed third-party MPE
> environment (`simple_reference` pairs, comm disabled) but found the
> harness's QMIX-style training could not learn to exploit even a
> directly-injected oracle signal within a 150k-step budget matching the
> rest of the paper, despite confirming (via a hand-scripted reference
> policy) that the environment itself has substantial headroom; we report
> this as an inconclusive attempt rather than a null result, since a null
> result requires the baseline to actually solve the task it is being
> compared against."

### A1.4 Implementation change authorized

`oracle_mode="color"|"position"` kwarg on `MPEReferencePairs`
(`src/gnnmarl/envs/mpe_reference_pairs.py`), default `"color"` (preserves
the existing `oracle` arm's behavior and `obs_dim=24` byte-for-byte — no
change to any already-run or already-authorized confirmatory config).
`oracle_mode="position"` injects the partner-held own-target landmark's
**relative position** (`obs_dim=23`, 2 floats) in place of its color
(`obs_dim=24`, 3 floats) — the same landmark the color mode already injects
(`agent_p`'s own target is `partner.goal_b`; see module docstring), just
its position instead of its color, so the network is hardwired to the target
without needing to learn the fixed color→landmark-index→position lookup at
all. This is a genuinely new diagnostic env config, authorized here, not
implied by the original pre-registration or design doc. Implemented behind
a unit test (`tests/test_mpe_reference_pairs.py`); full suite re-verified
green before any run under this amendment; commit precedes the runs.

### A1.5 What remains untouched

Cell R's own governance (section 8 above / the ADDENDUM to
`results/phaseC_classical/PREREGISTRATION.md`) is unconditional on this
amendment and on E1's oracle-headroom finding — its launch is authorized
separately (that ADDENDUM's own frozen budget) and proceeds regardless of
which way this re-smoke lands. E0/E2 remain un-launched pending E1's own
gate outcome, unchanged from the base pre-reg.

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
