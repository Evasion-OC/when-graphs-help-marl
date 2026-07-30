# Design: discharging the external-validity limitation before JAAMAS

Status: DESIGN — pending author approval; nothing has run.
Author: marl-research-scientist agent, 2026-07-31.

## 0. The decision, in one paragraph

Run the full isolator family (`gnn_true` / `gnn_wrong` / `gnn_complete` / `mlp`,
stabilised GCN, byte-matched params) on **K composed PettingZoo `simple_reference`
pairs (N = 2K) with the scripted communication channel disabled, so the coordination
graph is the sole route for the partner's goal.** This is the one vehicle that
(a) uses genuine third-party dynamics/observation/reward, (b) instantiates
private-partner information *natively* rather than by injection, and (c) is legal for
the wrong-graph and all-to-all controls that carry the paper's signature contrast.
It discharges the *literal* promise the paper already carries (limitations: "an
MPE/LBF **variant** with private per-agent goals"). It does **not** earn "replicates
on an unmodified benchmark" — and the design below is built so the authors never
claim that.

## 1. Why this vehicle, and the honest external-validity statement it supports

**What `simple_reference` gives natively (the strong fact — lead with it).** Each of
the 2 agents observes `goal_id` = *the partner's* target landmark, plus a
`communication` slot and a 50-way `say × move` action. An agent does **not** observe
its own target; the partner holds it. This mutual private-partner structure is
*exactly* pair_routing's information topology, and — decisively — **we did not
inject it.** Any LBF "private-goals variant" would require authoring the
private-goal structure into observation and reward; `simple_reference` ships with it.

**Minimal change:** disable the scripted comm channel (fix the `say` sub-action;
zero the `communication` obs slot) and impose the coordination graph as the routing
substrate over per-agent GCN embeddings — identical stack to the positive endpoint.
With comm off, agent i's `goal_id` (its partner's target) must reach the partner
over the graph or not at all. Everything else — continuous 2D physics with momentum,
landmark relative-position sensing, the native distance-based reward — is
third-party code we did not write.

**Claimable (Branch 1):** "the task-matched graph beats a density-matched wrong
graph, all-to-all, and the no-graph control at byte-identical parameters — so the
structure effect is not an artifact of our gridworld or token-matching dynamics: it
survives third-party continuous physics and a third-party reward, on a task whose
private-partner-goal structure is native rather than injected."

**NOT claimable (overclaim guardrail, frozen now):** "replicates on an *unmodified*
benchmark"; "replicates on coordination structure we did not design." What transfers
is robustness to third-party dynamics and reward, not to a third-party coordination
structure (composition + pairing graph + comm-disable remain author-imposed).

**Steelman referee + pre-written answers:**
- "You re-implemented pair_routing with MPE rendering." — Conceded as to the
  coordination structure; the scoped rebuttal: physics and reward are third-party
  and the private-goal property is native, so the claim is robustness to
  dynamics/reward — a real, previously-untested axis. Anchors on the paper's own
  wording ("an MPE variant").
- "Why disable communication in a communication benchmark?" — Learning to
  communicate (emergent-communication RL through a discrete `say` action) is a
  separate hard problem orthogonal to the thesis (routing TOPOLOGY, not protocol
  emergence); the harness is feedforward-only (verified — no recurrent path in
  src/gnnmarl/algos), so protocol emergence is not learnable in it regardless.

## 2. Rejected alternatives (decisive)

(a) **Unmodified N=2 simple_reference / speaker_listener — REJECTED as confirmatory
vehicle.** At N=2 the true graph and complete graph coincide; a density-matched
wrong graph is impossible; with comm ON, `mlp` can solve via the scripted channel.
The signature contrast is undefined at N=2. (A cheap non-confirmatory N=2 use
survives — §8.)

(b) **Graph-gated NATIVE comm (say-symbol reaches only graph neighbours) — REJECTED
on tractability, not stack-purity.** Requires emergent-communication learning that
feedforward QMIX cannot solve in realistic budget. Comm-disable routes a static
per-episode `goal_id` feature in one GCN hop — tractable. Disclose the reason; note
graph-gated comm as the right design for a future recurrent/actor-critic harness.

(c) **Composed LBF / rware with injected private goals — REJECTED.** Neither exposes
private per-agent goals natively; a variant would rebuild CoordGrid inside LBF and
forfeit the "native structure" advantage.

## 3. Arms and what "true graph" means

N = 2K composed `simple_reference` agents; native comm disabled; reward = sum of the
K native pair rewards (freeze sum vs mean in the pre-reg). GCN reads `adjacency()`.

| arm | adjacency | role |
|---|---|---|
| gnn_true | the K native pair edges (block-diagonal 1-regular matching) | task-matched graph |
| gnn_wrong | a 1-regular matching sharing NO edge with the true pairing (mispair across sub-envs), identical density | structure control |
| gnn_complete | all-to-all over 2K agents | channel control |
| mlp | none (parameter-matched no-graph control) | a-priori floor |
| oracle | partner's goal_id concatenated into own obs (routing done for free) | headroom / non-null gate — IN the confirmatory run |

"True graph" = the perfect matching pairing each agent with the partner whose target
it holds (the K original sub-environments' edges) — the task structure, fixed
regardless of the communication graph.

**obs_dim constancy is free here** (adjacency is separate from the observation; with
comm disabled the obs is identical across gnn arms — a cleaner isolation than
CoordGrid's max_neighbors_override).

**Param matching:** gnn_wrong/gnn_complete byte-identical to gnn_true (only
adjacency differs); mlp matched per the positive-endpoint protocol (disclose the
~256 LayerNorm-param gap as in STAGE_C_FINDINGS §3).

**Primary cell: K=3 (N=6)**, matching the positive endpoint's N=6 headline.
Optional K=2 (N=4) robustness point if budget allows.

## 4. Protocol

- Seeds: 12 (0–11) per confirmatory cell.
- Metric: final-window (last 20%) mean episode return (native MPE reward); also
  report fraction of oracle−mlp headroom captured: (arm − mlp)/(oracle − mlp).
- Contrast family (frozen, gnn_true-centred): gnn_true−gnn_wrong,
  gnn_true−gnn_complete, gnn_true−mlp. Two-sided Welch t, Holm across exactly these
  three; two-sided MWU as robustness; significance requires BOTH (house rule).
  Cohen's d, 95% CIs. oracle and mlp−oracle are descriptive anchors, not in the
  family.
- Budget: NOT hard-coded — calibrated in smoke (raise until oracle solves and
  gnn_true begins separating), then frozen in the pre-reg before confirmatory.
  Expect more than the gridworld's 30k; justify by the oracle gate.

## 5. Frozen decision rules (before the confirmatory run)

All branches conditioned on the ORACLE GATE (oracle significantly beats mlp on both
tests: task is learnable-when-routed and metric has headroom).

- **Branch 0 — oracle fails:** UNINTERPRETABLE, not a null. No paper sentence; raise
  budget and re-smoke, or record the vehicle as intractable and flag to authors.
- **Branch 1 — full replication** (gnn_true beats complete AND wrong AND mlp, both
  tests, Holm): external validity earned; limitation discharged. Paper sentence per
  design §5; limitations edit replaces the "natural next step" promise, retaining
  only the residual caveat that the coordination structure remains author-imposed.
- **Branch 2a — ties complete, beats wrong+mlp:** structure-over-wrong-and-no-graph
  earned; "more comm didn't help" NOT earned on MPE (dilution weak at K=3); scoped
  sentence per design.
- **Branch 2b — beats complete+wrong, weak vs mlp:** structure contrasts earned;
  true−mlp reported as directional/underpowered.
- **Branch 3 — TRUE NULL (oracle passes, gnn_true ties mlp):** genuine failure to
  transfer; the paper reports it and explicitly rescopes the positive endpoint to
  constructed archetypes; exact sentence frozen in design §5. HEADLINE-THREATENING;
  authors pre-commit to publishing this branch.

## 6. Smoke gates (3 seeds each, kill-switch order)

1. **Privacy/direction gate:** verify in mpe2 source that goal_id is the PARTNER's
   target and self-target is unobservable with comm off; operationalised: mlp must
   FLOOR (if mlp clears floor, the target is not private → NO-GO, fix wrapper).
   Adjacency/obs_dim unit tests.
2. **Oracle-headroom gate:** oracle must clearly beat mlp within smoke budget cap;
   else NO-GO for the vehicle (harness cannot train it) — do not run confirmatory.
3. **gnn_true liveness (calibration, not auto-kill):** should climb toward oracle;
   if flat, extend budget once (documented), re-read curve; freeze confirmatory
   budget where oracle solves and gnn_true has begun separating.

## 7. Validity gates

Privacy confirmed (source + mlp-floors); harness-trains-MPE sanity (unmodified N=2
simple_reference, comm on, non-degenerate return; published SOTA not the yardstick);
oracle solves + mlp floors in confirmatory; reference points measured and committed
before freezing; adjacency/obs_dim tests green.

## 8. Cheap N=2 honesty anchor (optional, NOT confirmatory)

Unmodified N=2 simple_reference, comm ON, gnn vs mlp → report the expected TIE:
"the graph is redundant when the env supplies its own channel — the topology
question only acquires content at N ≥ 4 with the channel controlled." Two cells,
cheap; no headline gated on it.

## 9. Implementation sketch and compute

To write: src/gnnmarl/envs/mpe_reference_pairs.py (compose K simple_reference
instances via mpe2, lazy import behind [mpe] extra, modeled on mpe_adapter.py:
obs stacking; comm disable; adjacency() true/wrong/complete via coord_grid's
_build_adjacency logic; summed reward; oracle mode); configs + sweep script modeled
on phaseC_classical_sweep.py (--smoke, --params-only); adjacency/privacy unit tests.

Reused: gnn_qmix (stabilised, 2-layer, hidden 64), mlp control, trainer, logger,
phaseC_analysis.py, reference-points pattern.

Compute: 48 confirmatory + 12 oracle + ~6 smoke runs; 30k-class budget ≈ overnight
on 16 cores; 150k-class ≈ 2–4 nights. Freeze from smoke.

## 10. Headline-rescoping flags (authors approve with eyes open)

1. **False-null is the #1 risk** — the oracle-in-confirmatory + Branch-0/3
   conditional is the entire defense; non-negotiable in the pre-reg.
2. **Branch 3 rescopes the external-validity claim** (positive endpoint still stands
   on CoordGrid + TokenMatch + deep-CG). Authors decide NOW to publish it if it lands.
3. **Branch 2a plausible at K=3** (partner's goal is 1 of 5 diluters). If the sharp
   "more comm didn't help" claim on MPE matters, pre-specify a K=4–5 escalation cell
   BEFORE running.
4. **Never frame as "unmodified benchmark"** — write the claimable/not-claimable
   pair into the paper before the run.

## 11. Secondary: relay-cell completion addendum

Add gnn_wrong + gnn_complete (12 seeds, 150k) to the existing relay_routing cell C
as a pre-registered ADDENDUM to results/phaseC_classical/PREREGISTRATION.md. It
CANNOT un-null cell C's fixed gnn_true≈mlp result; it only fills the missing-arms
sub-limitation (within-GNN structure on the relay). Frozen expectations:
- Most likely: all GNN arms at/below the ~61 mlp reference → characterized
  within-GNN null; delete the corresponding limitation clause; headline unchanged.
- gnn_true > wrong ≈ complete but ≈ mlp → "topology-sensitive but sub-baseline";
  still no multi-hop routing claim.
- gnn_true suddenly clears mlp → CONFLICTS with committed data; flag for
  re-examination (drift), NOT an automatic win.
