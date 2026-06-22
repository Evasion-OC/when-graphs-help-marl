# Stage C findings: a controlled boundary — message passing helps when (and only when) it follows the task's coordination edges

**Status: positive boundary result, pre-registered, and revised after a 4-reviewer
mock panel + rebuttal pass.** Stage C identifies a regime in which a coordination
graph gives a large, significant, reproducible advantage over no-graph baselines in
cooperative value factorization, and isolates that the advantage comes from the
graph's *structure* (which edges), not from extra capacity, not from "more
communication," and — for the contrast that is cleanly identified — not from the
parameter count. This **completes** (does not overturn) the project's prior negative
result: graphs are neutral-to-harmful in convention-reducible / full-information
settings, and help where the graph is the only path for task-structured private
information and the communication topology matches the coordination structure.

All confirmatory claims are pre-registered in `docs/STAGE_C.md` (hypotheses, metric,
seeds, decision rule fixed before the runs). Metric: final-window mean return (last
20% of episodes). Stats: two-sided Welch t (Holm-corrected across the registered
3-comparison family) AND two-sided Mann–Whitney U; Cohen's *d*; 95% CIs; >=12 seeds
for every confirmatory cell.

> **Read the controls first.** "A graph beats no-communication on a routing task" is
> near *a priori* — you cannot route information an agent is barred from observing. The
> load-bearing, non-obvious finding is that **a wrong graph of identical density and
> all-to-all communication both sit at the random floor, while only the task-matched
> graph rises.** More communication did not help; the *right* communication did.

---

## 1. The question Stages A/B left open

Earlier stages found `GNN-QMIX` never beats the parameter-matched `MLP-QMIX` control.
But those tasks were either reducible to a global convention (the graph is never
needed) or, when routing was required, not learnable. None tested whether graph
*structure* — *which agent communicates with which* — matters with the channel and
parameter count held fixed. That is the project's named question, and Stage C answers
it on a task constructed so the answer is identifiable.

## 2. A task where structure is load-bearing by construction

`CoordGrid(pair_routing=True)`: every agent privately observes its own goal and is
rewarded (dense, toroidal proximity) for reaching its **partner's** goal, where a
fixed perfect matching defines partners. Partners must therefore *swap* goals over
their edge. The needed information is held by **one specific other agent**, so the
communication graph must match the task structure. `max_neighbors_override=N-1` holds
`obs_dim` constant across comm graphs (unit-tested). The arms are the *same*
`gnn_qmix` (repaired GCN: residual+LayerNorm, 2 layers, hidden 64); only
`env.adjacency()` differs: `gnn_true` (the matching), `gnn_wrong` (a different
matching of identical 1-regular density), `gnn_complete` (all-to-all). `mlp` is the
no-comm parameter-matched control.

## 3. Headline — it's the structure, not the channel (N=6, 3×3, ego, 12 seeds, 30k)

Reference points on this task (measured): the **random/do-nothing return ≈ 50**; a
full-information greedy **oracle ≈ 149** (≈ max 150). So "the floor" is literally the
random policy, and the achievable routing headroom is ~50→~149.

| arm | comm graph | final return [95% CI] | vs random floor |
|-----|-----------|------------------------|-----------------|
| `mlp` | none | 49.85 [49.33, 50.38] | at floor |
| `gnn_wrong` | wrong matching (same density) | 50.42 [49.81, 51.02] | at floor |
| `gnn_complete` | all-to-all (most comms) | 50.94 [50.16, 51.73] | at floor |
| **`gnn_true`** | **the task matching** | **66.10 [63.56, 68.65]** | **+16** |

The decisive, cleanly-identified comparisons (same repaired-GCN architecture and
parameter count; only the adjacency differs):

| comparison | advantage | Cohen's *d* | Welch p_holm | MWU |
|------------|-----------|-------------|--------------|-----|
| `gnn_true` − `gnn_wrong` | **+15.7** | 5.39 | <1e-4 | non-overlapping |
| `gnn_true` − `gnn_complete` | **+15.2** | 5.11 | <1e-4 | non-overlapping |
| `gnn_true` − `mlp` (a-priori sanity) | +16.3 | 5.62 | <1e-4 | non-overlapping |

`min(gnn_true)=61.5 > max(all rivals)=52.6` — complete distributional separation across
12 seeds. The interpretation rests on `gnn_true` vs `gnn_wrong`/`gnn_complete`: the
arm with the *most* communication (all-to-all) sits at the random floor; only the
task-matched edges rise. The `gnn_true` vs `mlp` cell is the (conceded a-priori)
sanity check, not the headline. **`gnn_true` captures only ~16% of the routing
headroom (66 of 149 vs a 50 floor): routing helps significantly, it does not solve
the task.** (MWU p-values are at their small-sample floor under complete separation —
they indicate non-overlap, not additional effect magnitude.)

Parameter matching: `gnn_wrong` and `gnn_complete` are *exactly* parameter- and
architecture-identical to `gnn_true` (only `adjacency()` differs) — this is what
carries the structure claim. The `mlp` control matches the *plain*-GCN parameter
count; vs the as-run *repaired* GCN it differs by 256 LayerNorm parameters (<2% of
~14k), far too few to explain a *d*≈5 effect, and `gnn_wrong`/`gnn_complete` carry the
same LayerNorm yet floor — so stabilization is not the source of the win.

## 4. Does learned attention over all-to-all recover it? (inconclusive — scoped)

We tested whether single-head (GAT) and multi-head (DGN/G2ANet-style) attention over
the **complete** graph recover `gnn_true`. As run they did not (both ~floor). **We do
not, however, claim "attention cannot substitute for structure," because the attention
arms are not cleanly identified:** they lacked the residual+LayerNorm stabilization the
GCN arms carry, and GAT *given the correct graph* (`gat_true`) also floored (51.45) —
i.e. the attention arms failed to **train**, not demonstrably to **find** structure.
The "structure, not channel" conclusion therefore rests on the clean
`gnn_true` vs `gcn_complete` contrast (both repaired GCN, +15.2, *d*=5.1), **not** on
the attention arms. A matched-stabilization attention re-run (add residual+LayerNorm to
GAT/DGN) is the cheapest open follow-up; until then C3 is reported as inconclusive.

## 5. Effect vs team size N — significant at every N, but non-monotone (H-N not supported)

H-N pre-registered that the advantage *grows* with N. It does **not**: the advantage is
an inverted-U and the standardized/relative effect *attenuates*.

| N | `gnn_true`−floor | Cohen's *d* | fractional lift |
|---|------------------|-------------|-----------------|
| 4 | +11.8 | 3.3 | 35.5% |
| 6 | +16.3 | 5.6 | 32.6% |
| 8 | +16.4 | 3.0 | 24.7% |
| 10 | +11.0 | 1.6 | 13.1% |

It is **significant at every N=4–10** (both tests, Holm), which is real and valuable —
but **non-monotone, with effect size and fractional lift declining past N=6**. We
report H-N (as a monotone dose-response) as **not supported**, and frame N as a
robustness axis, not a dose-response.

## 6. Degree-2 (topology): the effect holds but attenuates sharply

A matching is degree-1 (point-to-point). To test genuine *topology*, `nbr_routing`
(degree-2 ring): each agent must reach the **centroid of its two ring neighbours'
goals**; `gnn_true`=ring, `gnn_wrong`=skip-ring (±2, matched degree),
`gnn_complete`=all-to-all. **Matched protocol, 12 seeds, 30k:**

| arm | return | `gnn_true` − arm | *d* | Welch p_holm | MWU |
|-----|--------|------------------|-----|--------------|-----|
| `gnn_true` | 52.67 | — | — | — | — |
| `gnn_wrong` | 50.07 | +2.59 | 1.68 | 0.0035 | 0.0002 |
| `gnn_complete` | 50.48 | +2.18 | 1.38 | 0.0044 | 0.0017 |
| `mlp` | 50.23 | +2.44 | 1.50 | 0.0042 | 0.0011 |

`gnn_true` significantly beats the wrong topology, all-to-all, and no-comm
(`true > wrong ≈ complete`), so the **structure/topology effect survives at degree-2** —
but it is **~7× smaller** than at degree-1 (+2.4 / ~5% vs +16 / ~33%). A fixed
mean-aggregating GCN exploits a neighbourhood-*set* (centroid) task far less
efficiently than single-partner routing. (Exploratory 3-seed runs showed the same
ordering, weak at 30k and ~+5 at 80k; only the 12-seed/30k matched cell above is
confirmatory.)

## 7. Honest scope and limitations

- **Constructed-task existence/boundary result.** The tasks are built so routing is
  necessary — the right design to *isolate* structure from capacity, convention, and
  observability, and exactly the pre-registered plan (`STRENGTHENING_PLAN.md` §2). It
  **coexists with** the prior negative; it does not show the prior negative was wrong.
- **No out-of-harness replication of the positive yet.** The negative half replicated
  on MPE/LBF/SMAC; the positive half is, so far, one env family (3×3 CoordGrid), N≤10.
  Do not generalize beyond "task-structured routing under partial information."
- **Robust to observability.** Under full obs the effect is *larger* (+22.6, *d*=5.4) —
  the partner's goal is private under any obs mode, so the win is not a partial-obs
  artifact.
- **Reproducible.** The N6 headline is **bit-identical** across two independent runs
  spanning a script refactor — genuine seed determinism, not relabeled copies.
- **Practical rule.** Add a coordination graph, and route along *task-relevant* edges
  (not all-to-all), only when agents must act on information held by specific others
  they cannot observe; pointing the channel at the wrong neighbours is no better than
  no channel.

---

## 8. Proposed paper reframe (for confirmation — not yet applied to `paper/`)

Turn the pure negative into a **boundary paper** (negative endpoint + identified
positive endpoint), a direct answer to the title question.

- **Title (option):** *"When Does Graph Structure Help in Multi-Agent RL? Route Along
  Task Edges, or Not At All — A Controlled Boundary."*
- **Abstract delta:** keep the identified negative (GNN-QMIX ≤ MLP-QMIX in
  full-info/convention-reducible settings; penalty grows with N; replicates on
  MPE/LBF). **Add:** when the task requires routing private information to specific
  partners, the task-matched graph beats a density-matched wrong graph and all-to-all
  at byte-identical params/obs (*d*≈5 at degree-1; significant but ~7× smaller at
  degree-2), and the effect is the *structure*, not the channel. State plainly: H-N
  (grows-with-N) not supported (significant but non-monotone); attention-vs-structure
  inconclusive; positive result is a single-env existence/boundary result.
- **Claims that change:** "graphs never help here" → "graphs help *iff* message passing
  follows the task's coordination edges and the information is otherwise unavailable."
  The H1–H3 negative becomes the full-obs/convention-reducible endpoint of the boundary.
- **Figures:** structure-bar with the random/oracle reference lines (controls
  foregrounded); N-robustness forest plot showing the attenuation; degree-1 vs degree-2
  magnitude.

Full LaTeX integration (abstract, intro, a new results section, related-work on
comm-MARL under partial obs) is a separate, confirm-first step.
