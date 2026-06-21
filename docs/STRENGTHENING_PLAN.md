# Strengthening Plan: from a clean negative result to a positive, novel, practical one

**Status:** pre-registration draft. No experiments run yet. This document is the
*protocol* — hypotheses, environment changes, and decision rules are fixed
**before** any sweep so the positive turn (if it materialises) is credible and
not the product of HARKing or p-hacking.

---

## 0. Integrity guardrails (non-negotiable)

The current paper's main asset is its rigor. We keep it.

1. **No HARKing.** New hypotheses (H4–H6 below) are registered here, before
   running, and are *motivated by a code-level mechanism*, not fitted to results.
2. **No p-hacking / metric shopping / seed selection.** Same protocol as the
   existing paper: pre-fixed metric (final-window mean return), ≥10 seeds for
   decisive cells, Holm-corrected Welch + Mann–Whitney, Cohen's *d*.
3. **The result can still be negative.** If the graph fails to help *even* in the
   regime where it should, we report that. The "where is the boundary?" question
   is publishable either way — a negative-with-boundary is strictly stronger than
   the current pure null.
4. **No claim outpaces evidence.** The scope of every claim is stated explicitly.

---

## 1. Diagnosis (what the existing data already tells us)

The current finding — `GNN-QMIX` never beats the parameter-matched `MLP-QMIX`
control, and is harmful with team size — is **correct but explained by two
confounds that suppress the graph's value**:

- **D1 — Observation redundancy.** In `CoordGrid`, each agent's observation
  already includes its neighbours' coordinates
  (`obs_dim = 2 + 2·max_neighbors + n_agents`, `coord_grid.py:128/254`). The GCN
  aggregates information the agent *already perceives*, so it adds no signal —
  only the cost of message-passing (over-smoothing, optimisation noise).
- **D2 — Over-smoothing cliff.** The pre-registered depth ablation found a sharp
  drop above `L=2`. With redundant inputs, extra hops only wash out the
  per-agent utilities the monotonic mixer needs.

**Implication:** the experiment accidentally removed the graph's job. Graph
message-passing earns its keep only when an agent must rely on *neighbours'*
information that it cannot observe directly — i.e. under **partial
observability**. That is exactly the regime DGN/G2ANet/CommNet operate in, and
it is absent here.

---

## 2. Positive thesis (the reframe)

> **Graph structure helps cooperative MARL exactly when local observability is
> too limited for an agent to perceive its coordination partners — and the
> advantage grows with the graph diameter that information must traverse.**

This is a *constructive boundary* result, not a null. It:
- gives a **positive headline** (a regime where the graph wins, with effect size);
- is **novel** (no prior controlled study maps the observability × diameter
  boundary with a capacity-matched control);
- is **practical** (a deployable rule: *use a graph mixer only when agents cannot
  already observe their coordination partners*);
- **reconciles the literature with our negative result** (prior wins are
  partial-obs comm settings; our null is a full-obs setting).

### Candidate titles
- *Where Message Passing Earns Its Keep: The Observability Boundary of
  Graph-Based Cooperative MARL*
- *Graphs Help Perception, Not Factorisation: When Coordination Graphs Pay Off*
- *When Does Graph Structure Help in MARL? It Depends on What Agents Can See*

---

## 3. Pre-registered hypotheses

- **H4 (observability gates the graph).** Under restricted observability (agents
  see only own state + the coordination graph, not neighbour coordinates),
  `GNN-QMIX` achieves higher final return than both `QMIX` and the
  parameter-matched `MLP-QMIX` on the structured graph. *Falsification:* mean
  advantage ≤ 0 or not significant under Holm-corrected Welch + MWU.
- **H5 (advantage increases with diameter).** Holding observability restricted,
  the `GNN-QMIX − MLP-QMIX` advantage is monotonically increasing in graph
  diameter `d` (ring/line/grid sweep), because information must route farther.
  *Falsification:* non-positive or non-monotone trend over ≥3 diameters.
- **H6 (the over-depth cliff inverts under information need).** Under restricted
  observability with high diameter, the optimal depth `L*` increases with `d`
  (the routing requirement) rather than collapsing to `L≤2`. *Falsification:*
  `L*` still ≤2 regardless of `d` — i.e. over-smoothing dominates even when depth
  is needed; in that case the practical recommendation strengthens (residual GNN
  required, see Stage C).

A **2-D phase diagram** (observation radius × diameter, colour = graph
advantage) is the headline figure: a region that crosses from negative (current
result) to positive (the new regime).

---

## 4. Minimal code changes (most infrastructure already exists)

The GNN-at-encoder path and adjacency plumbing already exist. New work is small:

1. **`CoordGrid` observability knob** (`coord_grid.py`): add
   `obs_mode ∈ {full, ego, radius-k}`.
   - `full` — current behaviour (neighbour coords visible).
   - `ego` — agent sees only own position + id; neighbour slots zeroed. The graph
     is then the *only* channel for neighbour information.
   - `radius-k` — neighbour coords visible only within Manhattan radius `k`.
   Keep `obs_dim` constant (zero-fill hidden slots) so all algorithms share I/O
   and the contrast stays clean.
2. **Sanity tests** (`tests/test_coord_grid.py`): obs masking is correct;
   `obs_dim` unchanged; `MLP-QMIX`/`QMIX` see the same masked obs (fair control).
3. No algorithm changes for H4/H5. Stage C adds anti-over-smoothing variants.

---

## 5. Experiment stages

| Stage | What | Compute | Gate |
|------|------|---------|------|
| **A** | Implement `obs_mode`, tests, smoke run (1 seed) to confirm learning | minutes (CPU) | obs masking correct; agents still learn under `ego` |
| **B1** | H4: 5 algos × {full, ego} × ring × N∈{4,8} × ≥10 seeds × locked budget | overnight CPU | does the graph win under `ego`? |
| **B2** | H5: `ego` × {ring, line, grid (3 diameters)} × N∈{4,8} × ≥10 seeds | overnight CPU | is advantage monotone in `d`? |
| **B3** | H6: depth `L∈{1,2,3,4}` × diameter under `ego` | overnight CPU | does `L*` track `d`? |
| **C** *(if H4 holds weakly or H6 shows over-smoothing)* | anti-over-smoothing GNNs (residual GCN / GCNII / jumping-knowledge) as a *fix* axis | overnight CPU | does a residual GNN recover/extend the win? |
| **D** *(stretch, GPU)* | multi-seed SMAC on a coordination-heavy map (corridor / MMM2) to close the external-validity hole | GPU-bound | replication beyond our harness |

All Stage B/C runs fit the existing single-CPU / overnight budget — the positive
result, if real, is reachable **without new hardware**, because the graph's job
(routing unobserved information) is cheap to create.

---

## 6. Decision rules → outcome → venue

- **Strong positive** (H4 + H5 hold, clean phase diagram, ± Stage C fix):
  headline becomes a positive boundary + practical rule + mechanism. Realistic
  homes broaden to **JMLR, JAIR, a top ML conference (NeurIPS/ICML/ICLR), or
  Neural Networks**. This is the target.
- **Partial positive** (graph wins under `ego` but not monotone in `d`, or only
  with a residual GNN): still a positive, novel "when + how" paper for **JMLR /
  JAIR / TMLR**.
- **Negative even under `ego`** (graph never helps): report it. The
  observability-boundary study + capacity control is a complete, rigorous null →
  **TMLR / JMLR**. We do **not** dress it up as positive.

Venue/format is therefore **downstream of the result**; we defer the LaTeX-class
reformat and the zero-warnings pass until the direction is locked, so that work
is done once, in the right template.

---

## 7. What is NOT changing

The harness, the capacity-matched control, the pre-registration discipline, the
stats protocol, and the existing negative result (now reframed as the `full`-obs
endpoint of the phase diagram) all stay. We are *extending*, not discarding —
the existing rigor is the foundation the positive claim will stand on.
