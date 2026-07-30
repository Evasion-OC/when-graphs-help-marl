# JAAMAS Information Sheet (regular paper)

*To accompany the submission of* **When Does Graph Structure Help in Multi-Agent
Reinforcement Learning? A Controlled Empirical Study** *(Jabbary & Ghanavati).*
*Draft — paste into the submission form / upload as PDF after author review.*

## 1. What is the main claim of the paper? Why is this an important contribution to the autonomous agents and multi-agent systems literature?

The main claim is a **design rule with measured boundaries**: adding a coordination
graph to value-factorised cooperative MARL helps **only when the graph's edges match
the task's information dependencies** — the right edges, not more edges — and this is
a property of the *topology*, not of any one coordination mechanism. Concretely:
(i) when an agent must act on private information held by one specific, unobserved
partner, a task-matched graph decisively outperforms all-to-all communication, a
density-matched wrong graph, and a parameter-matched no-graph control (Cohen's
d ≈ 5, complete seed-level separation); (ii) the same result is reproduced by a deep
coordination graph in the classical Guestrin/Kok–Vlassis lineage (DCG with max-plus
action selection), which likewise collapses to floor on wrong or dense graphs — so
the rule generalises across mechanism families; (iii) when coordination is
convention-reducible or fully observed, the graph confers no benefit at matched
capacity, and the commonly used unstabilised GCN operator is actively harmful — an
optimisation pathology that stabilisation removes.

This matters to the AAMAS community because coordination graphs originate here, yet
modern graph-MARL papers rarely separate the contribution of *structure* from the
capacity and communication machinery that carries it. The paper supplies the missing
controlled attribution — two isolators (byte-identical no-graph control;
density-matched wrong-graph control) — and reconciles the field's conflicting
positive and negative reports via the task's information structure.

## 2. What is the evidence you provide to support your claim? Be precise.

Controlled experiments on two constructed archetypes (pair-routing: distributed task
allocation under partial observability with private goals; TokenMatch: referential
signalling) plus MPE, with fixed encoder, mixer, optimiser, and parameter count
across arms:

- **Positive endpoint:** task-matched graph vs all-to-all / wrong-graph /
  no-graph at matched parameters and observations: d ≈ 5 at 30k steps, growing to
  d ≈ 12 at 150k (12 seeds, Welch + Holm and Mann–Whitney, p_holm < 1e-4, complete
  seed-level separation). Replicated out-of-harness on TokenMatch (~96% of maximum
  vs rivals far below). Learned attention (GAT/DGN) recovers the signal only on the
  right graph, at both tested budgets.
- **Classical baseline (pre-registered, timestamp-verifiable):** a steelman deep
  coordination graph (DCG, max-plus) *matches* the 1-hop positive on the true graph
  (n.s. vs GNN-QMIX) and collapses to floor on wrong/dense graphs (d > 12,
  p_holm < 1e-10) — establishing mechanism-agnostic structure-dependence. On a
  2-hop relay task, neither mechanism routes above a no-communication baseline at
  150k steps; reported in full as a pre-registered negative.
- **Negative endpoint:** at full information the graph confers no benefit over the
  byte-identical no-graph control; the unstabilised operator's penalty grows with
  team size and recurs on MPE simple_spread (directionally on LBF); residual +
  LayerNorm stabilisation restores parity on both tested topologies.
- **Protocol:** n = 10–12 seeds per confirmatory cell, Holm-corrected Welch t and
  Mann–Whitney across the pre-declared contrast family, effect sizes and CIs
  throughout; two pre-registrations are timestamp-verifiable in the public
  repository's git history; all code, configurations, per-run manifests, and
  analysis scripts are released.

## 3. What papers by other authors make the most closely related contributions, and how is your paper related to them?

- **Kok & Vlassis, JMLR 7:1789–1828 (2006)** (and Sparse Cooperative Q-learning,
  ICML 2004): established sparse-beats-dense for tabular payoff propagation over a
  known coordination graph. We test this lineage directly via its neural descendant
  and show the sparse-structure rule survives deep function approximation under
  partial observability and private state — while adding the capacity-matched and
  wrong-graph controls that the classical results did not run.
- **Guestrin, Koller & Parr (NIPS 2001; ICML 2002); Guestrin et al., JAIR
  19:399–468 (2003)**: origin of the coordination-graph formalism. Our environment's
  edge-factored reward is a pairwise-factored payoff in exactly this sense; our
  contribution is empirical attribution of the structure's value inside deep
  value-factorisation.
- **Böhmer, Kurin & Whiteson, ICML 2020 (Deep Coordination Graphs)**: the modern
  deep CG; we run it as our pre-registered classical baseline.
- **Rashid et al. (QMIX, ICML 2018/JMLR 2020) and the value-decomposition line
  (VDN, AAMAS 2018)**: the factorisation backbone our harness holds fixed.
- **Agarwal et al., NeurIPS 2021 (statistical precipice)**: the evaluation-rigor
  protocol our statistics follow.
- **Panait & Luke, JAAMAS 11(3):387–434 (2005)**: the cooperative-MAL survey whose
  framing of coordination our boundary results refine.

## 4. Have you published parts of your paper before, for instance in a conference? If so, give details.

No part of this paper has been published in any archival venue, conference, or
workshop, and no part is under review elsewhere. An earlier version was submitted to
and rejected by TMLR (desk) and JAIR (desk, without review); neither constitutes a
publication, and the manuscript has been substantially revised and extended since
(new capacity-matched and wrong-graph controls, the 150k robustness study, and the
pre-registered classical coordination-graph baseline are all new). There is no
arXiv preprint at submission time.
