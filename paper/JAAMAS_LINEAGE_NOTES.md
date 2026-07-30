# JAAMAS lineage engagement notes

Purpose: close the referee's venue-fatal gap. The paper defines and uses
"coordination graph" ~30 times but cites only post-2020 deep-MARL descendants
(Bohmer DCG, CASEC, DICG) and never the ancestors. These notes tell the
academic-writer, per entry, what each work established, where in `main.tex` it
must be engaged, and the precise one-sentence delta separating this paper's
contribution.

All metadata web-verified 2026-07-30 (publisher / DBLP / JMLR / JAIR of record).

## Primary attachment point (read first)

The single most important fix is the **coordination-graph definition paragraph,
`main.tex` lines 532-537** ("We define a coordination graph $\Gtask=(\Nset,
\Egraph)$..."). That sentence currently cites nothing foundational and is exactly
what a lineage-checking referee greps for. The formalism must be attributed there
to the Guestrin-Koller-Parr factored-MDP line (2001-2003) and the Kok-Vlassis
coordination-graph line (2004-2006). Related Work (lines 570-673) is the
*secondary* engagement; if the lineage appears only in Related Work while the
definition stands bare, the named gap is not closed.

Scholarly caution on coinage: do not attribute "introduced the coordination
graph" to a single paper. The formalism is *developed across* the
Guestrin-Koller-Parr line (2001-2003); the explicit "directed graph over agents,
edge A_i -> A_j iff A_i influences Q_j" definition appears in Guestrin,
Venkataraman & Koller (2002). Safe framing: "the coordination-graph formalism
developed in the Guestrin-Koller-Parr line (2001-2003)."

---

## Required entries

### guestrin_2001_factoredmdp
Guestrin, Koller & Parr, "Multiagent Planning with Factored MDPs," NIPS 14 (2001),
pp. 1523-1530.
- Established: in a cooperative multiagent MDP with an additively factored value
  function, coordination and communication are not imposed but *derived* from the
  system dynamics, inducing a message-passing pattern over agents.
- Engage at: definition paragraph 532-537 (origin of the formalism); Related Work
  570-673.
- Delta: they compute the coordination structure analytically from a *known*
  factored value function and coordinate via exact message passing for planning;
  this paper hands a *fixed* graph to a *learned* deep mixer and tests whether that
  structure helps under partial observability and private state, at byte-matched
  capacity and against a wrong-graph control.

### guestrin_2002_coordrl
Guestrin, Lagoudakis & Parr, "Coordinated Reinforcement Learning," ICML 19 (2002),
pp. 227-234.
- Established: coordinated RL -- agents *learn* the parameters of a factored joint
  value function and select joint actions at runtime by variable elimination over
  the coordination graph.
- Engage at: definition paragraph 532-537; value-decomposition Related Work para
  (543-557), as the RL-era root of value factorization.
- Delta: they learn tabular/linear factored payoffs with the graph assumed known
  and correct; this paper isolates the graph's contribution inside a *nonlinear*
  function approximator and asks whether *topology-correctness* (task-matched vs a
  density-matched wrong graph) -- not merely having a graph -- is what pays.

### kok_2004_sparseq
Kok & Vlassis, "Sparse Cooperative Q-learning," ICML 21 (2004), pp. 481-488.
- Established: representing the joint Q only over the *edges* of a coordination
  graph, so sparse interaction structure shrinks the representation and beats
  dense/independent learners on coordination tasks (the tabular "sparse beats
  dense" result).
- Engage at: definition paragraph 532-537; sparse-beats-dense discussion 597-607
  (as the pre-deep precedent behind the CASEC credit already given there).
- Delta: they established sparse-beats-dense for tabular payoff propagation with
  known factored payoffs; this paper isolates the effect inside a deep function
  approximator under partial observability and private state, with byte-matched
  capacity and a wrong-graph control.

### kok_2006_payoffprop
Kok & Vlassis, "Collaborative Multiagent Reinforcement Learning by Payoff
Propagation," JMLR 7 (2006), pp. 1789-1828.
- Established: max-plus / payoff-propagation (belief-propagation-style message
  passing) on the coordination graph computes near-optimal joint actions at scale;
  the canonical journal treatment of coordination-graph action selection.
- Engage at: definition paragraph 532-537; Related Work 570-673.
- Delta: they run exact/approximate *inference* over a given graph with known
  payoffs; this paper neither assumes known payoffs nor performs inference -- it
  tests whether supplying the same graph as an inductive bias to a *learner* yields
  gains, and attributes any gain to which edges are present rather than to the
  message-passing machinery.

### shoham_1995_sociallaws
Shoham & Tennenholtz, "On Social Laws for Artificial Agent Societies: Off-line
Design," Artificial Intelligence 73(1-2) (1995), pp. 231-252.
- Established: coordination can be secured *offline* by designing a shared social
  law / convention that constrains agents, removing the need for runtime
  negotiation or communication.
- Engage at: the "convention-reducible" coinage, lines 380-384 (this is the literal
  ancestor of that term); Related Work 570-673.
- Delta: they *design* the convention offline to obviate coordination; this paper
  borrows the notion to characterize a *task class* ("convention-reducible")
  empirically and shows a coordination graph is provably inert precisely there --
  no convention is designed, a boundary is measured.

### melo_2009_sparseinteractions
Melo & Veloso, "Learning of Coordination: Exploiting Sparse Interactions in
Multiagent Systems," AAMAS 8 (2009), pp. 773-780.
- Established: agents can *learn when and where* they must coordinate, learning both
  individual policies and the (sparse) interaction states, without a pre-given
  interaction model.
- Engage at: sparse-interaction Related Work / 597-607.
- Delta: this is the pre-deep, RL-side ancestor of the sparse-beats-dense result the
  paper already credits to CASEC; the delta is that this paper *fixes* topology and
  runs a density-matched wrong-graph control inside a deep mixer rather than
  *learning* where the interactions are.

### melo_2011_decsimdp
Melo & Veloso, "Decentralized MDPs with Sparse Interactions," Artificial
Intelligence 175(11) (2011), pp. 1757-1789.
- Established: the Dec-MDP-with-sparse-interactions (Dec-SIMDP) formalism -- a
  decision-theoretic model in which agents act independently except in a small set
  of interaction states, with associated planning guarantees.
- Engage at: definition paragraph 532-537 (formal grounding for "only edges
  contribute reward"); Related Work 570-673.
- Delta: they formalize sparse coupling in a *planning* model with known dynamics;
  coord_grid instantiates the same "only edges matter" reward but tests *learned*
  value factorization under partial observability, not planning with a known model.
- (Pairing note: melo_2009 is the RL/learning ancestor, melo_2011 the
  planning/formalism companion. If only one is used, keep melo_2009 -- it is closer
  to this paper's learning setting. Both are verified and included.)

### modi_2005_adopt
Modi, Shen, Tambe & Yokoo, "ADOPT: Asynchronous Distributed Constraint Optimization
with Quality Guarantees," Artificial Intelligence 161(1-2) (2005), pp. 149-180.
- Established: the first asynchronous, *complete* DCOP algorithm with bounded-error
  quality guarantees -- optimal coordination *given* a known constraint graph.
- Engage at: definition paragraph 532-537 (the coordination graph as heir to the
  constraint graph); Related Work 570-673, as the "optimization-given-structure"
  lineage.
- Delta: DCOP solves for the optimal joint assignment given a *fixed, known*
  constraint graph and known payoffs with guarantees; this paper asks the orthogonal
  question -- does handing that structure to a *learner* (no known payoffs, no
  optimality guarantee) change what it learns.
- Author-order check: the paper's order is Modi, Shen, Tambe, Yokoo (some indexers
  reorder Shen-first; the bib entry above is correct).

### fioretto_2018_dcopsurvey
Fioretto, Pontelli & Yeoh, "Distributed Constraint Optimization Problems and
Applications: A Survey," JAIR 61 (2018), pp. 623-698.
- Established: the canonical modern survey positioning DCOP as the framework for
  coordination over a *known* constraint/interaction graph, tying the
  coordination-graph idea to a mature MAS subfield and its applications.
- Engage at: definition paragraph 532-537 / Related Work 570-673, as the one-line
  framing that the coordination graph descends from the constraint-graph / DCOP
  tradition (valuable for the JAAMAS audience specifically).
- Delta: the survey covers optimization *given* structure across many exact and
  approximate solvers; this paper's contribution is empirical *attribution* --
  whether the structure, once handed to a deep learner rather than solved over,
  changes return, isolated from capacity and from the communication channel.

---

## Additional foundational references judged indispensable (max 3)

These are NOT on the referee's explicit list but strengthen the lineage for a
JAAMAS audience. All web-verified; bib entries provided in lineage_additions.bib
under the "OPTIONAL / RECOMMENDED" block.

1. **guestrin_2003_factoredmdp_jair** -- Guestrin, Koller, Parr & Venkataraman,
   "Efficient Solution Algorithms for Factored MDPs," JAIR 19 (2003), pp. 399-468.
   The fullest journal treatment of factored-MDP solution via variable elimination
   over the coordination graph, and the reference Bohmer's DCG cites for the
   concept. Strongest single addition: it is the definitional anchor the current
   532-537 paragraph is missing.

2. **guestrin_2002_contextspecific** -- Guestrin, Venkataraman & Koller,
   "Context-Specific Multiagent Coordination and Planning with Factored MDPs," AAAI
   (2002), pp. 253-259. Contains the explicit "directed graph over agents, edge
   A_i -> A_j iff A_i influences Q_j" definition -- the literal source of the term
   the referee flags. Best cited *at the definition sentence itself* (532-537).

3. **panait_2005_cooperativemarl** -- Panait & Luke, "Cooperative Multi-Agent
   Learning: The State of the Art," JAAMAS 11(3) (2005), pp. 387-434. The canonical
   cooperative-MARL survey, published in the *target venue*; situates the whole
   value-decomposition / coordination line for a JAAMAS readership and signals
   venue-appropriate scholarship.

Recommendation: (1) is close to mandatory for closing the gap; (2) is the cleanest
way to satisfy "who defined the coordination graph" without overstatement; (3) is
strategically apt for JAAMAS but optional.
