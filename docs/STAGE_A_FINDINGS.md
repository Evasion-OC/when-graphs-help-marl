# Stage A findings: the positive-result hunt (observability + architecture repair)

**Outcome: negative, and robustly so.** We tested the two most promising,
pre-registered routes to a positive result. Neither produced one. The negative
result survives both — which *strengthens* the paper rather than weakening it.

All numbers below are final-window mean return, 3 seeds, 20k env steps
(directional smoke-grade; Stage-B significance would use >=10 seeds).

## Experiment 1 — Observability gating (H4/H5)

Hypothesis: under restricted observability (`ego`: agents see only own
cell + id, so the graph is the only channel to neighbours), GNN-QMIX should
beat the no-graph controls, with the advantage growing with diameter.

| cell | mode | QMIX | MLP-QMIX | GNN-QMIX | A = GNN − max(ctrl) |
|------|------|------|----------|----------|---------------------|
| ring-N4 (d2) | full | 83.4 | 71.3 | 13.9 | −69 |
| ring-N4 (d2) | ego  | 46.3 | 77.8 | 5.5  | −72 |
| line-N6 (d5) | full | 59.5 | 88.1 | 4.8  | −83 |
| line-N6 (d5) | ego  | 33.2 | 48.0 | 5.7  | −42 |

**Verdict: falsified.** No sign flip. Restricting observability did not help the
graph; the no-graph control (MLP-QMIX) actually *thrives* under `ego` (77.8 on
ring), because the task is solvable by a **graph-free rendezvous convention**
from each agent's own position. The apparent narrowing on line-N6 (`A`: −83→−42)
is the controls degrading, not the graph improving (GNN stays ~5).

## Experiment 2 — Architecture repair (was the GNN just broken?)

GNN-QMIX sat near the random floor (~5) everywhere, so Experiment 1 might have
been an unfair test of a non-training model. We added anti-over-smoothing knobs
(`gnn_residual`, `gnn_layernorm`; GCNII-flavoured) and re-tested.

| cell | mode | GNN plain | gnn_L1 | gnn_repair | best control | A_repair |
|------|------|-----------|--------|------------|--------------|----------|
| ring-N4 | full | 13.9 | 14.6 | **37.0** | 83.4 | −46 |
| ring-N4 | ego  | 5.5  | 6.9  | 8.6      | 77.8 | −69 |
| line-N6 | full | 4.8  | 5.2  | 7.0      | 88.1 | −81 |
| line-N6 | ego  | 5.7  | 4.1  | 5.6      | 48.0 | −42 |

**Verdict: the repair is real but insufficient.** On ring-N4 `full` the repaired
GNN climbed off the floor (13.9 → 37.0), confirming over-smoothing was a genuine
factor — but it still loses to the matched no-graph control by ~2×, and stays
floored under `ego` and on the higher-diameter line. **A GNN that trains properly
still does not beat the parameter-matched control.**

## What this means for the paper

The negative result is now **robust to the two biggest reviewer objections**:
1. *"You under-trained / broke the GNN."* — No: with residual + LayerNorm the GNN
   trains markedly better and the verdict is unchanged.
2. *"You never gave the graph an information advantage."* — We did (`ego`); the
   graph still did not benefit, while the no-graph control did.

These become **robustness subsections** of the negative paper. They make the
claim *"graph aggregation does not help cooperative value factorisation, and the
effect is not an artefact of capacity, training budget, observability, or
GNN training pathology"* considerably harder to attack.

## The one remaining *principled* positive avenue (a genuine gamble)

Every result is consistent with a single explanation: **CoordGrid's coordination
is reducible to a global convention**, so no method needs the graph. A different,
well-motivated hypothesis (not a refit of H4) is:

> Graphs help only when coordination is *irreducibly local* — no single global
> action suffices. Operationalise by making each graph edge demand a
> *pair-specific* rendezvous (e.g. meet at a target that depends on both
> endpoints' private states) and/or hiding absolute position so agents must
> navigate by neighbour sensing.

This is a real new experiment with uncertain payoff, not p-hacking. But the
accumulated evidence (graph dominated in every controlled condition so far)
means the base rate for a strong positive is now low. Pursue only as a bounded,
pre-registered probe — and report honestly if it, too, comes back null.

## Experiment 3 — Goal-routing task (H7): INCONCLUSIVE (task not learned)

We built the convention-proof routing task and tested QMIX / MLP-QMIX /
gnn_repair on it (3 seeds, 30k steps).

| cell | chance | QMIX | MLP-QMIX | gnn_repair | A |
|------|--------|------|----------|------------|---|
| ring-N4 (d2) | ~4.0 | 6.1 | 6.4 | 5.5 | −0.9 |
| line-N6 (d5) | ~3.1 | 3.6 | 3.3 | 3.3 | −0.3 |

**Verdict: inconclusive, not a clean negative.** Every arm sits at/near chance —
including QMIX, whose source agent *can see the goal*. The task was simply not
learned by anyone at this budget (rendezvous at an exact random cell is too
sparse), so it does not fairly test the routing hypothesis. To turn it into a
valid test we would need to make it *learnable* (denser reward shaping, a
smaller grid, or a longer budget) so that at minimum the goal-seeing agent
succeeds — only then is "does routing help the others" a meaningful question.

## Experiment 4 — Goal-routing on a *learnable* (3×3) grid: FIRST SIGNAL

Calibration showed a 3×3 grid makes the routing task learnable (QMIX source
rises 11→22 above chance). Re-ran qmix / mlp_qmix / gnn_repair, 3 seeds, 30k.

| cell | diameter | chance | QMIX | MLP-QMIX | gnn_repair | A = gnn − max(ctrl) |
|------|----------|--------|------|----------|------------|---------------------|
| ring-N4-g3 | 2 | 11.1 | 24.6 | **31.3** | 23.1 | **−8.2** |
| line-N6-g3 | 5 | 16.7 | 24.6 | 19.8 | **26.2** | **+1.6** |

**Honest reading: a real but weak signal, NOT a confirmed positive.**
- The line-N6 "win" (+1.6) is **within noise**: gnn_repair 26.2 ± 6.8 vs QMIX
  24.6 ± 4.8 over 3 seeds. Not significant on its own.
- What *is* notable: the graph advantage **increases with diameter**
  (−8.2 at d=2 → +1.6 at d=5), exactly the pre-registered H5 prediction, and at
  d=5 the graph stops losing for the first time in the whole hunt.
- Caveat: ring-N4 vs line-N6 also changes N (4 vs 6), so diameter and team size
  are confounded here. A clean test must vary diameter at fixed N.

## Overall status of the positive-result hunt

Four attempts: observability (clean negative), architecture repair (clean
negative), routing-sparse (inconclusive/unlearnable), routing-learnable (first
non-negative signal: advantage rises with diameter, consistent with H5, but
weak/noisy and N-confounded). No *confirmed* positive yet — but the first
principled reason to run a proper sweep.

**Recommended next step — Stage B (the test that decides it):** a multi-seed
(≥10) diameter sweep at *fixed* N (e.g. graphs giving d ∈ {2,3,4,5,6} at N=6) on
the learnable routing task, pre-registering H5 (graph advantage increases with
diameter and becomes significantly positive at high d). If the trend holds with
significance, that is a genuine, novel, practical positive result — "graphs help
route non-local information, and the benefit grows with coordination distance."
If the d=5 edge evaporates with more seeds, it was noise and we lock in the
strengthened negative paper.
