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
