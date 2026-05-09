# Phase 1 pilot — summary

**Task:** 2 agents on a 4×4 grid, max 20 steps/episode, distance-shaping enabled.
**Budget:** 10,000 env steps per run, 3 seeds per algorithm.
**Eval:** 20 greedy episodes every 50 training episodes.

## Headline metrics

| Algorithm | Peak success (mean across seeds) | Final-window success (mean ± SD) | Post-peak drop (peak→final) | Episodes to 70% success |
|---|---|---|---|---|
| iql | 0.750 | 0.16 ± 0.14 | 0.572 | — |
| vdn | 0.567 | 0.14 ± 0.09 | 0.422 | — |
| qmix | 0.033 | 0.01 ± 0.02 | 0.033 | — |
| gnn_qmix | 0.050 | 0.00 ± 0.00 | 0.050 | — |

## Interpretation

- **Peak vs. final**: A large drop (peak ≫ final) means the algorithm
  found a good policy and then collapsed. This is the classic IQL
  cooperative-MARL instability — independent learners chase a moving
  target as their teammates' policies update.
- **Episodes to 70%**: how quickly an algorithm first reaches 70% eval
  success rate (averaged across seeds). Sample-efficiency proxy.
- **Final-window**: mean over the last 25% of eval checkpoints; the
  number we'd report if we trusted late-training behaviour. Pairs with
  peak to expose collapse.
