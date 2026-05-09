# Phase 1b mixer-capacity diagnostic — summary

**Setup:** seed 0 only, 10,000 env steps, same env config as the Phase 1
pilot (2 agents on a 4×4 grid with distance shaping).

**Question:** does shrinking the mixer's `embed_dim` (32 → 16, 32 → 8)
fix the QMIX/GNN-QMIX learning failure observed in Phase 1?

## Numbers

| algo + embed_dim | peak success | final-window | late q_tot | late loss |
|---|---|---|---|---|
| qmix `emb=32` (Phase 1 baseline) | 0.05 | 0.00 | **+12.44** | **8.81** |
| qmix `emb=16` | 0.10 | 0.05 | +3.22 | 0.57 |
| qmix `emb=8`  | 0.10 | 0.00 | +3.64 | 0.66 |
| gnn_qmix `emb=32` (Phase 1 baseline) | 0.10 | 0.00 | +4.12 | 1.11 |
| gnn_qmix `emb=16` | **0.30** | 0.10 | +3.24 | 0.53 |
| gnn_qmix `emb=8`  | **0.30** | 0.05 | +2.49 | 0.39 |

(For comparison: VDN with the same Q-net, no mixer hypernetwork at all,
reaches peak success 0.55, q_tot +2.21, loss 0.20.)

## Verdict

**Differential.** Shrinking `embed_dim`:

- For vanilla **QMIX**: contains the divergence (`q_tot` dropped from
  +12 to +3) but did **not** restore learning. Peak stayed at ~0.10
  regardless of `embed_dim ∈ {8, 16}`. There's a second issue beyond
  capacity — likely related to mixer initialization, mixer learning
  rate, or the state-conditioned bias path absorbing TD error before
  it can flow back to the Q-net.

- For **GNN-QMIX**: shrinking `embed_dim` actually fixed learning.
  Peak rose from 0.10 (emb=32) to 0.30 (emb=8 or 16) — crossing the
  ≥0.30 threshold for "learned" used in this diagnostic. The GNN's
  mean-aggregation step appears to provide an implicit regularizer
  that vanilla QMIX's pure hypernetwork mixer lacks.

This single-seed result is **suggestive, not conclusive** — Phase 2
needs to verify across seeds.

## Phase 2 implications

The Phase 2 tuning grid was already widened in `docs/PHASE_2_PLAN.md`
to add `mixer_lr` and explicit initialization as axes. This diagnostic
confirms that widening was the right call for QMIX (where `embed_dim`
alone wasn't enough) and shows that for GNN-QMIX, the existing axes
already include a regime that works.

## Caveats

- n=1 seed. Phase 2's 3-seed tuning grid is the actual test.
- 10k step budget is tight; QMIX might learn at this `embed_dim` with
  more steps. Phase 2's locked sweep uses 30k steps.
- We did not vary `mixer_lr` or initialization here — those are
  separate Phase 2 axes.
