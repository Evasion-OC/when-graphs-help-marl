# Phase 2 — full E1 sweep (plan)

## Inputs from Phase 1

The pilot surfaced two phenomena that shape Phase 2's design:

1. **IQL transient learning followed by collapse.** Independent Q-learners
   peak around 70–80% success in the early-mid phase of training, then
   collapse to near-zero on this 2-agent task. This is consistent with
   the well-known instability of independent learning in cooperative
   MARL (other agents' policies become non-stationary from each
   learner's perspective).

2. **QMIX / GNN-QMIX fail to learn at all** with literature-default
   `embed_dim=32` on the 8-dimensional global state. The mixer's
   state-conditioned bias diverges (q_tot ~ +12 vs VDN's +2 over the
   same budget) — classic value over-estimation. Hypothesis:
   `embed_dim=32` is too large relative to `state_dim=8`. Phase 2
   tests this directly.

## What Phase 2 must do

### 2A. Per-algorithm hyperparameter tuning (small, principled grid)

Same per-agent Q-net for all algorithms (parameter-matched control),
but mixer capacity per algorithm. Grid:

| algo | hidden | embed_dim | gnn_layers | lr |
|---|---|---|---|---|
| iql | 64 | — | — | {3e-4, 5e-4} |
| vdn | 64 | — | — | {3e-4, 5e-4} |
| qmix | 64 | {8, 16, 32} | — | {3e-4, 5e-4} |
| gnn_qmix | 64 | {8, 16, 32} | {1, 2, 3} | {3e-4, 5e-4} |

Pick the best (algo, hyperparameters) per algorithm by mean final-window
success rate over 3 seeds. Then **lock those hyperparameters** for the
full sweep — no further tuning.

### 2B. The actual sweep

For the locked hyperparameters per algorithm:

- 5 seeds (Phase 1 used 3; underpowered for stat tests)
- 3 task scales: small (2 ag / 4×4), medium (3 ag / 5×5), large (4 ag / 6×6)
- 3 graph regimes: full, ring, random Erdős–Rényi at p=0.5
- 30k env steps per run

Total runs: 4 algos × 5 seeds × 3 scales × 3 graphs = **180 runs**.

Wall-clock estimate on M1 Pro / MPS: ~150s avg per run (heaviest is
gnn_qmix at the largest scale, ~250s). Total: ~6.5 hours. Will run
overnight.

### 2C. Statistical comparisons

For each (scale, graph) cell:

- Per-seed peak and final-window success rate (5 seeds → 5 datapoints)
- Welch t-test for each pair, then Holm correction across the three
  pairwise contrasts that matter: VDN vs QMIX, QMIX vs GNN-QMIX, and
  IQL vs (VDN, QMIX, GNN-QMIX) collapsed.
- Effect sizes (Cohen's d).

n=5 is still modest for parametric tests — we'll also report
non-parametric Mann-Whitney U as a sanity check.

### 2D. The H1 result

Hypothesis 1 (paper version): "GNN-QMIX outperforms QMIX in tasks with
graph structure aligned to the GNN's receptive field; matches QMIX
otherwise."

Phase 2 contributes the **'matches QMIX otherwise' half** by holding
graph structure constant (full graph) and comparing GNN-QMIX vs QMIX
across task scales. The 'graph helps' half waits for Phase 4's
ablation (where graph structure varies on a fixed task).

## What Phase 2 won't do

- No PettingZoo / MPE — that's Phase 3.
- No depth-vs-diameter ablation — that's Phase 4.
- No paper LaTeX writing — that's Phase 5.

## Reproducibility checklist for Phase 2 outputs

- [ ] Tune-stage results CSV with all (algo, hyperparams) grid points
- [ ] Locked-hyperparameters config in `configs/phase_2_locked.yaml`
- [ ] Full sweep CSVs in `results/phase_2/`
- [ ] Per-cell forest plots
- [ ] Statistical comparison table (CSV + markdown)
- [ ] `docs/PHASE_2_NOTES.md` documenting any deviations from this plan
