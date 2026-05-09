# Phase 2A — tuning sweep summary

**Setup:** 2 agents on a 4×4 grid (Phase 1's pilot env, smallest), 10k env
steps per run, 3 seeds per (algo, hyperparams) cell, distance-shaping on,
`eps_decay_steps=5000`.

**Goal:** find a `(embed_dim, mixer_lr, mixer_init)` configuration per
algorithm that learns this small task reliably, addressing Phase 1's
finding that QMIX and GNN-QMIX with literature-default hyperparameters
fail to learn at this scale.

**Grid:** `embed_dim ∈ {8, 16}`, `mixer_lr ∈ {1e-4, 5e-4}`,
`mixer_init ∈ {default, orthogonal(0.1)}`. 8 cells per algo × 2 algos
× 3 seeds = 48 runs.

## Headline findings

1. **Orthogonal init is the dominant axis.** For both QMIX and GNN-QMIX,
   every orthogonal-init config beats every default-init config on
   peak success rate. Default init never reaches the orthogonal floor.

2. **Smaller mixer LR (1e-4) wins both algorithms.** Among
   orthogonal-init cells, `mixer_lr=1e-4` produces higher final-window
   success than `mixer_lr=5e-4`, consistent with the hypothesis that
   the mixer needs to evolve more slowly than the Q-net so that TD
   error can flow back to the Q-net before the bias path absorbs it.

3. **Embed_dim differs by algorithm.** QMIX prefers `embed_dim=16`,
   GNN-QMIX prefers `embed_dim=8`. The GNN's mean-aggregation
   provides additional implicit regularization that lets it work with
   a smaller mixer; vanilla QMIX needs more capacity.

4. **The locked configs now learn the task.** With
   `mixer_init=orthogonal(0.1) + mixer_lr=1e-4`, QMIX reaches
   final-window success **0.28** (was 0.01 in Phase 1, default
   hyperparameters) and GNN-QMIX reaches **0.34** (was 0.00). Both
   now exceed Phase 1's IQL (0.16) and VDN (0.14) at the same env.

5. **Phase 1b's diagnosis was partially right.** Phase 1b proposed
   that capacity (embed_dim) was the issue. Phase 2A shows capacity is
   *necessary but not sufficient* — orthogonal init does most of the
   work, and a smaller mixer LR provides additional gain.

## Locked hyperparameters (used by Phase 2C)

| algo | embed_dim | mixer_lr | mixer_init | final-window mean ± std |
|---|---|---|---|---|
| qmix | 16 | 1e-4 | orthogonal(0.1) | 0.28 ± 0.13 |
| gnn_qmix | 8 | 1e-4 | orthogonal(0.1) | 0.34 ± 0.08 |
| iql | (defaults — `lr=5e-4`, `hidden=64`) | — | — | 0.16 (from Phase 1) |
| vdn | (defaults — `lr=5e-4`, `hidden=64`) | — | — | 0.14 (from Phase 1) |

Locked picks were chosen by mean **final-window** success (last 25%
of eval checkpoints), with mean peak as tiebreaker. Final-window is
preferred over peak because peak is a single best moment — it can
reflect transient noise and is followed by collapse in some configs.
Final-window averages over many late checkpoints and is the metric
Phase 2C's H1 statistical tests use.

See `tuning_grid.png` for the full visual grid.
See `locked.md` for top-5 rankings per algo.

## What Phase 2A is *not* claiming

- That these hyperparameters generalize to larger envs. Phase 2C
  re-tests them at 3 task scales with 5 seeds and 30k steps.
- That orthogonal init or small mixer_lr is "the right answer" in
  general. They're the right answer for this small env. The wider
  literature uses `embed_dim ≥ 32` for SMAC-scale problems where the
  state dim is much larger.
- That `n=3` seeds is enough to make confident pairwise comparisons.
  The picks here are direction-of-effect calls; Phase 2C runs n=5
  per cell and applies Holm-corrected statistical tests.

## Bug caught and fixed

The first attempt of this sweep crashed at run 4 with
`NotImplementedError: aten::linalg_qr.out is not currently implemented
for the MPS device`. Apple's MPS backend doesn't yet implement QR
decomposition, which `nn.init.orthogonal_` uses internally.

Fix: build mixer modules on CPU, apply orthogonal init there, then
move to the target device. New regression test
(`tests/test_algos.py::test_orthogonal_init_works_on_target_device`)
exercises both QMIX and GNN-QMIX on `pick_device('auto')` to prevent
the bug from regressing.
