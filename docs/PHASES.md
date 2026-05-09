# Phase plan

This document tracks the empirical study from harness build to paper-ready results.
Each phase ends in a pull request opened against `main` (not merged) so the work
history is preserved as a sequence of reviewable units.

## Phase 0 — Experimental harness

**Branch:** `phase-0-harness`

Build the bare minimum to run any algorithm on any environment, with seedable
training and CSV logging. No results yet, just the rails.

Deliverables:
- `src/gnnmarl/envs/coord_grid.py` — custom cooperative gridworld with
  *tunable graph structure* (sparse/dense, controllable diameter, modular vs
  random connectivity).
- `src/gnnmarl/algos/iql.py` — independent Q-learning baseline.
- `src/gnnmarl/algos/vdn.py` — value decomposition (linear sum).
- `src/gnnmarl/algos/qmix.py` — monotonic mixing.
- `src/gnnmarl/algos/gnn_qmix.py` — graph-conditioned mixing
  (Algorithm 1 from the source paper, Appendix C).
- `src/gnnmarl/training/loop.py` — generic trainer.
- `src/gnnmarl/utils/logging.py` — CSV row-per-episode logger.
- `tests/` — unit tests on env determinism + algorithm forward-pass shapes.
- `configs/coord_grid_smoke.yaml` — 60-second smoke-test config.

Acceptance: `pytest -q` green, smoke config trains all four algorithms for
1k steps without crashing, logs land in `results/`.

## Phase 1 — Pilot results

**Branch:** `phase-1-pilot` (off `phase-0-harness`)

Tiny end-to-end study to sanity-check that the four algorithms differentiate
on a task with obvious coordination demand.

- E1 (CoordGrid), 4 agents, 3 seeds, 50k env steps each.
- Plot learning curves (mean ± SD across seeds).
- Compute episodes-to-80%-of-best for each algorithm.

Acceptance: VDN and QMIX both clearly beat IQL (sanity for the harness);
GNN-QMIX runs to completion. Results may or may not show GNN-QMIX winning —
this is a sanity gate, not a hypothesis test.

## Phase 1b — Mixer-capacity diagnostic

**Branch:** `phase-1b-diagnostic` (off `phase-1-pilot`)

Phase 1 found that QMIX and GNN-QMIX with `embed_dim=32` fail to learn
on this 8-dim global state. Phase 1b is a 4-run diagnostic on seed 0
that shrinks the mixer's `embed_dim` to {8, 16} and asks: does that
fix the divergence?

This is cheap insurance (~10 min wall-clock) before committing to
Phase 2's larger tuning grid. Three possible outcomes:

- `embed_dim=8` learns: hypothesis confirmed; Phase 2 tunes around 8.
- `embed_dim=16` learns but 8 doesn't: sweet spot somewhere between
  8 and 32; Phase 2's grid is the right shape.
- Neither learns: hypothesis wrong; the issue is elsewhere
  (initialization, LR schedule, target-update period). Phase 2
  needs to be redesigned with a wider diagnostic first.

Acceptance: every diagnostic run completes; `check_diagnostic.py`
prints a verdict. The verdict informs Phase 2 — it doesn't gate
shipping Phase 1b.

## Phase 2 — Full E1 sweep (H1, H2)

**Branch:** `phase-2-e1-full` (off `phase-1-pilot`)

The first real test of H1.

- E1, agents ∈ {4, 8, 16}, 5 seeds, 200k steps.
- Two graph regimes: **structured** (lattice + nearest-neighbor edges; small
  diameter relative to N) vs **random** (Erdős–Rényi at matched edge density).
- Algorithms: IQL, VDN, QMIX, GNN-QMIX.

Outputs:
- Forest plot of episodes-to-80%, per algorithm × condition.
- Statistical test (Welch's t / Mann-Whitney) per pair, with Holm correction.
- A single summary CSV the paper reads from.

## Phase 3 — MPE (E2, E3)

**Branch:** `phase-3-mpe` (off `phase-2-e1-full`)

External-validity replication on canonical PettingZoo benchmarks.

- E2: simple_spread, agents ∈ {3, 6}.
- E3: simple_tag (3 predators, 1 prey).
- Same algorithm set, 5 seeds, ≥ 1M env steps per run.

## Phase 4 — Depth / structure ablations (H3)

**Branch:** `phase-4-ablations` (off `phase-3-mpe`)

Causal probe of *why* GNN helps when it does.

- Vary GNN depth L ∈ {1, 2, 3, 4, 6}.
- Vary task graph diameter d (in CoordGrid).
- Predict and test the inverted-U: GNN-QMIX peaks when L ≈ d.

## Phase 5 — Paper writeup integration

**Branch:** `phase-5-writeup` (off `phase-4-ablations`)

Rewrite of `paper/main.tex`:
- Drop fabricated meta-analysis. Replace with empirical figures from `results/`.
- Trim survey from 85 → ~30 verified citations.
- Reframe abstract & contributions around the empirical findings.
- Theorems → propositions (citing prior work).
- Strip LLM-stylistic prose.

Outputs: TMLR-ready PDF + supplementary materials zip.
