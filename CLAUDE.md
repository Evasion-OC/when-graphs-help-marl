# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

This repository is an **empty scaffold** at the time of writing. The package layout
(`src/gnnmarl/{algos,envs,training,utils}`) exists but every `__init__.py` is empty
and no module is implemented. The README's quick-start command
(`python -m gnnmarl.train ...`) is aspirational — there is no `train` entrypoint
yet. Treat `docs/PHASES.md` as the spec for what to build, not as documentation of
what exists.

## Commands

```bash
pip install -e ".[dev]"          # core + dev (pytest, ruff)
pip install -e ".[dev,mpe]"      # add PettingZoo/SuperSuit (only needed from Phase 3)

pytest -q                        # full test suite (per pyproject defaults)
pytest tests/test_foo.py::test_x # single test
ruff check src tests             # lint (line-length 100, target py310)
ruff format src tests            # format
```

There is no Makefile, no CI config, and no run scripts yet — those land in later
phases.

## Architecture & intended layout

The project is structured as a **controlled empirical study** comparing four
cooperative-MARL value-factorization algorithms (IQL, VDN, QMIX, GNN-QMIX) on
tasks with tunable graph structure. The package boundaries are designed so the
training loop is *algorithm-agnostic* and the environment exposes a *tunable
coordination graph* — this orthogonality is what makes the H1–H3 hypotheses
testable.

Intended module responsibilities (from `docs/PHASES.md` Phase 0):

- `gnnmarl.envs.coord_grid` — custom cooperative gridworld whose underlying
  coordination graph is a knob (sparse/dense, controlled diameter, modular vs
  random). The *graph* is the experimental variable, not incidental.
- `gnnmarl.algos.{iql,vdn,qmix,gnn_qmix}` — the four algorithms under test.
  GNN-QMIX is "Algorithm 1 from the source paper, Appendix C" (graph-conditioned
  monotonic mixing). All four must share the same forward-pass / update signature
  so the trainer can swap them.
- `gnnmarl.training.loop` — single generic trainer that consumes any algo + any
  env. Seedable; one row per episode to CSV.
- `gnnmarl.utils.logging` — CSV row-per-episode logger. The summary CSV is what
  the paper reads from (Phase 2 onward), so its schema is load-bearing.

External validity (Phase 3) is via PettingZoo MPE (`simple_spread`, `simple_tag`)
behind the `mpe` extra — keep those imports lazy so the core package installs
without them.

## Phase-based branch workflow

Work proceeds as a chain of branches, each opening (not merging) a PR against
`main`:

```
main → phase-0-harness → phase-1-pilot → phase-2-e1-full
     → phase-3-mpe → phase-4-ablations → phase-5-writeup
```

Each phase branches off the previous one (not off `main`). PRs are left open so
the work history is preserved as a sequence of reviewable units. When picking up
work, check `docs/PHASES.md` for the current phase and its **acceptance** criteria
— those are the gates, not vibes.

## Constraints that shape design choices

- **Reproducibility on commodity hardware.** Target is Apple Silicon, no CUDA.
  Full sweeps must be runnable overnight on a laptop. Don't introduce GPU-only
  paths or dependencies that break on MPS/CPU.
- **Empirical claims over theoretical ones.** The paper this code backs
  *measures* when graph structure helps. Avoid scope creep into proofs/theorems
  — Phase 5 explicitly downgrades "theorems → propositions citing prior work."
- **Statistics are part of the artifact.** Phase 2 specifies Welch's t /
  Mann-Whitney with Holm correction across algo × condition pairs. The
  summary-CSV → stats → plot pipeline is the deliverable, not just trained models.
