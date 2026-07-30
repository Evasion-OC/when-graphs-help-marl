# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A controlled empirical study of **when graph structure helps cooperative-MARL
value factorization**, together with the paper it backs. The code
(`src/gnnmarl`), experiment scripts (`run_phase*.sh`, `scripts/`), results,
manuscript (`paper/`), and the TMLR submission package (`submission_tmlr/`)
all live here. The deliverable is the summary-CSV → stats → figures → paper
pipeline, not just trained models.

`docs/PHASES.md` and `docs/STAGE_*.md` hold the phase specs, findings, and
**acceptance criteria** — those are the gates when picking up work, not vibes.

## Design invariants (keep these true)

- The trainer (`gnnmarl.training.loop`) is *algorithm-agnostic*: every
  algorithm shares the same forward-pass / update signature so the trainer can
  swap them. Don't let an algorithm leak special cases into the loop.
- The environments expose a *tunable coordination graph* — the graph is the
  experimental variable, not incidental. This orthogonality is what makes the
  hypotheses testable.
- The per-episode summary CSV schema is load-bearing: the stats and the paper
  read from it.
- Keep PettingZoo/SuperSuit imports lazy (behind the `mpe` extra) so the core
  package installs without them.

## Branch workflow

Work proceeds as a chain of phase branches, each opening (not merging) a PR
against `main`; each phase branches off the previous one, and PRs are left
open so the history is preserved as a sequence of reviewable units.

## Constraints that shape design choices

- **Reproducibility on commodity hardware.** No CUDA-only paths or
  dependencies that break on CPU/MPS; full sweeps must run overnight on a
  local machine.
- **Empirical claims over theoretical ones.** The paper *measures* when graph
  structure helps. Avoid scope creep into proofs/theorems — claims stay
  propositions citing prior work.
- **Statistics are part of the artifact.** Welch's t / Mann-Whitney with Holm
  correction across algo × condition pairs. The summary-CSV → stats → plot
  pipeline is the deliverable.
