# When Does Graph Structure Help in Multi-Agent Reinforcement Learning?

A controlled empirical study of graph-based value factorization (GNN-QMIX) versus
graph-free baselines (IQL, VDN, QMIX) and dense-attention alternatives in cooperative
multi-agent reinforcement learning.

## Why this exists

Graph-based MARL methods are commonly *claimed* to help in cooperative tasks via
relational inductive bias. The conditions under which they actually help are
under-characterized in the literature. This repository accompanies a paper that
**measures** when GNN-MARL helps, when it doesn't, and why — using small,
controlled benchmarks where the underlying coordination graph is tunable.

The study deliberately targets reproducibility on commodity hardware (Apple Silicon,
no CUDA), with full sweeps runnable overnight on a laptop.

## Status

Active development. See `docs/PHASES.md` for the phase plan.

| Phase | Branch | PR | Status |
|---|---|---|---|
| 0 — Experimental harness | `phase-0-harness` | open | complete |
| 1 — Pilot results | `phase-1-pilot` | open | complete |
| 1b — Mixer-capacity diagnostic | `phase-1b-diagnostic` | open | complete |
| 2A — Tuning sweep | `phase-2-tuning` | open | complete |
| 2C — Locked main sweep | `phase-2-main` | open | **complete** |
| 3 — MPE benchmarks (E2, E3) | `phase-3-mpe` | — | not started |
| 4 — Depth/structure ablations | `phase-4-ablations` | — | not started |
| 5 — Paper writeup integration | `phase-5-writeup` | — | not started |

## Phase 1 headline

A 12-run pilot (4 algorithms × 3 seeds × 10k steps on 2 agents / 4×4)
surfaced two findings that re-shape Phase 2:

1. **IQL and VDN both exhibit peak-then-collapse** on this small coop
   coordination task — peak ~0.6–0.75 success around episode 150, then
   monotonic decline to ~0.15 by end of training.
2. **QMIX and GNN-QMIX with literature-default `embed_dim=32` fail to
   learn at all** — q_tot diverges (+12 vs VDN's +2), suggesting the
   monotonic mixer is over-parameterized for this 8-dim global state.

See `docs/PHASE_1_NOTES.md` and `results/pilot/SUMMARY.md` for the full
breakdown.

## Quick start

Requires Python ≥3.10, `torch`, `torch_geometric`. See `pyproject.toml` for
the full dependency list.

```bash
pip install -e ".[dev]"
python -m gnnmarl.train --algo gnn_qmix --env coord_grid --n-agents 4 --seeds 5
```

## Layout

```
src/gnnmarl/          # algorithms, environments, training loop
configs/              # YAML run configs
scripts/              # sweep launchers + plotting
tests/                # unit tests
results/              # generated CSVs / plots (gitignored binaries)
docs/                 # phase plan, design notes
```

## License

MIT (see `LICENSE`).
