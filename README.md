# When Does Graph Structure Help in Multi-Agent Reinforcement Learning? The Right Edges, Not More Edges

Code, data, and analysis artifact for the paper of the same name
(Ali Jabbary, Kasra Ghanavati).

The paper asks a designer's question: in cooperative, value-factorised MARL,
**when is it worth wiring a coordination graph, and what does the wrong wiring
cost?** The answer we measure is *the right edges, not more edges* — and we
establish it by controlled attribution rather than by benchmark score, holding
the encoder, mixer, optimiser, and parameter count fixed while the graph itself
is the experimental variable.

Two isolators carry the argument, and both are implemented here:

- a **no-graph control** — byte-identical apart from the graph module, so a
  measured win cannot be attributed to added capacity;
- a **density-matched wrong-graph control** — the same number of edges pointed
  at the wrong neighbours, so a win cannot be attributed to merely *more*
  connectivity.

## What the study found

- **Positive endpoint.** When an agent must act on private information held by
  one specific unobserved partner, the task-matched graph beats all-to-all
  communication, the wrong graph, and the no-graph control at matched
  parameters (Cohen's *d* ≈ 5 at 30k steps, ≈ 12 at 150k; 12 seeds; complete
  seed-level separation; Holm-corrected).
- **Mechanism-agnostic.** A pre-registered deep coordination graph (DCG,
  max-plus — the classical Guestrin / Kok–Vlassis lineage) reproduces the
  1-hop result and shares its structure-dependence, so the active ingredient is
  the topology, not the message-passing mechanism.
- **Negative endpoint.** Where coordination is full-information or
  convention-reducible, the same architecture confers no benefit over the
  parameter-matched baseline. The commonly used *unstabilised* GCN operator is
  actively harmful there — an optimisation pathology that residual + LayerNorm
  stabilisation removes.
- **Honest negatives, published as such.** On a 2-hop relay neither mechanism
  routes above a no-communication baseline. A pre-registered attempt to
  transfer the positive endpoint onto composed MPE dynamics did **not** carry
  it there: all graph arms diverged below the no-graph control, so the positive
  endpoint is scoped to the constructed archetypes on which it was measured.

## Pre-registrations

Decision rules, contrast families, and budget freezes were committed **before**
the data they govern, in both wall-clock time and git ancestry:

| Pre-registration | Governs | Commit |
|---|---|---|
| `results/phaseC_150k/PREREGISTRATION.md` | 150k matched-budget attention rerun | `9bea642` |
| `results/phaseC_classical/PREREGISTRATION.md` | classical DCG baseline | `d7b8dd3` |
| `results/phaseD_external/PREREGISTRATION.md` | MPE external-validity attempt | `568e759`, amended `b00e26b` |

The Phase-D confirmatory data landed at `b5d76f9`, a strict descendant of both.
`git log` verifies the ordering independently of anything asserted in the paper.

Each results directory also carries an `ANALYSIS.md` applying the frozen rules,
and Phases C-classical and D carry an independent `FINAL_NUMBER_AUDIT.md` that
re-derives every number in the manuscript from the raw manifests.

## Quick start

Requires Python 3.10–3.13. CPU-only by default — no CUDA path is needed, and
full sweeps are designed to run overnight on a commodity machine.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # POSIX
pip install -e ".[dev]"
pytest -q                       # 183 tests
python scripts/smoke.py         # 1k-step harness check across all algorithms
```

**`docs/REPRO.md` lists the exact command for every table and figure in the
paper**, phase by phase. PettingZoo/SuperSuit are optional and kept behind the
`mpe` extra, so the core package installs without them.

## Layout

```
src/gnnmarl/          algorithms, environments, training loop
  algos/              IQL, VDN, QMIX, GNN-QMIX, GAT, DGN, DCG
  envs/               CoordGrid, PairRouting, TokenMatch, MPE adapters
  training/loop.py    the shared, algorithm-agnostic trainer
configs/              YAML run configs
scripts/              sweep launchers, analysis, figure generation
results/              per-phase manifests, summary CSVs, pre-registrations,
                      frozen analyses, number audits
paper/                manuscript sources (main.tex is the scientific master;
                      main_jaamas.tex is assembled from it by _assemble_jaamas.py)
docs/                 phase specs, acceptance criteria, reproduction guide
tests/                unit tests
```

Two design invariants hold throughout: the trainer is algorithm-agnostic (every
algorithm shares one forward/update signature), and the environments expose a
*tunable* coordination graph orthogonal to everything else — which is what makes
the hypotheses testable at all.

## Citation

The manuscript is under review. Please cite the repository until a DOI is
available.

## License

MIT (see `LICENSE`).
