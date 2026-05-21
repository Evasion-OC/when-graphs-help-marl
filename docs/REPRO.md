# Reproducing the paper

This document lists the exact commands used to produce every table and figure
in the paper. Each phase corresponds to a section of the manuscript and a
subdirectory under `results/`.

## Environment

- Python 3.10–3.13. Tested on Windows Server 2019 + Python 3.13 with
  `torch==2.6.0+cpu` (the 2.12 wheel shipped with Python 3.13 had a broken
  `c10.dll`; pin to 2.6 if you hit DLL init errors).
- Apple Silicon (M-series) parity: the harness is CPU-only by default.
  Setting `device=torch.device("mps")` works for CoordGrid but is not
  required and was not used for the published results.
- All seeds are fixed in code; results are bit-identical across runs given
  the same seed and PyTorch version.

```bash
python -m venv .venv
.venv/Scripts/activate         # Windows
# or: source .venv/bin/activate # POSIX
pip install -e ".[dev]"
pytest -q                      # 70+ tests should all pass
```

## Phase 0 — Harness validation

```bash
python scripts/smoke.py
```

Trains each of {IQL, VDN, QMIX, GNN-QMIX} for 1k env steps on CoordGrid
(N=4, ring) and verifies a well-formed CSV lands under `results/smoke/`.
This is the Phase-0 acceptance gate per `docs/PHASES.md`.

## Phase 1 — Pilot (Section 6.1 / Figure ~1)

```bash
PYTHONIOENCODING=utf-8 python scripts/phase1_pilot.py
PYTHONIOENCODING=utf-8 python scripts/phase1_analysis.py
```

Outputs under `results/phase1/`:
- `iql__ring__N4__seed{0,1,2}/episodes.csv` (and analogues for vdn/qmix/gnn_qmix)
- `summary.csv` — per (algo, seed) final-window mean + episodes-to-80%
- `summary_aggregate.csv` — per-algo mean + 95% t-CI
- `pairwise.csv` — Welch's t with Holm correction
- `figures/learning_curves.{pdf,png}`

## Phase 2 — E1 sweep / H1 + H2 (Section 6.2 / Figure ~2)

```bash
PYTHONIOENCODING=utf-8 python scripts/phase2_e1_sweep.py
PYTHONIOENCODING=utf-8 python scripts/phase2_analysis.py
```

Outputs under `results/phase2/`:
- per-condition learning curves
- `forest_ep80.{pdf,png}` — forest plot of episodes-to-80% per (algo, condition)
- `h1_h2_summary.csv` — GNN-QMIX vs QMIX gap per condition + p-values
- `pairwise_N{4,8}_{ring,erdos_renyi}.csv` — Welch + Holm per condition

Pre-registered budget was `N ∈ {4, 8, 16} × 5 seeds × 200k env steps × 2
graphs × 4 algos = ~24M env steps`. Executed at the reduced budget noted in
`scripts/phase2_e1_sweep.py` (defaults: N ∈ {4, 8}, 3 seeds, 30k steps);
override via `--steps`, `--seeds`, etc. Compute budget table in Appendix C.

## Phase 3 — MPE (Section 6.3, deferred)

The PettingZoo adapter at `src/gnnmarl/envs/mpe_adapter.py` wraps
`simple_spread_v3` and `simple_tag_v3` as `MultiAgentEnv`. Install the
extra and run:

```bash
pip install -e ".[mpe]"
python -m gnnmarl --env "mpe:simple_spread" --env-kwargs.n-agents 3 \
    --algo gnn_qmix --total-env-steps 100000 --seed 0
```

The current release does not include full MPE sweep results; they are
deferred to future work pending GPU compute.

## Phase 4 — Depth × diameter ablation / H3 (Section 6.4 / Figure ~3)

```bash
PYTHONIOENCODING=utf-8 python scripts/phase4_ablation.py
PYTHONIOENCODING=utf-8 python scripts/phase4_analysis.py
```

Outputs under `results/phase4/`:
- `summary_grid.csv` — per (diameter, depth, seed) final return
- `h3_test.csv` — per diameter, is argmax-depth at L≈d?
- `figures/heatmap_final_return.{pdf,png}`
- `figures/depth_diameter_curves.{pdf,png}`

## Building the paper

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

LaTeX deps: `natbib`, `algorithm`/`algpseudocode`, `booktabs`, `microtype`,
`hyperref`. Any recent TeXLive ships these.
