# Stage B — the decisive diameter sweep (run this on your PC)

Pre-registered test of whether the Stage-A signal is real: does graph
message-passing help *route* a privately-held goal, and does the benefit grow
with coordination distance? Team size is held **fixed at N=6**; only the graph's
diameter changes, removing the N-vs-diameter confound in the Stage-A result.

## Pre-registration (fixed before running)

- **Task:** learnable 3×3 `goal_routing` CoordGrid (a secret goal is seen only by
  the source agent; reward = agents at the goal; non-source agents must have it
  routed over the graph).
- **Cells (N=6, grid 3×3):** `complete` (d=1), `ring` (d=3), `line` (d=5).
- **Arms:** `qmix`, `mlp_qmix` (parameter-matched no-graph control),
  `gnn_repair_L2` (repaired GCN, depth 2), `gnn_repair_Ld` (repaired GCN,
  depth = diameter — the repair is what keeps a deep router trainable).
- **Seeds:** ≥10. **Budget:** 30k env steps.
- **Primary hypothesis H5:** `A(d) = mean(gnn_repair_Ld) − max(mean(qmix),
  mean(mlp_qmix))` **increases with d** and is **significantly > 0 at d=5**
  (Welch t Holm-corrected within each diameter family, AND Mann–Whitney U,
  both two-sided, p < 0.05).
- **Secondary:** matched-depth routing beats fixed-shallow
  (`gnn_repair_Ld` > `gnn_repair_L2`) at high diameter.
- **Decision:** H5 supported → genuine positive result, pursue the positive
  paper. H5 not supported → the Stage-A signal was noise; lock in the
  strengthened negative paper. Report honestly either way.

## How to run (on your desktop)

```bash
# 1. set up once
pip install -e ".[dev]"

# 2. run the sweep in parallel (≈120 runs). Set --workers near your core count.
#    GPU does NOT help here (models are tiny); keep the default --device cpu.
python scripts/phaseB_diameter_sweep.py --seeds 10 --workers 16

# 3. analyse: prints the H5 verdict, writes results/phaseB/{summary,h5_verdict}.csv
python scripts/phaseB_analysis.py
```

`--dry-run` lists the jobs without training, to sanity-check the plan first.

## Performance notes

- The launcher pins each worker to **one CPU thread** and fans the independent
  runs across `--workers` processes, so wall-clock ≈ (total runs / workers) ×
  per-run time. On a 16–24 core Raptor Lake desktop the whole sweep is well
  under an hour.
- **Skip the GPU.** These networks are tiny (64-dim, ≤5 graph layers, batch 32);
  per-op kernel-launch + transfer overhead makes a 4070 Ti the same or slower
  than CPU. `--device cpu` is intentional.
- 64 GB RAM is far more than enough (each run's replay buffer is a few MB).

## What lands in the repo

Results under `results/phaseB/` are git-ignored except the two small summary
CSVs you may choose to commit (`summary.csv`, `h5_verdict.csv`) if Stage B
becomes paper material.
