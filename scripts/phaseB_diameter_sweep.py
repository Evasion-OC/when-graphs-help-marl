"""Stage B -- the decisive diameter sweep (parallel; pre-registers H5).

Tests whether graph message-passing helps *route* a privately-held goal, and
whether the benefit grows with coordination distance, on the learnable 3x3
goal-routing task. Team size is held FIXED at N=6; only the coordination graph's
diameter changes, so diameter is no longer confounded with N (the confound in
the Stage-A signal).

Cells (fixed N=6, grid 3x3, goal_routing):
  complete -> diameter 1
  ring     -> diameter 3
  line     -> diameter 5

Arms:
  qmix            -- no-graph baseline
  mlp_qmix        -- parameter-matched no-graph control (isolates the graph)
  gnn_repair_L2   -- repaired GCN (residual+layernorm), fixed depth 2
  gnn_repair_Ld   -- repaired GCN, depth = diameter (can route across the whole
                     graph; the repair is what keeps a deep stack trainable)

Pre-registered H5: A(d) = mean(gnn_repair_Ld) - max(mean(qmix), mean(mlp_qmix))
increases with diameter d and is significantly > 0 at the largest d (Welch +
Mann-Whitney, two-sided, Holm-corrected). Analysis: scripts/phaseB_analysis.py.

This launcher is PARALLEL: it fans the independent runs across CPU cores, one
torch thread per process. These models are tiny, so CPU + many processes is far
faster than a GPU (which adds kernel-launch/transfer overhead) -- keep
``--device cpu``. On a 16-24 core desktop the full sweep runs in well under an
hour.

Usage (from repo root)::

    python scripts/phaseB_diameter_sweep.py --seeds 10 --workers 16
    python scripts/phaseB_diameter_sweep.py --dry-run      # list jobs, no training
"""

from __future__ import annotations

import os

# Pin BLAS / OpenMP to a single thread BEFORE torch is imported, so that N
# worker processes use N cores cleanly instead of oversubscribing. (Spawned
# children re-execute this module, so the cap applies to them too.)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# (graph, diameter, label)
CELLS = (
    ("complete", 1, "d1_complete"),
    ("ring", 3, "d3_ring"),
    ("line", 5, "d5_line"),
)
N_AGENTS = 6
GRID = 3
EPISODE_STEPS = 25


def arms_for(diameter: int) -> tuple[tuple[str, str, dict], ...]:
    """(arm_name, algo, extra algo_kwargs) for a given diameter."""
    repaired = {"gnn_residual": True, "gnn_layernorm": True}
    return (
        ("qmix", "qmix", {}),
        ("mlp_qmix", "mlp_qmix", {"gnn_layers": 2}),
        ("gnn_repair_L2", "gnn_qmix", {"gnn_layers": 2, **repaired}),
        ("gnn_repair_Ld", "gnn_qmix", {"gnn_layers": max(1, diameter), **repaired}),
    )


def _run_one(job: dict) -> dict:
    """Worker: train one run (single-threaded) and return its final-window mean."""
    import torch

    torch.set_num_threads(1)
    from gnnmarl.training import TrainConfig, train

    steps = job["steps"]
    cfg = TrainConfig(
        env="coord_grid",
        env_kwargs={
            "n_agents": N_AGENTS,
            "grid_size": GRID,
            "episode_steps": EPISODE_STEPS,
            "graph": job["graph"],
            "goal_routing": True,
        },
        algo=job["algo"],
        algo_kwargs={
            "buffer_capacity": 50_000,
            "batch_size": 32,
            "target_update_interval": 200,
            "lr": 5e-4,
            "gamma": 0.99,
            "gnn_hidden": 64,
            **job["extra"],
        },
        total_env_steps=steps,
        eps_start=1.0,
        eps_end=0.05,
        eps_anneal_steps=int(steps * 0.8),
        seed=job["seed"],
        device=job["device"],
        log_dir=Path(job["log_dir"]),
        run_id=job["run_id"],
    )
    run_dir = train(cfg)
    ret = pd.read_csv(Path(run_dir) / "episodes.csv")["return"].to_numpy(dtype=float)
    k = max(1, int(len(ret) * 0.2))
    return {
        "graph": job["graph"],
        "diameter": job["diameter"],
        "arm": job["arm"],
        "algo": job["algo"],
        "seed": job["seed"],
        "run_dir": str(run_dir),
        "final": float(ret[-k:].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel processes; set near your physical core count")
    ap.add_argument("--device", type=str, default="cpu",
                    help="cpu is recommended -- these models are too small for a GPU to help")
    ap.add_argument("--log-dir", type=Path, default=REPO_ROOT / "results" / "phaseB")
    ap.add_argument("--dry-run", action="store_true", help="list jobs and exit")
    args = ap.parse_args()

    jobs: list[dict] = []
    for graph, diameter, label in CELLS:
        for arm, algo, extra in arms_for(diameter):
            for seed in range(args.seeds):
                jobs.append({
                    "graph": graph,
                    "diameter": diameter,
                    "arm": arm,
                    "algo": algo,
                    "extra": extra,
                    "seed": seed,
                    "steps": args.steps,
                    "device": args.device,
                    "log_dir": str(args.log_dir / label),
                    "run_id": f"{arm}__{graph}__d{diameter}__seed{seed}",
                })

    print(f"[phaseB] {len(jobs)} runs x {args.steps} steps | N={N_AGENTS} grid={GRID} "
          f"| workers={args.workers} device={args.device}", flush=True)
    if args.dry_run:
        for j in jobs:
            print(f"  {j['run_id']:<40} algo={j['algo']:<9} layers={j['extra'].get('gnn_layers', '-')}")
        print(f"[phaseB] dry run: {len(jobs)} jobs (no training).")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run_one, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            done += 1
            print(f"  [{done}/{len(jobs)}] {r['arm']:<14} d{r['diameter']} "
                  f"seed{r['seed']}: final={r['final']:.2f} "
                  f"({time.monotonic() - t0:.0f}s)", flush=True)

    manifest = args.log_dir / "manifest.csv"
    pd.DataFrame(rows).sort_values(["diameter", "arm", "seed"]).to_csv(manifest, index=False)
    print(f"\n[phaseB] complete in {time.monotonic() - t0:.0f}s -> {manifest}")
    print("[phaseB] now run: python scripts/phaseB_analysis.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
