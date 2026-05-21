"""Phase 1 — pilot sweep on CoordGrid.

Trains each of {IQL, VDN, QMIX, GNN-QMIX} for 3 seeds × 50k env steps on
CoordGrid (N=4, ring). Writes per-run CSVs under ``results/phase1/``.

Usage::

    python scripts/phase1_pilot.py
    python scripts/phase1_pilot.py --steps 20000 --seeds 2   # smaller smoke

The sister script ``scripts/phase1_analysis.py`` consumes the resulting CSVs
and produces the figures + summary table.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnmarl.training import TrainConfig, train  # noqa: E402


ALGOS = ("iql", "vdn", "qmix", "gnn_qmix")
DEFAULT_SEEDS = (0, 1, 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--episode-steps", type=int, default=25)
    parser.add_argument("--graph", type=str, default="ring")
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "results" / "phase1")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    args.log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    n_jobs = len(ALGOS) * len(seeds)
    print(f"[phase1] {n_jobs} runs x {args.steps} env steps "
          f"-> {args.log_dir}", flush=True)

    for i, algo in enumerate(ALGOS):
        for j, seed in enumerate(seeds):
            tag = f"[{i*len(seeds)+j+1}/{n_jobs}] {algo} seed={seed}"
            run_t0 = time.monotonic()
            cfg = TrainConfig(
                env="coord_grid",
                env_kwargs={
                    "n_agents": args.n_agents,
                    "grid_size": args.grid_size,
                    "episode_steps": args.episode_steps,
                    "graph": args.graph,
                },
                algo=algo,
                algo_kwargs={
                    "buffer_capacity": 50_000,
                    "batch_size": 32,
                    "target_update_interval": 200,
                    "lr": 5e-4,
                    "gamma": 0.99,
                    "gnn_layers": 2,
                    "gnn_hidden": 64,
                },
                total_env_steps=args.steps,
                eps_start=1.0,
                eps_end=0.05,
                eps_anneal_steps=int(args.steps * 0.8),
                seed=seed,
                log_dir=args.log_dir,
                run_id=f"{algo}__{args.graph}__N{args.n_agents}__seed{seed}",
            )
            train(cfg)
            print(f"{tag}  done in {time.monotonic() - run_t0:.1f}s "
                  f"(elapsed {time.monotonic() - t0:.1f}s)", flush=True)

    print(f"[phase1] all {n_jobs} runs complete in "
          f"{time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
