"""Phase 1 pilot sweep.

3 agents on 5x5 grid with distance-based potential shaping. 4 algorithms x
3 seeds. Sequential execution — MPS GPU is shared and we want clean
per-run wall-clock numbers.

Run:
    python scripts/run_pilot.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gnnmarl.train import Args, main  # noqa: E402

ALGOS = ["iql", "vdn", "qmix", "gnn_qmix"]
SEEDS = [0, 1, 2]

BASE = dict(
    env="coord_grid",
    grid_size=4,
    n_agents=2,
    max_steps=20,
    coordination_graph="full",
    shape_distance=True,
    total_steps=10_000,
    warmup_steps=500,
    batch_size=64,
    lr=5e-4,
    gamma=0.99,
    hidden=64,
    embed_dim=32,
    gnn_layers=2,
    target_update_every=200,
    grad_clip=10.0,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=5_000,
    eval_every_episodes=50,
    eval_episodes=20,
    device="auto",
)


def main_sweep() -> None:
    runs = [(a, s) for a in ALGOS for s in SEEDS]
    print(f"Phase 1 pilot: {len(runs)} runs ({len(ALGOS)} algos x {len(SEEDS)} seeds)")
    print(f"Env: {BASE['n_agents']} agents on {BASE['grid_size']}x{BASE['grid_size']}, "
          f"shaping={BASE['shape_distance']}, steps={BASE['total_steps']}")
    print()
    t_total = time.time()
    for i, (algo, seed) in enumerate(runs):
        log_path = f"results/pilot/{algo}_seed{seed}.csv"
        args_dict = {**BASE, "algo": algo, "seed": seed, "log_path": log_path}
        accepted = {k: v for k, v in args_dict.items() if k in Args.__annotations__}
        args = Args(**accepted)
        print(f"=== [{i+1}/{len(runs)}] {algo} seed={seed} ===")
        t0 = time.time()
        path = main(args)
        dt = time.time() - t0
        print(f"  done in {dt:.1f}s -> {path}\n")
    print(f"Total wall: {(time.time() - t_total) / 60:.1f} min")


if __name__ == "__main__":
    main_sweep()
