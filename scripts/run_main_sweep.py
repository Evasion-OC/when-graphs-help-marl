"""Phase 2C — locked-hyperparameter main sweep.

Reads `configs/phase_2_locked.yaml` (produced by `scripts/lock_tuning.py`)
and runs the locked configuration of each algorithm across the matrix:

    4 algorithms × 3 task scales × 5 seeds  =  60 runs

Task scales:
- small:  2 agents on 4×4, max 20 steps
- medium: 3 agents on 5×5, max 25 steps
- large:  4 agents on 6×6, max 40 steps

Coordination graph fixed at 'full' (Phase 4 ablates graph structure).
Step budget: 30k env steps per run (3× the tuning budget).

Wall-clock estimate: ~3 hours on M1 Pro / MPS.

Output: results/phase_2_main/{algo}/{scale}_seed{N}.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gnnmarl.train import Args, main  # noqa: E402

LOCKED = ROOT / "configs" / "phase_2_locked.yaml"

SCALES = {
    "small":  dict(grid_size=4, n_agents=2, max_steps=20, eps_decay_steps=15_000),
    "medium": dict(grid_size=5, n_agents=3, max_steps=25, eps_decay_steps=20_000),
    "large":  dict(grid_size=6, n_agents=4, max_steps=40, eps_decay_steps=20_000),
}

ALGOS = ("iql", "vdn", "qmix", "gnn_qmix")
SEEDS = [0, 1, 2, 3, 4]

COMMON = dict(
    env="coord_grid",
    coordination_graph="full",
    shape_distance=True,
    total_steps=30_000,
    warmup_steps=1_000,
    batch_size=64,
    gamma=0.99,
    hidden=64,
    target_update_every=200,
    grad_clip=10.0,
    eps_start=1.0,
    eps_end=0.05,
    eval_every_episodes=100,
    eval_episodes=20,
    device="auto",
    lr=5e-4,
    gnn_layers=2,
)


def main_sweep() -> None:
    if not LOCKED.exists():
        sys.stderr.write(
            f"!! {LOCKED} not found.\n"
            "   Run `python scripts/lock_tuning.py` first to produce it.\n"
        )
        sys.exit(2)

    locked = yaml.safe_load(LOCKED.read_text())
    print(f"Loaded locked hyperparameters from {LOCKED}:")
    for algo, params in locked.items():
        print(f"  {algo}: {params or '(defaults)'}")

    runs = [(a, s, sc) for a in ALGOS for sc in SCALES for s in SEEDS]
    print(
        f"\nPhase 2C main sweep: {len(ALGOS)} algos × "
        f"{len(SCALES)} scales × {len(SEEDS)} seeds = {len(runs)} runs"
    )
    print(f"Budget: {COMMON['total_steps']} env steps/run")
    print()

    t_total = time.time()
    for i, (algo, seed, scale) in enumerate(runs):
        scale_cfg = SCALES[scale]
        algo_params = locked.get(algo, {}) or {}
        log_path = f"results/phase_2_main/{algo}/{scale}_seed{seed}.csv"
        merged = {
            **COMMON,
            **scale_cfg,
            **algo_params,
            "algo": algo,
            "seed": seed,
            "log_path": log_path,
        }
        accepted = {k: v for k, v in merged.items() if k in Args.__annotations__}
        args = Args(**accepted)
        print(f"=== [{i+1}/{len(runs)}] {algo} {scale} seed={seed} ===")
        t0 = time.time()
        path = main(args)
        dt = time.time() - t0
        print(f"  done in {dt:.1f}s -> {path}\n")
    print(f"Phase 2C main sweep total wall: {(time.time() - t_total) / 60:.1f} min")


if __name__ == "__main__":
    main_sweep()
