"""Phase 3 smoke — CPU-only, runs all 4 algos for ~3000 steps each.

Goal: validate the MPE pipeline (env adapter + locked Phase 2A configs +
trainer) end-to-end before committing to the overnight 40-run sweep.
Catches "orthogonal init misbehaves at 18-dim observations" or
"locked hyperparameters don't transfer" issues cheaply (~15 min on CPU).

Not a real experiment — 1 seed, 3000 steps. Don't read final numbers as
results.
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

ALGOS = ("iql", "vdn", "qmix", "gnn_qmix")

COMMON = dict(
    env="simple_spread",
    n_agents=3,
    coordination_graph="full",
    mpe_max_cycles=25,
    total_steps=10_000,
    warmup_steps=500,
    batch_size=64,
    gamma=0.99,
    hidden=64,
    target_update_every=200,
    grad_clip=10.0,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=5_000,
    eval_every_episodes=50,
    eval_episodes=10,
    device="cpu",  # avoid MPS contention with Phase 2C
    lr=5e-4,
    gnn_layers=2,
    seed=0,
)


def main_smoke() -> None:
    if not LOCKED.exists():
        sys.stderr.write(f"!! {LOCKED} not found.\n")
        sys.exit(2)
    locked = yaml.safe_load(LOCKED.read_text())

    print("Phase 3 smoke (CPU): 4 runs, locked Phase 2A configs")
    print()
    t_total = time.time()
    for i, algo in enumerate(ALGOS):
        algo_params = locked.get(algo, {}) or {}
        merged = {
            **COMMON,
            **algo_params,
            "algo": algo,
            "log_path": f"results/phase_3_smoke/{algo}.csv",
        }
        accepted = {k: v for k, v in merged.items() if k in Args.__annotations__}
        args = Args(**accepted)
        print(f"=== [{i+1}/{len(ALGOS)}] {algo} ===")
        t0 = time.time()
        path = main(args)
        dt = time.time() - t0
        print(f"  done in {dt:.1f}s -> {path}\n")
    print(f"Phase 3 smoke total: {(time.time() - t_total) / 60:.1f} min")


if __name__ == "__main__":
    main_smoke()
