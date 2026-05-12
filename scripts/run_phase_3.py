"""Phase 3 — external-validity sweep on PettingZoo simple_spread.

Reads `configs/phase_2_locked.yaml` (Phase 2A's locked hyperparameters)
and runs all four algorithms on a canonical cooperative MARL benchmark.
The point is to check whether the Phase 2 conclusions hold up on a
benchmark that the field actually uses, with a different observation
space, reward structure, and physics.

Sweep matrix:
    4 algorithms × 2 agent counts {3, 6} × 5 seeds = 40 runs

Total budget: 30k env steps per run, max_cycles=25 per episode.
Wall-clock estimate on M1 Pro / MPS: ~3.5 hours.

simple_spread reward structure:
    - At each step, each agent gets a negative scalar reward equal to
      the minimum distance from its assigned landmark to any agent.
      This is a global cooperative shared reward (negative the lower
      the better). No explicit success indicator.
    - Headline metric: mean evaluation return over the last 25% of
      training (less negative = better).

Output: results/phase_3_mpe/{algo}/{n_agents}_seed{N}.csv
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
N_AGENTS = (3, 6)
SEEDS = [0, 1, 2, 3, 4]

COMMON = dict(
    env="simple_spread",
    coordination_graph="full",
    mpe_max_cycles=25,
    total_steps=30_000,
    warmup_steps=1_000,
    batch_size=64,
    gamma=0.99,
    hidden=64,
    target_update_every=200,
    grad_clip=10.0,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=15_000,
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
            "   Run `python scripts/lock_tuning.py` (Phase 2A) first.\n"
        )
        sys.exit(2)

    locked = yaml.safe_load(LOCKED.read_text())
    print(f"Phase 3 — loaded locked hyperparameters from {LOCKED}:")
    for algo, params in locked.items():
        print(f"  {algo}: {params or '(defaults)'}")

    runs = [(a, n, s) for a in ALGOS for n in N_AGENTS for s in SEEDS]
    print(
        f"\nPhase 3 sweep: {len(ALGOS)} algos × {len(N_AGENTS)} N "
        f"× {len(SEEDS)} seeds = {len(runs)} runs"
    )
    print(f"Budget: {COMMON['total_steps']} env steps/run on simple_spread")
    print()

    t_total = time.time()
    for i, (algo, n_agents, seed) in enumerate(runs):
        algo_params = locked.get(algo, {}) or {}
        log_path = f"results/phase_3_mpe/{algo}/n{n_agents}_seed{seed}.csv"
        merged = {
            **COMMON,
            **algo_params,
            "algo": algo,
            "n_agents": n_agents,
            "seed": seed,
            "log_path": log_path,
        }
        accepted = {k: v for k, v in merged.items() if k in Args.__annotations__}
        args = Args(**accepted)
        print(f"=== [{i+1}/{len(runs)}] {algo} N={n_agents} seed={seed} ===")
        t0 = time.time()
        path = main(args)
        dt = time.time() - t0
        print(f"  done in {dt:.1f}s -> {path}\n")
    print(f"Phase 3 sweep total wall: {(time.time() - t_total) / 60:.1f} min")


if __name__ == "__main__":
    main_sweep()
