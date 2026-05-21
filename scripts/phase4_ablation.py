"""Phase 4 — GNN depth × graph diameter ablation (H3).

Tests the predicted *inverted-U*: GNN-QMIX should peak when the GNN depth L
approximately matches the task graph's diameter d. Layers below d cannot
propagate signal across the full coordination graph; layers above d are
unnecessary depth that hurts optimisation.

Design: 3 graphs × 4 GNN depths × 3 seeds = 36 runs by default. **All
graphs use N=4 fixed** so the diameter is the only varying factor (a
previous design varied N alongside d, which confounded the two; that
confound is fixed here per the pre-submission review).

Graph kind → diameter on CoordGrid (N=4):

    complete  d = 1
    ring      d = 2  (cycle on 4 nodes)
    line      d = 3  (path on 4 nodes)

This caps the largest diameter at 3 (because the line graph on N nodes has
diameter N-1, and we hold N=4). With L ∈ {1, 2, 3, 4} we therefore probe
the *right* side of the inverted-U at d=3 (L=4 should be flat or worse
than L=3) but do not get an L-too-small / L-too-large bracket for d ≥ 4.
Larger-diameter probes need larger teams and a heavier compute budget;
out of scope for this revision.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnmarl.training import TrainConfig, train  # noqa: E402


# (label_for_table, graph_kwarg, n_agents) -> diameter; N fixed at 4.
DIAMETER_LEVELS: list[tuple[str, dict, int]] = [
    ("complete (d=1)", {"graph": "complete"}, 4),
    ("ring (d=2)",     {"graph": "ring"},     4),
    ("line (d=3)",     {"graph": "line"},     4),
]

DEPTHS = (1, 2, 3, 4)
DEFAULT_SEEDS = (0, 1, 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--log-dir", type=Path,
                        default=REPO_ROOT / "results" / "phase4")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    args.log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    n_jobs = len(DIAMETER_LEVELS) * len(DEPTHS) * len(seeds)
    print(f"[phase4] {n_jobs} runs x {args.steps} env steps "
          f"-> {args.log_dir}", flush=True)

    counter = 0
    for (label, env_extra, n_agents) in DIAMETER_LEVELS:
        for depth in DEPTHS:
            for seed in seeds:
                counter += 1
                tag = f"[{counter}/{n_jobs}] {label} L={depth} seed={seed}"
                run_t0 = time.monotonic()
                cfg = TrainConfig(
                    env="coord_grid",
                    env_kwargs={
                        "n_agents": n_agents,
                        "grid_size": 5,
                        "episode_steps": 25,
                        # Hold obs_dim constant across graphs so the
                        # depth-diameter ablation is not confounded with
                        # input dimension. N-1 covers any graph's max
                        # degree at this team size.
                        "max_neighbors_override": n_agents - 1,
                        **env_extra,
                    },
                    algo="gnn_qmix",
                    algo_kwargs={
                        "buffer_capacity": 50_000,
                        "batch_size": 32,
                        "target_update_interval": 200,
                        "lr": 5e-4,
                        "gamma": 0.99,
                        "gnn_layers": depth,
                        "gnn_hidden": 64,
                    },
                    total_env_steps=args.steps,
                    update_every=4,
                    warmup_steps=200,
                    eps_anneal_steps=int(args.steps * 0.8),
                    seed=seed,
                    log_dir=args.log_dir,
                    run_id=f"d{DIAMETER_LEVELS.index((label, env_extra, n_agents))+1}"
                           f"__L{depth}__seed{seed}",
                )
                train(cfg)
                print(f"{tag}  done in {time.monotonic() - run_t0:.1f}s "
                      f"(elapsed {time.monotonic() - t0:.1f}s)", flush=True)

    print(f"[phase4] all {n_jobs} runs complete in "
          f"{time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
