"""Phase 9 — external-validity replication on MPE ``simple_spread``.

The single biggest scope objection to the paper is that every number comes
from one synthetic environment (CoordGrid). This sweep re-runs the *exact
same* controlled comparison on a canonical cooperative benchmark ---
PettingZoo / ``mpe2`` ``simple_spread`` --- using the identical algorithms,
the identical locked hyperparameters, and the identical 150k-step /
10-seed protocol as the Phase 8 confound-killer. The only thing that
changes is the environment.

Why ``simple_spread``: it is the standard fully-cooperative MPE task (N
agents cover N landmarks while avoiding collisions), it has no
task-supplied coordination graph, and it is widely used to benchmark
value-decomposition methods. MPE does not expose a graph, so GNN-QMIX runs
on the natural *k-NN-over-positions* coordination graph (the DGN
convention; see ``mpe_adapter.py``). This is the most favourable honest
setting for a graph prior: the edges encode genuine spatial proximity.

The three decisive contrasts are unchanged from Phase 8:
    QMIX     vs GNN-QMIX   does the (CoordGrid) negative ordering recur?
    GNN-QMIX vs MLP-QMIX   graph effect at matched params
    MLP-QMIX vs QMIX       capacity effect

Cells: ``simple_spread`` at N in {3, 6} (the canonical small setting and a
larger one, paralleling CoordGrid's N in {4, 8}). k-NN graph with k=2.

Default: 5 algos x 2 N x 10 seeds = 100 runs at 150k steps, ~4 h on an
M-series CPU. Tune --steps / --seeds / --algos / --ns / --k to fit.

Usage:
  python scripts/phase9_mpe_sweep.py
  python scripts/phase9_mpe_sweep.py --seeds 5 --ns 3
  python scripts/phase9_mpe_sweep.py --algos qmix,mlp_qmix,gnn_qmix
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnmarl.training import TrainConfig, train  # noqa: E402

DEFAULT_ALGOS = ("qmix", "mlp_qmix", "gnn_qmix", "vdn", "iql")
DEFAULT_NS = (3, 6)

# Byte-identical to the Phase 8 / Phase 2 locked config so the MPE run is a
# clean cross-environment replication, not a re-tuned new experiment.
ALGO_KWARGS = {
    "buffer_capacity": 50_000,
    "batch_size": 32,
    "target_update_interval": 200,
    "lr": 5e-4,
    "gamma": 0.99,
    "gnn_layers": 2,
    "gnn_hidden": 64,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--seeds", type=int, default=10,
                        help="number of seeds to run")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="first seed index (use to extend an existing run)")
    parser.add_argument("--algos", type=str, default=",".join(DEFAULT_ALGOS))
    parser.add_argument("--ns", type=str, default=",".join(map(str, DEFAULT_NS)))
    parser.add_argument("--k", type=int, default=2,
                        help="k for the kNN coordination graph")
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--local-ratio", type=float, default=0.5)
    parser.add_argument("--log-dir", type=Path,
                        default=REPO_ROOT / "results" / "phase9_mpe")
    args = parser.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    args.log_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = len(algos) * len(ns) * len(seeds)
    t0 = time.monotonic()
    print(f"[phase9] {n_jobs} runs x {args.steps} env steps -> {args.log_dir}",
          flush=True)
    print(f"[phase9] env=mpe:simple_spread k={args.k} "
          f"algos={algos} N={ns} seeds={seeds}", flush=True)

    counter = 0
    for n in ns:
        for algo in algos:
            for seed in seeds:
                counter += 1
                tag = (f"[{counter}/{n_jobs}] {algo} N={n} "
                       f"spread seed={seed}")
                run_t0 = time.monotonic()
                cfg = TrainConfig(
                    env="mpe:simple_spread",
                    env_kwargs={
                        "n_agents": n,
                        "k_neighbors": args.k,
                        "max_cycles": args.max_cycles,
                        "local_ratio": args.local_ratio,
                    },
                    algo=algo,
                    algo_kwargs=dict(ALGO_KWARGS),
                    total_env_steps=args.steps,
                    update_every=4,
                    warmup_steps=200,
                    eps_anneal_steps=int(args.steps * 0.8),
                    seed=seed,
                    log_dir=args.log_dir,
                    run_id=f"{algo}__N{n}__spread__seed{seed}",
                )
                train(cfg)
                print(f"{tag}  done in {time.monotonic() - run_t0:.1f}s "
                      f"(elapsed {time.monotonic() - t0:.1f}s)", flush=True)

    print(f"[phase9] all {n_jobs} runs complete in "
          f"{time.monotonic() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
