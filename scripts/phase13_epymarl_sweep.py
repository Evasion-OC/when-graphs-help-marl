"""Phase 13 --- external validity on EPyMARL cooperative benchmarks.

Replicates the controlled graph-vs-capacity comparison on two *recognized*
standard cooperative MARL benchmarks (Papoudakis et al., 2021), beyond
CoordGrid and MPE:
  * Level-Based Foraging (``lbforaging``)
  * Multi-Robot Warehouse (``rware``)

Same five algorithms, same locked hyperparameters, same Welch/Holm/MWU
protocol as Phases 8/9. Neither benchmark exposes a graph, so the GNNs run
on a complete coordination graph (the most-informative case; see
``epymarl_adapter.py``).

Run-dir naming: ``<algo>__<tag>__<graph>__seed<s>`` so the analysis can
group by env tag.

Usage:
  python scripts/phase13_epymarl_sweep.py \
      --envs lbf:Foraging-10x10-3p-3f-v3 --tags lbf3p \
      --steps 1000000 --seeds 5
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

# Byte-identical to the Phase 8/9 locked config.
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=str, required=True,
                    help="comma-separated full env ids, e.g. "
                         "'lbf:Foraging-10x10-3p-3f-v3,rware:rware-tiny-4ag-v2'")
    ap.add_argument("--tags", type=str, required=True,
                    help="comma-separated short tags parallel to --envs")
    ap.add_argument("--algos", type=str, default=",".join(DEFAULT_ALGOS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--graph", type=str, default="complete")
    ap.add_argument("--update-every", type=int, default=4)
    ap.add_argument("--log-dir", type=Path,
                    default=REPO_ROOT / "results" / "phase13_epymarl")
    args = ap.parse_args()

    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    assert len(envs) == len(tags), "--envs and --tags must be parallel"
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    args.log_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = len(envs) * len(algos) * len(seeds)
    t0 = time.monotonic()
    print(f"[phase13] {n_jobs} runs x {args.steps} env steps -> {args.log_dir}",
          flush=True)
    print(f"[phase13] envs={list(zip(tags, envs))} algos={algos} seeds={seeds}",
          flush=True)

    counter = 0
    for env, tag in zip(envs, tags):
        for algo in algos:
            for seed in seeds:
                counter += 1
                run_t0 = time.monotonic()
                cfg = TrainConfig(
                    env=env,
                    env_kwargs={"graph": args.graph},
                    algo=algo,
                    algo_kwargs=dict(ALGO_KWARGS),
                    total_env_steps=args.steps,
                    update_every=args.update_every,
                    warmup_steps=1000,
                    eps_anneal_steps=int(args.steps * 0.6),
                    seed=seed,
                    log_dir=args.log_dir,
                    run_id=f"{algo}__{tag}__{args.graph}__seed{seed}",
                )
                train(cfg)
                print(f"[{counter}/{n_jobs}] {algo} {tag} seed={seed} done in "
                      f"{time.monotonic() - run_t0:.1f}s "
                      f"(elapsed {time.monotonic() - t0:.1f}s)", flush=True)

    print(f"[phase13] all {n_jobs} runs complete in "
          f"{time.monotonic() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
