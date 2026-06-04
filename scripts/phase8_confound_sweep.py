"""Phase 8 — long-horizon confound-killer + capacity control.

Addresses the two reviewer objections that most threaten the paper's
central negative claim, in one sweep, using the SAME codebase and env
config that produced the submitted Phase 1/2 numbers (so results are a
clean apples-to-apples extension, not a re-baseline):

  (A) Undertraining confound. Phase 2 trained the larger GNN-QMIX for
      only 30k env steps; the paper flags undertraining as the leading
      alternative explanation for the negative result. This sweep trains
      every algorithm for --steps (default 150k, >= the 1e5 promised to
      reviewers) with --seeds (default 5, up from 3). If the
      QMIX > GNN-QMIX ordering survives, undertraining is ruled out.

  (B) Capacity vs graph. We add MLP-QMIX, the parameter-matched no-graph
      control: byte-identical to GNN-QMIX (24,518 params) except the GCN
      encoder is replaced by a shape-identical aggregation-free MLP. The
      well-identified comparison is GNN-QMIX vs MLP-QMIX at matched
      params:
        GNN-QMIX ~ MLP-QMIX  -> the graph adds nothing
        GNN-QMIX < MLP-QMIX  -> the graph actively hurts
        GNN-QMIX > MLP-QMIX  -> the graph helps (the positive case)
      and MLP-QMIX vs QMIX isolates the +8,320-param capacity effect.

Cells: coord_grid RING at N in {4, 8} — the two conditions where Phase 1
and Phase 2 showed the clearest QMIX > GNN-QMIX ordering. Env/algo
hyperparameters are copied verbatim from phase2_e1_sweep.py.

Default: 5 algos x 2 N x 5 seeds = 50 runs at 150k steps. On a Colab T4
this is roughly 6-9 h (GNN/MLP runs ~1.7x the IQL/VDN cost). Tune
--steps / --seeds / --algos / --ns to fit the session.

Usage:
  python scripts/phase8_confound_sweep.py
  python scripts/phase8_confound_sweep.py --steps 200000 --seeds 5
  python scripts/phase8_confound_sweep.py --algos qmix,mlp_qmix,gnn_qmix --ns 8
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
DEFAULT_NS = (4, 8)

# Verbatim from phase2_e1_sweep.py so the long run extends the published
# Phase 2 cells rather than introducing a new configuration.
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
    parser.add_argument("--seeds", type=int, default=5,
                        help="number of seeds to run")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="first seed index (use to extend an existing run, "
                             "e.g. --seed-start 5 --seeds 5 runs seeds 5..9)")
    parser.add_argument("--algos", type=str, default=",".join(DEFAULT_ALGOS))
    parser.add_argument("--ns", type=str, default=",".join(map(str, DEFAULT_NS)))
    parser.add_argument("--graph", type=str, default="ring",
                        help="coordination graph topology (ring, line, complete, ...)")
    parser.add_argument("--log-dir", type=Path,
                        default=REPO_ROOT / "results" / "phase8")
    args = parser.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    args.log_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = len(algos) * len(ns) * len(seeds)
    t0 = time.monotonic()
    print(f"[phase8] {n_jobs} runs x {args.steps} env steps "
          f"-> {args.log_dir}", flush=True)
    print(f"[phase8] algos={algos} N={ns} seeds={seeds}", flush=True)

    counter = 0
    for n in ns:
        for algo in algos:
            for seed in seeds:
                counter += 1
                tag = (f"[{counter}/{n_jobs}] {algo} N={n} "
                       f"graph={args.graph} seed={seed}")
                run_t0 = time.monotonic()
                cfg = TrainConfig(
                    env="coord_grid",
                    env_kwargs={
                        "n_agents": n,
                        "grid_size": 5,
                        "episode_steps": 25,
                        "graph": args.graph,
                    },
                    algo=algo,
                    algo_kwargs=dict(ALGO_KWARGS),
                    total_env_steps=args.steps,
                    update_every=4,
                    warmup_steps=200,
                    eps_anneal_steps=int(args.steps * 0.8),
                    seed=seed,
                    log_dir=args.log_dir,
                    run_id=f"{algo}__N{n}__{args.graph}__seed{seed}",
                )
                train(cfg)
                print(f"{tag}  done in {time.monotonic() - run_t0:.1f}s "
                      f"(elapsed {time.monotonic() - t0:.1f}s)", flush=True)

    print(f"[phase8] all {n_jobs} runs complete in "
          f"{time.monotonic() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
