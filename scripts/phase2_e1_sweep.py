"""Phase 2 — E1 sweep (H1, H2): scale + graph-regime test.

Sweeps team size and graph regime under the four algorithms. Two regimes:

* **structured** — the *ring* graph: sparse, structured, diameter floor(N/2).
* **random**     — *Erdős–Rényi* G(N, p) with ``p`` chosen so the expected
  edge count matches the ring (2N edges → p = 2/(N-1)). Same edge density,
  no structure.

The matched-density design isolates the *structure* effect from the *edge
budget* effect — i.e. it answers whether the coordination GRAPH (not just
the amount of coordination signal) is what GNN-QMIX exploits.

Default sweep (compute-bounded scope): N ∈ {4, 8} × 2 graphs × 4 algos ×
3 seeds = 48 runs at 30k env steps each. Per docs/PHASES.md this is scaled
down from the full N ∈ {4, 8, 16} × 5 seeds × 200k spec; the original spec
required ~24 M env steps which does not fit the CPU budget for a single
autonomous run on commodity hardware.
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
N_AGENTS = (4, 8)
GRAPHS = ("ring", "erdos_renyi")
DEFAULT_SEEDS = (0, 1, 2)


def er_prob_matching_ring(n: int) -> float:
    """Edge prob p s.t. expected ER edges = ring edges (= N)."""
    return min(1.0, 2.0 / max(n - 1, 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--log-dir", type=Path,
                        default=REPO_ROOT / "results" / "phase2")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    args.log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    n_jobs = len(ALGOS) * len(N_AGENTS) * len(GRAPHS) * len(seeds)
    print(f"[phase2] {n_jobs} runs x {args.steps} env steps "
          f"-> {args.log_dir}", flush=True)

    counter = 0
    for n in N_AGENTS:
        for graph in GRAPHS:
            env_extra = {"graph": graph}
            if graph == "erdos_renyi":
                env_extra["er_prob"] = er_prob_matching_ring(n)
            for algo in ALGOS:
                for seed in seeds:
                    counter += 1
                    tag = (f"[{counter}/{n_jobs}] {algo} N={n} graph={graph} "
                           f"seed={seed}")
                    run_t0 = time.monotonic()
                    cfg = TrainConfig(
                        env="coord_grid",
                        env_kwargs={
                            "n_agents": n,
                            "grid_size": 5,
                            "episode_steps": 25,
                            **env_extra,
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
                        update_every=4,
                        warmup_steps=200,
                        eps_anneal_steps=int(args.steps * 0.8),
                        seed=seed,
                        log_dir=args.log_dir,
                        run_id=f"{algo}__N{n}__{graph}__seed{seed}",
                    )
                    train(cfg)
                    print(f"{tag}  done in {time.monotonic() - run_t0:.1f}s "
                          f"(elapsed {time.monotonic() - t0:.1f}s)", flush=True)

    print(f"[phase2] all {n_jobs} runs complete in "
          f"{time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
