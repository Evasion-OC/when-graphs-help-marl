"""Stage C -- the graph-STRUCTURE test (parallel).

The decisive question (advisor): does the *right* graph beat a *wrong* graph at
matched communication and parameters? If yes, graph STRUCTURE -- not just having a
channel -- is what helps. Tested on private-goal routing tasks under ``ego``
observability, where a partner's/neighbour's goal can arrive only over the graph.

Two task topologies (``--task``):
  pair  -- perfect MATCHING: reach your one partner's goal. True=matching,
           wrong=matching_wrong (same density), complete=all-to-all. The clean
           mechanism-isolator (degree 1).
  nbr   -- +-1 RING neighbourhood (degree 2): reach the centroid of your two ring
           neighbours' goals. True=ring, wrong=skip_ring (+-2, same degree),
           complete=all-to-all. This makes the *neighbourhood set* (topology)
           load-bearing, not just a 1-to-1 link.

Two arm suites (``--suite``):
  core      -- mlp, gnn_true, gnn_wrong, gnn_complete (all GCN; isolate structure).
  attention -- gnn_true vs GAT/DGN on the COMPLETE graph: can *learned* attention
               recover the right neighbour from all-to-all, or does even attention
               need the right structure? (Defends the complete arm vs "mean-pool
               is just dumb"; ties to the DGN/G2ANet methods the paper critiques.)

Usage::
    python scripts/phaseC_structure_sweep.py --task nbr --seeds 3            # smoke
    python scripts/phaseC_structure_sweep.py --task nbr --seeds 12 --workers 16
    python scripts/phaseC_structure_sweep.py --task pair --suite attention --seeds 12
    python scripts/phaseC_structure_sweep.py --task nbr --dry-run
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

GRID = 3
EPISODE_STEPS = 25

# Per task: (env routing flag, true comm graph, wrong comm graph).
TASKS = {
    "pair": ("pair_routing", "matching", "matching_wrong"),
    "nbr": ("nbr_routing", "ring", "skip_ring"),
}


def arms(task: str, suite: str) -> tuple[tuple[str, str, str, dict], ...]:
    """(arm_name, algo, env_graph, extra algo_kwargs)."""
    _, true_g, wrong_g = TASKS[task]
    repaired = {"gnn_residual": True, "gnn_layernorm": True}
    if suite == "core":
        return (
            ("mlp", "mlp_qmix", true_g, {"gnn_layers": 2}),
            ("gnn_true", "gnn_qmix", true_g, {"gnn_layers": 2, **repaired}),
            ("gnn_wrong", "gnn_qmix", wrong_g, {"gnn_layers": 2, **repaired}),
            ("gnn_complete", "gnn_qmix", "complete", {"gnn_layers": 2, **repaired}),
        )
    if suite == "attention":
        # Does learned attention over all-to-all recover the true-structure GCN?
        return (
            ("mlp", "mlp_qmix", true_g, {"gnn_layers": 2}),
            ("gnn_true", "gnn_qmix", true_g, {"gnn_layers": 2, **repaired}),
            ("gcn_complete", "gnn_qmix", "complete", {"gnn_layers": 2, **repaired}),
            # Attention arms now stabilization-MATCHED to the GCN arms (residual+
            # LayerNorm) so GAT/DGN-vs-GCN is a clean aggregator comparison.
            ("gat_complete", "gat_qmix", "complete", {"gnn_layers": 2, **repaired}),
            ("dgn_complete", "dgn_qmix", "complete", {"gnn_layers": 2, **repaired}),
            ("gat_true", "gat_qmix", true_g, {"gnn_layers": 2, **repaired}),
        )
    raise ValueError(f"unknown suite {suite!r}")


def _run_one(job: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    from gnnmarl.training import TrainConfig, train

    steps = job["steps"]
    n = job["n_agents"]
    cfg = TrainConfig(
        env="coord_grid",
        env_kwargs={
            "n_agents": n,
            "grid_size": GRID,
            "episode_steps": EPISODE_STEPS,
            "graph": job["env_graph"],
            job["routing_flag"]: True,
            "obs_mode": job["obs"],
            "max_neighbors_override": n - 1,  # constant obs_dim across comm graphs
        },
        algo=job["algo"],
        algo_kwargs={
            "buffer_capacity": 50_000,
            "batch_size": 32,
            "target_update_interval": 200,
            "lr": 5e-4,
            "gamma": 0.99,
            "gnn_hidden": 64,
            **job["extra"],
        },
        total_env_steps=steps,
        eps_start=1.0,
        eps_end=0.05,
        eps_anneal_steps=int(steps * 0.8),
        seed=job["seed"],
        device=job["device"],
        log_dir=Path(job["log_dir"]),
        run_id=job["run_id"],
    )
    run_dir = train(cfg)
    ret = pd.read_csv(Path(run_dir) / "episodes.csv")["return"].to_numpy(dtype=float)
    k = max(1, int(len(ret) * 0.2))
    return {
        "arm": job["arm"], "algo": job["algo"], "env_graph": job["env_graph"],
        "task": job["task"], "suite": job["suite"], "obs": job["obs"],
        "n_agents": n, "seed": job["seed"], "run_dir": str(run_dir),
        # Stamp the budget so a manifest reveals co-mingled step counts (a label
        # dir reused across --steps would otherwise silently mix budgets).
        "steps": steps, "n_episodes": int(len(ret)),
        "final": float(ret[-k:].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="pair", choices=tuple(TASKS))
    ap.add_argument("--suite", type=str, default="core", choices=("core", "attention"))
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-agents", type=int, default=6)
    ap.add_argument("--obs", type=str, default="ego", choices=("ego", "full"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--log-dir", type=Path, default=REPO_ROOT / "results" / "phaseC")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    routing_flag = TASKS[args.task][0]
    suite_tag = "" if args.suite == "core" else f"_{args.suite}"
    label = f"{args.task}_N{args.n_agents}_{args.obs}{suite_tag}"
    jobs: list[dict] = []
    for arm, algo, env_graph, extra in arms(args.task, args.suite):
        for seed in range(args.seeds):
            jobs.append({
                "arm": arm, "algo": algo, "env_graph": env_graph, "extra": extra,
                "routing_flag": routing_flag, "task": args.task, "suite": args.suite,
                "obs": args.obs, "n_agents": args.n_agents, "seed": seed,
                "steps": args.steps, "device": args.device,
                "log_dir": str(args.log_dir / label),
                "run_id": f"{arm}__{label}__seed{seed}",
            })

    print(f"[phaseC] task={args.task} suite={args.suite} {len(jobs)} runs x {args.steps} "
          f"steps | N={args.n_agents} obs={args.obs} grid={GRID} | "
          f"workers={args.workers} device={args.device}", flush=True)
    if args.dry_run:
        for j in jobs:
            print(f"  {j['run_id']:<40} algo={j['algo']:<9} graph={j['env_graph']}")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run_one, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            done += 1
            print(f"  [{done}/{len(jobs)}] {r['arm']:<13} seed{r['seed']}: "
                  f"final={r['final']:.2f} ({time.monotonic() - t0:.0f}s)", flush=True)

    manifest = args.log_dir / f"manifest_{label}.csv"
    df = pd.DataFrame(rows)
    df.sort_values(["arm", "seed"]).to_csv(manifest, index=False)
    print(f"\n[phaseC] complete in {time.monotonic() - t0:.0f}s -> {manifest}")
    print(f"\n=== mean final return by arm (task={args.task} obs={args.obs}) ===")
    for arm, *_ in arms(args.task, args.suite):
        sub = df[df.arm == arm]["final"]
        print(f"  {arm:<13} {sub.mean():7.2f}  (sd {sub.std():.2f}, n={len(sub)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
