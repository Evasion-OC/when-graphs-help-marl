"""Stage C -- the graph-STRUCTURE test (parallel).

The decisive question the advisor raised: does the *right* graph beat a *wrong*
graph at matched communication and parameters? If yes, graph STRUCTURE (not just
having a comm channel) is what helps. We test it on the pair-routing task: every
agent holds a private goal and must reach its PARTNER's goal (a fixed perfect
matching defines partners), under ``ego`` observability so the partner's goal can
arrive only over the graph.

Arms (identical gnn_qmix architecture + params; ONLY the comm graph differs):
  mlp          -- no-comm, parameter-matched control (floor: no channel at all)
  gnn_true     -- GNN message passing over the TRUE matching   (comm == task)
  gnn_wrong    -- GNN over a WRONG matching, same density       (comm != task)
  gnn_complete -- GNN over the COMPLETE graph (all-to-all)      (most comm, unfocused)

Pre-registered prediction (structure matters):
  gnn_true  >  max(gnn_wrong, gnn_complete, mlp), significantly.
If instead gnn_true ~ gnn_complete, structure is irrelevant (comm alone helps) --
report that honestly (it is essentially the negative paper, reworded).

Usage::
    python scripts/phaseC_structure_sweep.py --seeds 3            # smoke
    python scripts/phaseC_structure_sweep.py --seeds 12 --workers 16   # confirmatory
    python scripts/phaseC_structure_sweep.py --obs full          # crossover control
    python scripts/phaseC_structure_sweep.py --dry-run
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


def arms() -> tuple[tuple[str, str, str, dict], ...]:
    """(arm_name, algo, env_graph, extra algo_kwargs). Same arch for all GNN arms."""
    repaired = {"gnn_residual": True, "gnn_layernorm": True}
    return (
        ("mlp", "mlp_qmix", "matching", {"gnn_layers": 2}),
        ("gnn_true", "gnn_qmix", "matching", {"gnn_layers": 2, **repaired}),
        ("gnn_wrong", "gnn_qmix", "matching_wrong", {"gnn_layers": 2, **repaired}),
        ("gnn_complete", "gnn_qmix", "complete", {"gnn_layers": 2, **repaired}),
    )


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
            "pair_routing": True,
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
        "arm": job["arm"],
        "algo": job["algo"],
        "env_graph": job["env_graph"],
        "obs": job["obs"],
        "n_agents": n,
        "seed": job["seed"],
        "run_dir": str(run_dir),
        "final": float(ret[-k:].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-agents", type=int, default=6)
    ap.add_argument("--obs", type=str, default="ego", choices=("ego", "full"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--log-dir", type=Path, default=REPO_ROOT / "results" / "phaseC")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    label = f"N{args.n_agents}_{args.obs}"
    jobs: list[dict] = []
    for arm, algo, env_graph, extra in arms():
        for seed in range(args.seeds):
            jobs.append({
                "arm": arm, "algo": algo, "env_graph": env_graph, "extra": extra,
                "obs": args.obs, "n_agents": args.n_agents, "seed": seed,
                "steps": args.steps, "device": args.device,
                "log_dir": str(args.log_dir / label),
                "run_id": f"{arm}__{args.obs}__N{args.n_agents}__seed{seed}",
            })

    print(f"[phaseC] {len(jobs)} runs x {args.steps} steps | N={args.n_agents} "
          f"obs={args.obs} grid={GRID} | workers={args.workers} device={args.device}",
          flush=True)
    if args.dry_run:
        for j in jobs:
            print(f"  {j['run_id']:<34} algo={j['algo']:<9} graph={j['env_graph']}")
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
    pd.DataFrame(rows).sort_values(["arm", "seed"]).to_csv(manifest, index=False)
    df = pd.DataFrame(rows)
    print(f"\n[phaseC] complete in {time.monotonic() - t0:.0f}s -> {manifest}")
    print("\n=== mean final return by arm (obs={}) ===".format(args.obs))
    for arm, _, _, _ in arms():
        sub = df[df.arm == arm]["final"]
        print(f"  {arm:<13} {sub.mean():7.2f}  (sd {sub.std():.2f}, n={len(sub)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
