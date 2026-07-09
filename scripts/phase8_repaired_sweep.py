"""Phase 8R — repaired-GCN control at the negative endpoint (parallel).

Closes the one cell the boundary framing leaves open: the negative-endpoint
phases (1/2/8/11) run the PLAIN GCN (no residual, no LayerNorm), while every
positive-endpoint arm runs the repaired GCN (residual+LayerNorm). Does the
full-information graph penalty survive the stabilisation? This sweep runs
GNN-QMIX with the repaired encoder on the exact Phase 8 cells (coord_grid
RING, N in {4, 8}, 150k steps) so the result is directly comparable to
results/phase8/ (QMIX / MLP-QMIX / GNN-QMIX at 10 seeds).

Env/algo hyperparameters are copied verbatim from phase8_confound_sweep.py;
the ONLY change is ``gnn_residual=True, gnn_layernorm=True`` (matching the
repaired arms of scripts/phaseC_structure_sweep.py).

Usage::
    python scripts/phase8_repaired_sweep.py                    # 2 N x 10 seeds
    python scripts/phase8_repaired_sweep.py --workers 10
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

DEFAULT_NS = (4, 8)

# Verbatim from phase8_confound_sweep.py; the repaired flags are the only add.
ALGO_KWARGS = {
    "buffer_capacity": 50_000,
    "batch_size": 32,
    "target_update_interval": 200,
    "lr": 5e-4,
    "gamma": 0.99,
    "gnn_layers": 2,
    "gnn_hidden": 64,
    "gnn_residual": True,
    "gnn_layernorm": True,
}


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
            "grid_size": 5,
            "episode_steps": 25,
            "graph": job["graph"],
        },
        algo="gnn_qmix",
        algo_kwargs=dict(ALGO_KWARGS),
        total_env_steps=steps,
        update_every=4,
        warmup_steps=200,
        eps_anneal_steps=int(steps * 0.8),
        seed=job["seed"],
        log_dir=Path(job["log_dir"]),
        run_id=job["run_id"],
    )
    run_dir = train(cfg)
    ret = pd.read_csv(Path(run_dir) / "episodes.csv")["return"].to_numpy(dtype=float)
    k = max(1, int(len(ret) * 0.1))  # Phase 8 final window (last 10%)
    return {
        "arm": "gnn_qmix_repaired", "n_agents": n, "graph": job["graph"],
        "seed": job["seed"], "steps": steps, "run_dir": str(run_dir),
        "n_episodes": int(len(ret)), "final": float(ret[-k:].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--ns", type=str, default=",".join(map(str, DEFAULT_NS)))
    ap.add_argument("--graph", type=str, default="ring")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 6))
    ap.add_argument("--log-dir", type=Path, default=REPO_ROOT / "results" / "phase8_repaired")
    args = ap.parse_args()

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    jobs = [
        {
            "n_agents": n, "seed": seed, "steps": args.steps, "graph": args.graph,
            "log_dir": str(args.log_dir),
            "run_id": f"gnn_rep__N{n}__{args.graph}__seed{seed}",
        }
        for n in ns
        for seed in range(args.seeds)
    ]
    args.log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase8R] {len(jobs)} runs x {args.steps} steps | N={ns} "
          f"graph={args.graph} workers={args.workers}", flush=True)

    t0 = time.monotonic()
    rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run_one, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            done += 1
            print(f"  [{done}/{len(jobs)}] N={r['n_agents']} seed{r['seed']}: "
                  f"final={r['final']:.2f} ({time.monotonic() - t0:.0f}s)", flush=True)

    manifest = args.log_dir / "manifest_repaired.csv"
    df = pd.DataFrame(rows)
    df.sort_values(["n_agents", "seed"]).to_csv(manifest, index=False)
    print(f"\n[phase8R] complete in {time.monotonic() - t0:.0f}s -> {manifest}")
    for n in ns:
        sub = df[df.n_agents == n]["final"]
        print(f"  N={n}: mean {sub.mean():7.2f}  (sd {sub.std():.2f}, n={len(sub)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
