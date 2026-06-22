"""Stage D -- out-of-harness replication of the Stage C structure result.

Same structure test (true vs wrong vs all-to-all comm graph at byte-identical
params), but on TokenMatch: a referential matching game with NO grid, NO movement,
NO navigation -- agents must OUTPUT their partner's private token, which can only
arrive over the coordination graph. A win here shows the Stage C effect is not a
CoordGrid/navigation artifact.

Arms (identical gnn_qmix architecture; only env.adjacency() differs):
  mlp           -- no-comm parameter-matched control
  gnn_true      -- GCN over the TRUE matching          (comm == task)
  gnn_wrong     -- GCN over a wrong matching, matched   (comm != task)
  gnn_complete  -- GCN over all-to-all                  (most comm, unfocused)

Usage::
    python scripts/phaseD_token_sweep.py --seeds 3            # smoke
    python scripts/phaseD_token_sweep.py --seeds 12 --workers 16
    python scripts/phaseC_analysis.py --label token_N6       # reuse the analysis
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

N_TOKENS = 5
EPISODE_STEPS = 10


def arms() -> tuple[tuple[str, str, str, dict], ...]:
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
        env="token_match",
        env_kwargs={
            "n_agents": n,
            "n_tokens": N_TOKENS,
            "episode_steps": EPISODE_STEPS,
            "graph": job["env_graph"],
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
        "n_agents": n, "seed": job["seed"], "run_dir": str(run_dir),
        "steps": steps, "n_episodes": int(len(ret)), "final": float(ret[-k:].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-agents", type=int, default=6)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--log-dir", type=Path, default=REPO_ROOT / "results" / "phaseD")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    label = f"token_N{args.n_agents}"
    jobs: list[dict] = []
    for arm, algo, env_graph, extra in arms():
        for seed in range(args.seeds):
            jobs.append({
                "arm": arm, "algo": algo, "env_graph": env_graph, "extra": extra,
                "n_agents": args.n_agents, "seed": seed, "steps": args.steps,
                "device": args.device, "log_dir": str(args.log_dir / label),
                "run_id": f"{arm}__{label}__seed{seed}",
            })

    print(f"[phaseD] token_match {len(jobs)} runs x {args.steps} steps | N={args.n_agents} "
          f"K={N_TOKENS} | workers={args.workers} device={args.device}", flush=True)
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

    # Write under results/phaseC so phaseC_analysis.py --label token_N{N} finds it.
    out_dir = REPO_ROOT / "results" / "phaseC"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f"manifest_{label}.csv"
    df = pd.DataFrame(rows)
    df.sort_values(["arm", "seed"]).to_csv(manifest, index=False)
    print(f"\n[phaseD] complete in {time.monotonic() - t0:.0f}s -> {manifest}")
    print(f"\n=== mean final return by arm (token_match, N={args.n_agents}) ===")
    for arm, *_ in arms():
        sub = df[df.arm == arm]["final"]
        print(f"  {arm:<13} {sub.mean():7.2f}  (sd {sub.std():.2f}, n={len(sub)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
