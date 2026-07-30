r"""Phase D -- authorized 150k-class re-smoke of {oracle, mlp, gnn_true} plus
the diagnostic-only ``oracle_pos`` arm.

Implements ``results/phaseD_external/PREREGISTRATION.md`` AMENDMENT 1: the
original 30k-cap smoke (recorded in the base ``## AMENDMENT (2026-07-31)``
section of that file) hit the oracle-headroom kill-switch (gate 2 FAIL,
``oracle`` -57.93 vs ``mlp`` -58.26, no separation) and used its one permitted
extension already. AMENDMENT 1 authorizes exactly one further re-smoke at the
150k-class budget used elsewhere in this paper, framed as testing whether the
color->landmark-index indirection becomes learnable at that budget -- NOT as
confirming a known-reachable ceiling (see AMENDMENT 1, A1.1-A1.2 for the
frozen framing and gate rule).

Arms (3 seeds x 150k steps, E1 config: k=3, comm_on=False):

  mlp         -- mlp_qmix, no oracle injection (unchanged gate-1 floor arm).
  oracle      -- mlp_qmix, oracle=True, oracle_mode="color" (unchanged from
                 the base pre-reg's confirmatory ``oracle`` arm -- this is
                 the arm the gate rule is read off).
  gnn_true    -- gnn_qmix (stabilised), graph="true" (calibration/liveness
                 read, not an auto-kill).
  oracle_pos  -- mlp_qmix, oracle=True, oracle_mode="position"
                 (**DIAGNOSTIC ONLY** -- AMENDMENT 1 A1.1.2. Cannot gate or
                 enter any confirmatory family. Exists solely to
                 disambiguate "indirection unlearnable" from "other harness
                 failure" in the Branch-0 write-up if the gate fails again).

Writes to ``results/phaseD_external/cellE1_N6_resmoke150k/`` and
``results/phaseD_external/manifest_cellE1_N6_resmoke150k.csv`` -- distinct
labels from the existing ``cellE1_N6_smoke`` / ``cellE1_N6_30000steps``
directories (never overwrites prior results).

Reuses ``phaseD_external_sweep.py``'s ``_run_mpe_job`` verbatim (job-dict
contract unchanged) so the manifest schema matches the rest of Phase D.

Usage::

    python scripts/phaseD_150k_resmoke.py --dry-run
    python scripts/phaseD_150k_resmoke.py --workers 5
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
from typing import Any  # noqa: E402

import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phaseD_external_sweep as phaseD  # noqa: E402  (sibling script; reuse _run_mpe_job)

STEPS = 150_000
SEEDS = 3
LABEL = "cellE1_N6_resmoke150k"
LOG_DIR = REPO_ROOT / "results" / "phaseD_external"

# (arm, algo, algo_extra, env_extra) -- k/graph/comm_on come from the shared
# base env_kwargs below (E1's frozen config: k=3, graph="true", comm_on=False).
_RESMOKE_ARMS: tuple[tuple[str, str, dict, dict], ...] = (
    ("mlp", "mlp_qmix", {"gnn_layers": 2}, {}),
    ("oracle", "mlp_qmix", {"gnn_layers": 2}, {"oracle": True, "oracle_mode": "color"}),
    ("gnn_true", "gnn_qmix", {"gnn_layers": 2, **phaseD._REPAIRED}, {}),
    (
        "oracle_pos",
        "mlp_qmix",
        {"gnn_layers": 2},
        {"oracle": True, "oracle_mode": "position"},
    ),
)

_BASE_ENV_KWARGS = {"k": 3, "graph": "true", "comm_on": False, "max_cycles": 25}


def _build_jobs(*, device: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for arm, algo, algo_extra, env_extra in _RESMOKE_ARMS:
        env_kwargs = {**_BASE_ENV_KWARGS, **env_extra}
        for seed in range(SEEDS):
            jobs.append(
                {
                    "arm": arm,
                    "algo": algo,
                    "algo_extra": algo_extra,
                    "graph": env_kwargs["graph"],
                    "env_kwargs": env_kwargs,
                    "cell": "E1",
                    "n_agents": 6,
                    "seed": seed,
                    "steps": STEPS,
                    "device": device,
                    "log_dir": str(LOG_DIR / LABEL),
                    "run_id": f"{arm}__{LABEL}__seed{seed}",
                }
            )
    return jobs


def _params_only() -> None:
    import torch

    from gnnmarl.algos import make_algo
    from gnnmarl.algos.maxplus import count_params
    from gnnmarl.envs import make_env

    printed_dims: set[tuple] = set()
    for arm, algo, algo_extra, env_extra in _RESMOKE_ARMS:
        env_kwargs = {**_BASE_ENV_KWARGS, **env_extra}
        env = make_env("mpe_reference_pairs", **env_kwargs)
        dims = (env.obs_dim, env.state_dim, env.n_agents, env.n_actions)
        if dims not in printed_dims:
            printed_dims.add(dims)
            print(f"  [param audit] arm={arm}: obs_dim={env.obs_dim} state_dim={env.state_dim}")
        kw = dict(
            n_agents=env.n_agents,
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            n_actions=env.n_actions,
            device=torch.device("cpu"),
            seed=0,
        )
        algo_kwargs = {"gnn_hidden": 64, **algo_extra}
        m = make_algo(algo, **kw, **algo_kwargs)
        env.close()
        print(f"    {arm:<12} {algo:<10} params={count_params(m)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--params-only", action="store_true")
    args = ap.parse_args()

    print(
        f"[phaseD-resmoke] AMENDMENT 1: {SEEDS} seeds x {STEPS} steps, "
        f"arms={[a[0] for a in _RESMOKE_ARMS]} -> {LOG_DIR / LABEL}",
        flush=True,
    )
    _params_only()
    if args.params_only:
        return 0

    jobs = _build_jobs(device=args.device)
    if args.dry_run:
        for j in jobs:
            print(f"  {j['run_id']:<40} algo={j['algo']:<9} env_kwargs={j['env_kwargs']}")
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / LABEL).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(phaseD._run_mpe_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            r = fut.result()
            rows.append(r)
            print(
                f"  [{i}/{len(jobs)}] {r['arm']:<12} seed{r['seed']}: "
                f"final={r['final']:.2f} ({time.monotonic() - t0:.0f}s)",
                flush=True,
            )

    manifest = LOG_DIR / f"manifest_{LABEL}.csv"
    df = pd.DataFrame(rows)
    df.sort_values(["arm", "seed"]).to_csv(manifest, index=False)
    print(f"[phaseD-resmoke] complete -> {manifest}")
    for arm, *_ in _RESMOKE_ARMS:
        sub = df[df.arm == arm]["final"]
        print(f"  {arm:<12} {sub.mean():8.2f}  (sd {sub.std():.2f}, n={len(sub)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
