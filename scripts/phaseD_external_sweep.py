"""Phase D -- external-validity sweep on MPE ``simple_reference`` pairs.

Implements the pre-approved design in ``docs/EXTERNAL_VALIDITY_DESIGN.md``:
run the isolator family (``gnn_true`` / ``gnn_wrong`` / ``gnn_complete`` /
``mlp`` / ``oracle``) on K composed ``simple_reference`` pairs with the
scripted communication channel disabled
(``gnnmarl.envs.mpe_reference_pairs.MPEReferencePairs``), modeled on
``scripts/phaseC_classical_sweep.py``'s cell/arm/manifest conventions.

  E0  N=2 honesty anchor      -- ``k=1``, ``comm_on=True``, unmodified
      simple_reference. Arms: ``gnn`` vs ``mlp``. 3-6 seeds, NON-confirmatory
      (design section 8): expected TIE, reported as an honesty anchor, no
      headline gated on it.
  E1  primary confirmatory    -- ``k=3`` (N=6), comm disabled. Arms:
      ``gnn_true`` / ``gnn_wrong`` / ``gnn_complete`` / ``mlp`` / ``oracle``.
      This is the vehicle's headline cell (design section 3).
  E2  K=5 escalation          -- pre-specified, implemented here but NOT part
      of the default ``--cell`` selection. Only run on the frozen Branch-2a
      trigger (design section 10.3) -- pass ``--cell E2`` explicitly.
  R   relay-cell addendum     -- adds ``gnn_wrong`` + ``gnn_complete`` to the
      EXISTING ``coord_grid`` ``relay_routing`` cell C
      (``results/phaseC_classical/``), 12 seeds, 150k steps FROZEN (design
      section 11) regardless of ``--steps``/``--seeds``. Reuses
      ``phaseC_classical_sweep.py``'s ``CELLS["C"]`` / ``_env_kwargs`` /
      ``_run_one`` so the env config is byte-identical to the existing cell-C
      runs -- the emitted manifest is schema- and cell-label-compatible with
      ``results/phaseC_classical/manifest_cellC_N6_ego_150000steps.csv`` and
      can be concatenated directly (``--merge-into-cellc`` does this).

``gnn_true`` / ``gnn_wrong`` / ``gnn_complete`` all use the *stabilised* GCN
config (``gnn_residual=True, gnn_layernorm=True``), matching
``phaseC_classical_sweep.py``'s ``_REPAIRED`` and the paper's positive
endpoint -- so ``gnn_true`` is NOT byte-identical to ``mlp``: the LayerNorm
adds ``2 layers x 2 x gnn_hidden = 256`` params at ``gnn_hidden=64`` (see
design section 3's disclosure requirement; verify with ``--params-only``).

The ``oracle`` arm is ``mlp_qmix`` on ``oracle=True`` env kwargs (obs_dim 24):
the no-graph architecture handed the routed answer directly, so
``oracle - mlp`` isolates the value of the routed information at matched
architecture, and the headroom denominator ``(arm - mlp)/(oracle - mlp)``
in the design's protocol (section 4) is well-defined. This reading is an
experimental-contract decision -- flagged for the research scientist /
statistician to confirm, not implied by the design doc's prose alone.

Budget is deliberately NOT frozen for E0/E1/E2 (design section 4: "calibrated
in smoke, then frozen in the pre-reg before confirmatory") -- the defaults
below are placeholders, override with ``--steps``/``--seeds``. Cell R's
budget IS frozen (12 seeds / 150k) per design section 11.

Usage::

    python scripts/phaseD_external_sweep.py --smoke                # E0,E1 smoke
    python scripts/phaseD_external_sweep.py --cell E1 --smoke
    python scripts/phaseD_external_sweep.py --cell E2               # escalation only
    python scripts/phaseD_external_sweep.py --cell R --smoke        # relay addendum smoke
    python scripts/phaseD_external_sweep.py --cell R                # frozen 12x150k
    python scripts/phaseD_external_sweep.py --params-only --cell E1,R
    python scripts/phaseD_external_sweep.py --dry-run --cell E0,E1,E2,R

    # After cell R's confirmatory run, splice its gnn_wrong/gnn_complete rows
    # into the existing cell-C manifest so phaseC_analysis.py's gnn_true-vs-
    # every-arm contrast sees them:
    python scripts/phaseD_external_sweep.py --merge-into-cellc \\
        results/phaseC_classical/manifest_cellC_N6_ego_150000steps.csv \\
        --merge-from results/phaseC_classical/manifest_cellR_N6_ego_150000steps.csv
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

import phaseC_classical_sweep as classical  # noqa: E402  (sibling script; cell-R reuse)

# Stabilised GCN config (matches phaseC_classical_sweep._REPAIRED / the
# positive endpoint). NOT byte-identical to mlp -- see module docstring.
_REPAIRED = {"gnn_residual": True, "gnn_layernorm": True}

MPE_CELLS: dict[str, dict[str, Any]] = {
    "E0": {
        "k": 1,
        "comm_on": True,
        "seeds_default": 6,
        "confirmatory": False,
        "role": "N=2 honesty anchor (comm ON, unmodified simple_reference) -- non-confirmatory",
    },
    "E1": {
        "k": 3,
        "comm_on": False,
        "seeds_default": 12,
        "confirmatory": True,
        "role": "primary confirmatory cell (K=3, N=6, comm disabled)",
    },
    "E2": {
        "k": 5,
        "comm_on": False,
        "seeds_default": 12,
        "confirmatory": True,
        "role": "K=5 escalation (pre-specified; run only on the frozen 2a trigger)",
    },
}


def _mpe_arms(cell: str) -> tuple[tuple[str, str, str, dict, dict], ...]:
    """(arm_name, algo, graph, algo_extra_kwargs, env_extra_kwargs) for one MPE cell."""
    if cell == "E0":
        return (
            ("mlp", "mlp_qmix", "true", {"gnn_layers": 2}, {}),
            ("gnn", "gnn_qmix", "true", {"gnn_layers": 2, **_REPAIRED}, {}),
        )
    if cell in ("E1", "E2"):
        return (
            ("mlp", "mlp_qmix", "true", {"gnn_layers": 2}, {}),
            ("gnn_true", "gnn_qmix", "true", {"gnn_layers": 2, **_REPAIRED}, {}),
            ("gnn_wrong", "gnn_qmix", "wrong", {"gnn_layers": 2, **_REPAIRED}, {}),
            ("gnn_complete", "gnn_qmix", "complete", {"gnn_layers": 2, **_REPAIRED}, {}),
            ("oracle", "mlp_qmix", "true", {"gnn_layers": 2}, {"oracle": True}),
        )
    raise ValueError(f"unknown MPE cell {cell!r}")


def _mpe_env_kwargs(cell: str, graph: str, env_extra: dict) -> dict[str, Any]:
    c = MPE_CELLS[cell]
    kw: dict[str, Any] = {"k": c["k"], "graph": graph, "comm_on": c["comm_on"], "max_cycles": 25}
    kw.update(env_extra)
    return kw


def _run_mpe_job(job: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    from gnnmarl.training import TrainConfig, train

    steps = job["steps"]
    cfg = TrainConfig(
        env="mpe_reference_pairs",
        env_kwargs=job["env_kwargs"],
        algo=job["algo"],
        algo_kwargs={
            "buffer_capacity": 50_000,
            "batch_size": 32,
            "target_update_interval": 200,
            "lr": 5e-4,
            "gamma": 0.99,
            "gnn_hidden": 64,
            **job["algo_extra"],
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
        "graph": job["graph"],
        "cell": job["cell"],
        "task": "mpe_reference_pairs",
        "suite": "external_validity",
        "k": job["env_kwargs"]["k"],
        "n_agents": job["n_agents"],
        "seed": job["seed"],
        "run_dir": str(run_dir),
        "steps": steps,
        "n_episodes": int(len(ret)),
        "final": float(ret[-k:].mean()),
    }


def _param_audit_mpe(cell: str) -> None:
    """Print gnn/mlp/oracle param counts for this cell's config (deliverable
    5: gnn vs mlp vs oracle param counts at K=3)."""
    import torch

    from gnnmarl.algos import make_algo
    from gnnmarl.algos.maxplus import count_params
    from gnnmarl.envs import make_env

    printed_dims = set()
    for arm, algo, graph, algo_extra, env_extra in _mpe_arms(cell):
        env_kwargs = _mpe_env_kwargs(cell, graph, env_extra)
        env = make_env("mpe_reference_pairs", **env_kwargs)
        dims = (env.obs_dim, env.state_dim, env.n_agents, env.n_actions)
        if dims not in printed_dims:
            printed_dims.add(dims)
            print(
                f"  [param audit] cell {cell} arm={arm}: obs_dim={env.obs_dim} "
                f"state_dim={env.state_dim} n_agents={env.n_agents} n_actions={env.n_actions}"
            )
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
        print(f"    {arm:<14} {algo:<10} graph={graph:<9} params={count_params(m)}")


def _run_mpe_cell(
    cell: str,
    *,
    steps: int,
    seeds: int,
    budget_tag: str,
    log_dir: Path,
    device: str,
    workers: int,
    dry_run: bool,
) -> list[dict]:
    c = MPE_CELLS[cell]
    label = f"cell{cell}_N{2 * c['k']}_{budget_tag}"
    jobs: list[dict] = []
    for arm, algo, graph, algo_extra, env_extra in _mpe_arms(cell):
        env_kwargs = _mpe_env_kwargs(cell, graph, env_extra)
        for seed in range(seeds):
            jobs.append(
                {
                    "arm": arm,
                    "algo": algo,
                    "graph": graph,
                    "algo_extra": algo_extra,
                    "env_kwargs": env_kwargs,
                    "cell": cell,
                    "n_agents": 2 * c["k"],
                    "seed": seed,
                    "steps": steps,
                    "device": device,
                    "log_dir": str(log_dir / label),
                    "run_id": f"{arm}__{label}__seed{seed}",
                }
            )

    print(
        f"\n[phaseD] cell {cell} ({c['role']}): {len(jobs)} runs -> {log_dir / label}", flush=True
    )
    if dry_run:
        for j in jobs:
            print(f"  {j['run_id']:<40} algo={j['algo']:<9} graph={j['graph']}")
        return []

    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_mpe_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            r = fut.result()
            rows.append(r)
            print(
                f"  [{cell} {i}/{len(jobs)}] {r['arm']:<14} seed{r['seed']}: "
                f"final={r['final']:.2f} ({time.monotonic() - t0:.0f}s)",
                flush=True,
            )

    manifest = log_dir / f"manifest_{label}.csv"
    df = pd.DataFrame(rows)
    df.sort_values(["arm", "seed"]).to_csv(manifest, index=False)
    print(f"[phaseD] cell {cell} complete -> {manifest}")
    for arm, *_ in _mpe_arms(cell):
        sub = df[df.arm == arm]["final"]
        print(f"  {arm:<14} {sub.mean():8.2f}  (sd {sub.std():.2f}, n={len(sub)})")
    return rows


# --------------------------------------------------------------------- cell R


_CELL_R_ARMS = (
    ("gnn_wrong", "gnn_qmix", "relay_wrong", {"gnn_layers": 2, **_REPAIRED}),
    ("gnn_complete", "gnn_qmix", "complete", {"gnn_layers": 2, **_REPAIRED}),
)
_CELL_R_STEPS = 150_000
_CELL_R_SEEDS = 12


def _run_cell_r(
    *, steps: int, seeds: int, budget_tag: str, device: str, workers: int, dry_run: bool
) -> list[dict]:
    """Relay-cell addendum: gnn_wrong + gnn_complete on the EXISTING coord_grid
    relay_routing cell C, reusing phaseC_classical_sweep's env config verbatim
    so the manifest is directly concatenable with cell C's (design section 11).
    """
    log_dir = REPO_ROOT / "results" / "phaseC_classical"
    label = f"cellR_N6_ego_{budget_tag}"
    jobs: list[dict] = []
    for arm, algo, env_graph, extra in _CELL_R_ARMS:
        for seed in range(seeds):
            jobs.append(
                {
                    "arm": arm,
                    "algo": algo,
                    "env_graph": env_graph,
                    "extra": extra,
                    "cell": "C",
                    "obs": "ego",
                    "n_agents": 6,
                    "seed": seed,
                    "steps": steps,
                    "device": device,
                    "log_dir": str(log_dir / label),
                    "run_id": f"{arm}__{label}__seed{seed}",
                }
            )

    print(
        f"\n[phaseD] cell R (relay addendum: {len(jobs)} runs, "
        f"steps={steps} seeds={seeds}) -> {log_dir / label}",
        flush=True,
    )
    if dry_run:
        for j in jobs:
            print(f"  {j['run_id']:<40} algo={j['algo']:<9} graph={j['env_graph']}")
        return []

    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(classical._run_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            r = fut.result()
            rows.append(r)
            print(
                f"  [R {i}/{len(jobs)}] {r['arm']:<14} seed{r['seed']}: "
                f"final={r['final']:.2f} ({time.monotonic() - t0:.0f}s)",
                flush=True,
            )

    manifest = log_dir / f"manifest_{label}.csv"
    df = pd.DataFrame(rows)
    df.sort_values(["arm", "seed"]).to_csv(manifest, index=False)
    print(f"[phaseD] cell R complete -> {manifest}")
    for arm, *_ in _CELL_R_ARMS:
        sub = df[df.arm == arm]["final"]
        print(f"  {arm:<14} {sub.mean():8.2f}  (sd {sub.std():.2f}, n={len(sub)})")
    return rows


def _param_audit_cell_r() -> None:
    import torch

    from gnnmarl.algos import make_algo
    from gnnmarl.algos.maxplus import count_params
    from gnnmarl.envs import make_env

    c = classical.CELLS["C"]
    e = make_env("coord_grid", **classical._env_kwargs("C", c["true_graph"]))
    print(
        f"  [param audit] cell R (extends cell C): obs_dim={e.obs_dim} "
        f"state_dim={e.state_dim} n_agents={e.n_agents}"
    )
    for arm, algo, _env_graph, extra in _CELL_R_ARMS:
        kw = dict(
            n_agents=e.n_agents,
            obs_dim=e.obs_dim,
            state_dim=e.state_dim,
            n_actions=e.n_actions,
            device=torch.device("cpu"),
            seed=0,
        )
        algo_kwargs = {"gnn_hidden": 64, **extra}
        m = make_algo(algo, **kw, **algo_kwargs)
        print(f"    {arm:<14} {algo:<10} params={count_params(m)}")


def merge_into_cellc(cellc_manifest: Path, cellr_manifest: Path) -> Path:
    """Splice cell R's gnn_wrong/gnn_complete rows into the cell-C manifest.

    Idempotent: drops any previously-spliced gnn_wrong/gnn_complete rows
    first (mirrors phaseC_classical_sweep.merge_gnn_true's convention).
    """
    c = pd.read_csv(cellc_manifest)
    r = pd.read_csv(cellr_manifest)
    r = r[r["arm"].isin(["gnn_wrong", "gnn_complete"])].copy()
    if r.empty:
        raise ValueError(f"no gnn_wrong/gnn_complete rows found in {cellr_manifest}")

    c = c[~c["arm"].isin(["gnn_wrong", "gnn_complete"])]  # idempotent
    for col in c.columns:
        if col not in r.columns:
            r[col] = None
    r = r[c.columns]

    merged = pd.concat([c, r], ignore_index=True)
    merged.sort_values(["arm", "seed"]).to_csv(cellc_manifest, index=False)
    print(
        f"[phaseD] merged {len(r)} rows from {cellr_manifest} -> {cellc_manifest} "
        f"({len(merged)} total rows)"
    )
    return cellc_manifest


# ------------------------------------------------------------------------ CLI


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cell",
        type=str,
        default="E0,E1",
        help="comma-separated subset of {E0,E1,E2,R}; E2 and R are opt-in "
        "(pre-specified escalation / addendum, not run by default)",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="env steps per run for E0/E1/E2 (default: 3000 with --smoke, "
        "else 150000; NOT frozen -- calibrate in smoke first per design "
        "section 4). Cell R ignores this outside --smoke (frozen 150k).",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="seed count for E0/E1/E2 (default: 3 with --smoke, else the "
        "cell's seeds_default). Cell R ignores this outside --smoke "
        "(frozen 12 seeds).",
    )
    ap.add_argument("--smoke", action="store_true", help="go/no-go smoke: 3 seeds, short budget")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=REPO_ROOT / "results" / "phaseD_external",
        help="ignored for cell R, which always writes to results/phaseC_classical/",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--params-only",
        action="store_true",
        help="print the param audit for each requested cell and exit",
    )
    ap.add_argument(
        "--merge-into-cellc",
        type=Path,
        default=None,
        help="post-process ONLY: splice cell R's gnn_wrong/gnn_complete rows "
        "(from --merge-from) into this existing cell-C manifest in "
        "place, then exit. No training is run.",
    )
    ap.add_argument(
        "--merge-from",
        type=Path,
        default=None,
        help="cell-R manifest to pull gnn_wrong/gnn_complete rows from",
    )
    args = ap.parse_args()

    if args.merge_into_cellc is not None:
        if args.merge_from is None:
            ap.error("--merge-into-cellc requires --merge-from")
        merge_into_cellc(args.merge_into_cellc, args.merge_from)
        return 0

    cells = [c.strip().upper() for c in args.cell.split(",") if c.strip()]
    valid = set(MPE_CELLS) | {"R"}
    for c in cells:
        if c not in valid:
            ap.error(f"unknown cell {c!r}; expected one of {sorted(valid)}")

    steps = args.steps if args.steps is not None else (3_000 if args.smoke else 150_000)
    seeds = args.seeds if args.seeds is not None else (3 if args.smoke else None)
    budget_tag = "smoke" if args.smoke and args.steps is None else f"{steps}steps"

    print(
        f"[phaseD] cells={cells} smoke={args.smoke} workers={args.workers} device={args.device}",
        flush=True,
    )

    for cell in cells:
        if cell == "R":
            _param_audit_cell_r()
        else:
            print(f"\n=== cell {cell}: {MPE_CELLS[cell]['role']} ===")
            _param_audit_mpe(cell)
    if args.params_only:
        return 0

    t0 = time.monotonic()
    for cell in cells:
        if cell == "R":
            r_steps = 3_000 if args.smoke else _CELL_R_STEPS
            r_seeds = 3 if args.smoke else _CELL_R_SEEDS
            r_budget_tag = "smoke" if args.smoke else f"{_CELL_R_STEPS}steps"
            _run_cell_r(
                steps=r_steps,
                seeds=r_seeds,
                budget_tag=r_budget_tag,
                device=args.device,
                workers=args.workers,
                dry_run=args.dry_run,
            )
        else:
            cell_seeds = seeds if seeds is not None else MPE_CELLS[cell]["seeds_default"]
            _run_mpe_cell(
                cell,
                steps=steps,
                seeds=cell_seeds,
                budget_tag=budget_tag,
                log_dir=args.log_dir,
                device=args.device,
                workers=args.workers,
                dry_run=args.dry_run,
            )

    print(f"\n[phaseD] all requested cells complete in {time.monotonic() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
