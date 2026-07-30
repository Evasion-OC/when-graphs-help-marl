"""Stage C -- the classical coordination-graph (DCG) baseline sweep.

Implements the three pre-registered cells from
``docs/CLASSICAL_CG_BASELINE_DESIGN.md``: does a deep coordination graph
(DCG; Kok & Vlassis max-plus, function-approximated per Boehmer, Kurin &
Whiteson 2020) subsume the GNN's positive results, or is there a cell it
structurally cannot reach?

  A  full-info edge-reward ring   -- validity + honest dissociation.
     ``graph=ring``, ``obs_mode=full``, N=4. Arms: mlp, gnn_true (existing
     Stage-C core-suite arms) + dcg_jointobs, dcg_canonical.
  B  PairRouting-ego               -- the subsumption check (1-hop).
     ``pair_routing=True``, ``obs_mode=ego``, N=6. Arms: dcg_jointobs on
     true/wrong/complete comm graphs + dcg_canonical_true. Design section 7
     says reuse the existing ``gnn_true`` 150k run rather than re-run it --
     but ``phaseC_analysis.py`` reads ONE manifest file and hard-requires an
     ``arm=="gnn_true"`` row (its whole KEY contrast is ``gnn_true`` vs every
     other arm), and does NOT concatenate manifests itself. So cell B's
     DCG-only manifest is NOT directly analyzable as emitted -- run
     ``--merge-gnn-true-from PATH`` (see below) AFTER cell B finishes to
     splice the historical ``gnn_true`` rows in before calling
     ``phaseC_analysis.py``.
  C  relay_routing (NEW env mode)  -- the payload; the only cell that meets
     the referee's bar (design section 0). ``relay_routing=True``,
     ``obs_mode=ego``, N=6. Arms: mlp, gnn_true (2-layer) + dcg_jointobs on
     true/wrong/complete + dcg_canonical_true (+ gat_true, optional per
     design, on by default here -- disable with ``--no-gat``).

Usage::

    python scripts/phaseC_classical_sweep.py --smoke                  # all 3 cells, 3 seeds
    python scripts/phaseC_classical_sweep.py --cell C --smoke         # cell C only, smoke
    python scripts/phaseC_classical_sweep.py --cell A,B,C             # full 150k x 12-seed sweep
    python scripts/phaseC_classical_sweep.py --cell C --steps 150000 --seeds 12
    python scripts/phaseC_classical_sweep.py --dry-run                # print the job list only

    # After a cell-B run, splice in the historical gnn_true 150k rows so
    # phaseC_analysis.py's gnn_true-vs-every-arm contrast is computable:
    python scripts/phaseC_classical_sweep.py --merge-gnn-true-from \\
        results/phaseC/manifest_pair_N6_ego.csv \\
        --merge-gnn-true-into results/phaseC_classical/manifest_cellB_N6_ego_150000steps.csv
    python scripts/phaseC_analysis.py --log-dir results/phaseC_classical \\
        --label cellB_N6_ego_150000steps

Go/no-go order (design section 4): run ``--cell C --smoke`` FIRST (the
kill-switch -- if ``gnn_true`` floors on relay there is no differentiation to
claim and the confirmatory DCG relay sweep should not be run), then
``--cell B --smoke`` (expect ``dcg_jointobs_true`` to match/beat ``gnn_true``
-- also validates the DCG implementation), then ``--cell A --smoke``
(home-turf sanity).
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

_REPAIRED = {"gnn_residual": True, "gnn_layernorm": True}

# Per-cell env base config: n_agents, grid_size, episode_steps, obs_mode,
# routing flag name (None for cell A -- the default co-location reward
# doesn't take a routing flag), true/wrong comm graphs.
CELLS: dict[str, dict[str, Any]] = {
    "A": {
        "n_agents": 4, "grid_size": 5, "episode_steps": 25, "obs_mode": "full",
        "routing_flag": None, "true_graph": "ring", "wrong_graph": None,
        "role": "full-info edge-reward ring (validity + honest dissociation)",
    },
    "B": {
        "n_agents": 6, "grid_size": 3, "episode_steps": 25, "obs_mode": "ego",
        "routing_flag": "pair_routing", "true_graph": "matching", "wrong_graph": "matching_wrong",
        "role": "PairRouting-ego (subsumption check, 1-hop)",
    },
    "C": {
        "n_agents": 6, "grid_size": 3, "episode_steps": 25, "obs_mode": "ego",
        "routing_flag": "relay_routing", "true_graph": "relay", "wrong_graph": "relay_wrong",
        "role": "2-hop relay (the payload, NEW env mode)",
    },
}


def arms(cell: str, *, include_gat: bool) -> tuple[tuple[str, str, str, dict], ...]:
    """(arm_name, algo, env_graph, extra algo_kwargs) for one cell."""
    c = CELLS[cell]
    true_g, wrong_g = c["true_graph"], c["wrong_graph"]

    if cell == "A":
        # No true/wrong/complete structure variable here -- graph=ring is
        # fixed; DCG's own structure variable is the CONDITIONING arm.
        return (
            ("mlp", "mlp_qmix", true_g, {"gnn_layers": 2}),
            ("gnn_true", "gnn_qmix", true_g, {"gnn_layers": 2, **_REPAIRED}),
            ("dcg_jointobs", "dcg", true_g, {"conditioning": "jointobs"}),
            ("dcg_canonical", "dcg", true_g, {"conditioning": "canonical"}),
        )
    if cell == "B":
        return (
            ("dcg_jointobs_true", "dcg", true_g, {"conditioning": "jointobs"}),
            ("dcg_jointobs_wrong", "dcg", wrong_g, {"conditioning": "jointobs"}),
            ("dcg_jointobs_complete", "dcg", "complete", {"conditioning": "jointobs"}),
            ("dcg_canonical_true", "dcg", true_g, {"conditioning": "canonical"}),
        )
    if cell == "C":
        out = [
            ("mlp", "mlp_qmix", true_g, {"gnn_layers": 2}),
            ("gnn_true", "gnn_qmix", true_g, {"gnn_layers": 2, **_REPAIRED}),
            ("dcg_jointobs_true", "dcg", true_g, {"conditioning": "jointobs"}),
            ("dcg_jointobs_wrong", "dcg", wrong_g, {"conditioning": "jointobs"}),
            ("dcg_jointobs_complete", "dcg", "complete", {"conditioning": "jointobs"}),
            ("dcg_canonical_true", "dcg", true_g, {"conditioning": "canonical"}),
        ]
        if include_gat:
            out.append(("gat_true", "gat_qmix", true_g, {"gnn_layers": 2, **_REPAIRED}))
        return tuple(out)
    raise ValueError(f"unknown cell {cell!r}")


def _env_kwargs(cell: str, env_graph: str) -> dict[str, Any]:
    c = CELLS[cell]
    n = c["n_agents"]
    kw: dict[str, Any] = {
        "n_agents": n,
        "grid_size": c["grid_size"],
        "episode_steps": c["episode_steps"],
        "graph": env_graph,
        "obs_mode": c["obs_mode"],
        # Constant obs_dim across true/wrong/complete comm graphs (design
        # section 3 "fairness/matched protocol").
        "max_neighbors_override": n - 1,
    }
    if c["routing_flag"] is not None:
        kw[c["routing_flag"]] = True
    return kw


def _run_one(job: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    from gnnmarl.training import TrainConfig, train

    steps = job["steps"]
    cfg = TrainConfig(
        env="coord_grid",
        env_kwargs=_env_kwargs(job["cell"], job["env_graph"]),
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
        "cell": job["cell"], "task": job["cell"], "suite": "classical", "obs": job["obs"],
        "n_agents": job["n_agents"], "seed": job["seed"], "run_dir": str(run_dir),
        "steps": steps, "n_episodes": int(len(ret)),
        "final": float(ret[-k:].mean()),
    }


def _param_audit(cell: str, include_gat: bool) -> None:
    """Print DCG vs GNN-QMIX param counts for this cell's config (design
    section 3: report param counts; DCG must be >= GNN-QMIX -- generosity
    toward DCG)."""
    import torch

    from gnnmarl.algos import make_algo
    from gnnmarl.algos.maxplus import count_params
    from gnnmarl.envs import make_env

    c = CELLS[cell]
    env = make_env(
        "coord_grid",
        **_env_kwargs(cell, c["true_graph"]),
    )
    kw = dict(
        n_agents=env.n_agents, obs_dim=env.obs_dim, state_dim=env.state_dim,
        n_actions=env.n_actions, device=torch.device("cpu"), seed=0,
    )
    print(f"  [param audit] cell {cell}: obs_dim={env.obs_dim} state_dim={env.state_dim} "
          f"n_agents={env.n_agents}")
    for arm, algo, _env_graph, extra in arms(cell, include_gat=include_gat):
        algo_kwargs = {"gnn_hidden": 64, **extra}
        m = make_algo(algo, **kw, **algo_kwargs)
        print(f"    {arm:<24} {algo:<10} params={count_params(m)}")


def merge_gnn_true(cell_b_manifest: Path, external_manifest: Path) -> Path:
    """Splice the historical ``gnn_true`` rows into a cell-B DCG manifest.

    Design section 7: cell B reuses the existing ``gnn_true`` 150k run
    rather than re-running it. But ``phaseC_analysis.py`` reads exactly ONE
    manifest CSV and hard-requires an ``arm=="gnn_true"`` row (its KEY
    contrast is ``gnn_true`` vs every other arm) -- it does not concatenate
    manifests itself. This performs that one-time splice: filters
    ``external_manifest`` to ``arm=="gnn_true"``, aligns columns to the
    cell-B manifest's schema (older manifests may be missing columns this
    script added, e.g. ``steps``/``n_episodes``/``cell`` -- filled with
    ``None``, which ``phaseC_analysis.py`` never reads for those extra
    columns), and overwrites ``cell_b_manifest`` in place with the union.
    Idempotent: re-running drops any previously-spliced ``gnn_true`` rows
    first, so calling it twice does not duplicate them.

    Returns the (overwritten) ``cell_b_manifest`` path.
    """
    b = pd.read_csv(cell_b_manifest)
    ext = pd.read_csv(external_manifest)
    gnn_rows = ext[ext["arm"] == "gnn_true"].copy()
    if gnn_rows.empty:
        raise ValueError(f"no arm=='gnn_true' rows found in {external_manifest}")

    b = b[b["arm"] != "gnn_true"]  # idempotent: drop any prior splice first
    for col in b.columns:
        if col not in gnn_rows.columns:
            gnn_rows[col] = None
    gnn_rows = gnn_rows[b.columns]

    merged = pd.concat([b, gnn_rows], ignore_index=True)
    merged.sort_values(["arm", "seed"]).to_csv(cell_b_manifest, index=False)
    print(f"[phaseC_classical] merged {len(gnn_rows)} gnn_true rows from "
          f"{external_manifest} -> {cell_b_manifest} ({len(merged)} total rows)")
    return cell_b_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=str, default="A,B,C",
                     help="comma-separated subset of {A,B,C}")
    ap.add_argument("--steps", type=int, default=None,
                     help="env steps per run (default: 3000 with --smoke, else 150000)")
    ap.add_argument("--seeds", type=int, default=None,
                     help="seed count (default: 3 with --smoke, else 12)")
    ap.add_argument("--smoke", action="store_true",
                     help="go/no-go smoke: 3 seeds, short budget (design section 4)")
    ap.add_argument("--no-gat", dest="gat", action="store_false", default=True,
                     help="drop the optional gat_true arm from cell C")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--log-dir", type=Path, default=REPO_ROOT / "results" / "phaseC_classical")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--params-only", action="store_true",
                     help="print the param audit for each requested cell and exit")
    ap.add_argument("--merge-gnn-true-into", type=Path, default=None,
                     help="post-process ONLY: splice historical gnn_true rows (from "
                          "--merge-gnn-true-from) into this existing cell-B manifest CSV "
                          "in place, then exit. No training is run.")
    ap.add_argument("--merge-gnn-true-from", type=Path, default=None,
                     help="existing phaseC_structure_sweep.py manifest to pull the "
                          "arm=='gnn_true' rows from (e.g. results/phaseC/"
                          "manifest_pair_N6_ego.csv)")
    args = ap.parse_args()

    if args.merge_gnn_true_into is not None:
        if args.merge_gnn_true_from is None:
            ap.error("--merge-gnn-true-into requires --merge-gnn-true-from")
        merge_gnn_true(args.merge_gnn_true_into, args.merge_gnn_true_from)
        return 0

    cells = [c.strip().upper() for c in args.cell.split(",") if c.strip()]
    for c in cells:
        if c not in CELLS:
            ap.error(f"unknown cell {c!r}; expected one of {sorted(CELLS)}")

    steps = args.steps if args.steps is not None else (3_000 if args.smoke else 150_000)
    seeds = args.seeds if args.seeds is not None else (3 if args.smoke else 12)
    budget_tag = "smoke" if args.smoke and args.steps is None else f"{steps}steps"

    print(f"[phaseC_classical] cells={cells} steps={steps} seeds={seeds} "
          f"budget_tag={budget_tag} workers={args.workers} device={args.device}", flush=True)

    for cell in cells:
        print(f"\n=== cell {cell}: {CELLS[cell]['role']} ===")
        _param_audit(cell, include_gat=args.gat)
    if args.params_only:
        return 0

    jobs: list[dict] = []
    for cell in cells:
        c = CELLS[cell]
        obs_mode = c["obs_mode"]
        n_agents = c["n_agents"]
        label = f"cell{cell}_N{n_agents}_{obs_mode}_{budget_tag}"
        for arm, algo, env_graph, extra in arms(cell, include_gat=args.gat):
            for seed in range(seeds):
                jobs.append({
                    "arm": arm, "algo": algo, "env_graph": env_graph, "extra": extra,
                    "cell": cell, "obs": obs_mode, "n_agents": n_agents, "seed": seed,
                    "steps": steps, "device": args.device,
                    "log_dir": str(args.log_dir / label),
                    "run_id": f"{arm}__{label}__seed{seed}",
                    "label": label,
                })

    print(f"\n[phaseC_classical] {len(jobs)} total runs", flush=True)
    if args.dry_run:
        for j in jobs:
            print(f"  {j['run_id']:<48} algo={j['algo']:<9} graph={j['env_graph']}")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    for cell in cells:
        c = CELLS[cell]
        label = f"cell{cell}_N{c['n_agents']}_{c['obs_mode']}_{budget_tag}"
        cell_jobs = [j for j in jobs if j["label"] == label]
        rows: list[dict] = []
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_run_one, j) for j in cell_jobs]
            for fut in as_completed(futs):
                r = fut.result()
                rows.append(r)
                done += 1
                print(f"  [{cell} {done}/{len(cell_jobs)}] {r['arm']:<22} seed{r['seed']}: "
                      f"final={r['final']:.2f} ({time.monotonic() - t0:.0f}s)", flush=True)

        manifest = args.log_dir / f"manifest_{label}.csv"
        df = pd.DataFrame(rows)
        df.sort_values(["arm", "seed"]).to_csv(manifest, index=False)
        print(f"\n[phaseC_classical] cell {cell} complete -> {manifest}")
        print(f"=== cell {cell} mean final return by arm ===")
        for arm, *_ in arms(cell, include_gat=args.gat):
            sub = df[df.arm == arm]["final"]
            print(f"  {arm:<22} {sub.mean():7.2f}  (sd {sub.std():.2f}, n={len(sub)})")

    print(f"\n[phaseC_classical] all cells complete in {time.monotonic() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
