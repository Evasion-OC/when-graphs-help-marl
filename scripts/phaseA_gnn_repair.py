"""Stage A follow-up: can a *repaired* GCN leave the floor (and benefit from ego)?

The ego smoke showed GNN-QMIX pinned near the random floor (~5) in every cell,
i.e. it barely trains -- so "the graph doesn't help" was not yet a fair test.
This diagnostic gives the graph two anti-over-smoothing repairs and asks whether
a GNN that *actually trains* (a) reaches the no-graph controls and (b) gains
anything from restricted observability.

Arms (GNN only -- QMIX / MLP-QMIX references are read back from the smoke run):
  * gnn_L1     : plain GCN, depth 1 (minimal over-smoothing).
  * gnn_repair : depth 2 GCN + residual skips + LayerNorm (GCNII-flavoured).

Cells/modes/seeds match scripts/phaseA_obs_smoke.py so the references line up.

Usage::

    python scripts/phaseA_gnn_repair.py --steps 20000 --seeds 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnmarl.training import TrainConfig, train  # noqa: E402

MODES = ("full", "ego")
CELLS = (
    ("ring", 4, 5, "ring-N4-d2"),
    ("line", 6, 7, "line-N6-d5"),
)
# New GNN arms: (run-name, extra algo_kwargs).
GNN_ARMS = (
    ("gnn_L1", {"gnn_layers": 1}),
    ("gnn_repair", {"gnn_layers": 2, "gnn_residual": True, "gnn_layernorm": True}),
)
SMOKE_DIR = REPO_ROOT / "results" / "phaseA_smoke"


def final_window_return(csv: Path, frac: float = 0.2) -> float:
    df = pd.read_csv(csv)
    k = max(1, int(len(df) * frac))
    return float(df["return"].tail(k).mean())


def ref_mean(label: str, mode: str, algo: str, graph: str, n: int, seeds: list[int]) -> float | None:
    """Read a reference algo's mean final-window return back from the smoke run."""
    vals = []
    for seed in seeds:
        csv = SMOKE_DIR / label / mode / f"{algo}__{graph}__N{n}__{mode}__seed{seed}" / "episodes.csv"
        if csv.exists():
            vals.append(final_window_return(csv))
    return float(np.mean(vals)) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--log-dir", type=Path, default=REPO_ROOT / "results" / "phaseA_repair")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    new: dict[tuple[str, str, str], list[float]] = {}
    t0 = time.monotonic()
    n_jobs = len(CELLS) * len(MODES) * len(GNN_ARMS) * len(seeds)
    done = 0
    print(f"[repair] {n_jobs} GNN runs x {args.steps} steps -> {args.log_dir}", flush=True)

    for graph, n_agents, grid_size, label in CELLS:
        for mode in MODES:
            for arm_name, extra in GNN_ARMS:
                vals: list[float] = []
                for seed in seeds:
                    cfg = TrainConfig(
                        env="coord_grid",
                        env_kwargs={
                            "n_agents": n_agents,
                            "grid_size": grid_size,
                            "episode_steps": 25,
                            "graph": graph,
                            "obs_mode": mode,
                        },
                        algo="gnn_qmix",
                        algo_kwargs={
                            "buffer_capacity": 50_000,
                            "batch_size": 32,
                            "target_update_interval": 200,
                            "lr": 5e-4,
                            "gamma": 0.99,
                            "gnn_hidden": 64,
                            **extra,
                        },
                        total_env_steps=args.steps,
                        eps_start=1.0,
                        eps_end=0.05,
                        eps_anneal_steps=int(args.steps * 0.8),
                        seed=seed,
                        log_dir=args.log_dir / label / mode,
                        run_id=f"{arm_name}__{graph}__N{n_agents}__{mode}__seed{seed}",
                    )
                    vals.append(final_window_return(Path(train(cfg)) / "episodes.csv"))
                    done += 1
                new[(label, mode, arm_name)] = vals
                print(
                    f"  [{done}/{n_jobs}] {label}/{mode}/{arm_name}: "
                    f"mean={np.mean(vals):.3f} sd={np.std(vals):.3f} "
                    f"({time.monotonic() - t0:.0f}s)",
                    flush=True,
                )

    print("\n=== COMBINED: final-window mean return (refs from smoke) ===", flush=True)
    any_repair_competitive = False
    for graph, n_agents, _, label in CELLS:
        print(f"  [{label}]", flush=True)
        for mode in MODES:
            qmix = ref_mean(label, mode, "qmix", graph, n_agents, seeds)
            mlp = ref_mean(label, mode, "mlp_qmix", graph, n_agents, seeds)
            gnn_floor = ref_mean(label, mode, "gnn_qmix", graph, n_agents, seeds)
            l1 = np.mean(new[(label, mode, "gnn_L1")])
            rep = np.mean(new[(label, mode, "gnn_repair")])
            ctrl = max(x for x in (qmix, mlp) if x is not None) if (qmix or mlp) else float("nan")
            adv = rep - ctrl
            left_floor = (gnn_floor is not None) and (rep > gnn_floor + 5)
            competitive = adv > 0
            any_repair_competitive = any_repair_competitive or competitive
            print(
                f"    {mode:4s}: qmix={fmt(qmix)} mlp={fmt(mlp)} "
                f"gnn(L2plain)={fmt(gnn_floor)} | gnn_L1={l1:.2f} "
                f"gnn_repair={rep:.2f}  A_repair={adv:+.2f}  "
                f"{'[left floor]' if left_floor else '[still floored]'}"
                f"{'  <<< BEATS CONTROL' if competitive else ''}",
                flush=True,
            )

    print(
        "\nVERDICT: "
        + (
            "a repaired GNN beats the no-graph control in at least one cell -- "
            "the earlier null was partly a broken-model artefact; pursue Stage B "
            "with the repaired architecture."
            if any_repair_competitive
            else "even repaired + trained, the GNN does not beat the matched "
            "no-graph control. The negative result is robust to model quality; "
            "the honest contribution is the controlled null."
        ),
        flush=True,
    )
    return 0


def fmt(x: float | None) -> str:
    return "  n/a" if x is None else f"{x:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
