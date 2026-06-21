"""Stage A gamble: does the graph help when coordination is irreducibly local?

New, pre-registered hypothesis (NOT a refit of the observability one). The
earlier null was explained by CoordGrid being solvable with a graph-free global
convention. Here we remove that escape hatch with the goal-routing task: each
episode samples a secret goal cell that only the *source* agent observes, and
reward = number of agents co-located at the goal. Non-source agents can only
learn the goal if it is *routed* to them over the coordination graph -- so the
GNN has a channel the no-graph controls structurally lack, although every
algorithm receives identical observations.

H7 (routing): on the goal-routing task, GNN-QMIX (repaired: residual+layernorm)
beats both QMIX and the parameter-matched MLP-QMIX, because only message passing
can deliver the privately-observed goal to the rest of the team.

Verdict signal: A = mean(gnn_repair) - max(mean(QMIX), mean(MLP-QMIX)) > 0.

Usage::

    python scripts/phaseA_goal_routing.py --steps 30000 --seeds 3
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

CELLS = (
    ("ring", 4, 5, "ring-N4-d2"),
    ("line", 6, 7, "line-N6-d5"),
)
# (run-name, algo, extra algo_kwargs)
ARMS = (
    ("qmix", "qmix", {}),
    ("mlp_qmix", "mlp_qmix", {}),
    ("gnn_repair", "gnn_qmix", {"gnn_layers": 2, "gnn_residual": True, "gnn_layernorm": True}),
)


def final_window_return(run_dir: Path, frac: float = 0.2) -> float:
    df = pd.read_csv(Path(run_dir) / "episodes.csv")
    k = max(1, int(len(df) * frac))
    return float(df["return"].tail(k).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--log-dir", type=Path, default=REPO_ROOT / "results" / "phaseA_goal")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    res: dict[tuple[str, str], list[float]] = {}
    t0 = time.monotonic()
    n_jobs = len(CELLS) * len(ARMS) * len(seeds)
    done = 0
    print(f"[goal] {n_jobs} runs x {args.steps} steps -> {args.log_dir}", flush=True)

    for graph, n_agents, grid_size, label in CELLS:
        for arm_name, algo, extra in ARMS:
            vals: list[float] = []
            for seed in seeds:
                cfg = TrainConfig(
                    env="coord_grid",
                    env_kwargs={
                        "n_agents": n_agents,
                        "grid_size": grid_size,
                        "episode_steps": 25,
                        "graph": graph,
                        "goal_routing": True,
                    },
                    algo=algo,
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
                    log_dir=args.log_dir / label,
                    run_id=f"{arm_name}__{graph}__N{n_agents}__seed{seed}",
                )
                vals.append(final_window_return(train(cfg)))
                done += 1
            res[(label, arm_name)] = vals
            print(
                f"  [{done}/{n_jobs}] {label}/{arm_name}: "
                f"mean={np.mean(vals):.3f} sd={np.std(vals):.3f} "
                f"({time.monotonic() - t0:.0f}s)",
                flush=True,
            )

    print("\n=== goal-routing: final-window mean return (chance ~= N*ep_len/grid^2) ===", flush=True)
    win = False
    for graph, n_agents, grid_size, label in CELLS:
        chance = n_agents * 25 / (grid_size ** 2)
        q = np.mean(res[(label, "qmix")])
        m = np.mean(res[(label, "mlp_qmix")])
        g = np.mean(res[(label, "gnn_repair")])
        adv = g - max(q, m)
        cell_win = adv > 0
        win = win or cell_win
        print(
            f"  [{label}] chance~{chance:.1f} | qmix={q:.2f} mlp={m:.2f} "
            f"gnn_repair={g:.2f}  A={adv:+.2f}  "
            f"{'<<< GRAPH WINS (routing helps)' if cell_win else ''}",
            flush=True,
        )

    print(
        "\nVERDICT: "
        + (
            "the graph beats the no-graph controls on the routing task -- a "
            "genuine positive. Graphs help when coordination cannot be reduced "
            "to a global convention. Proceed to Stage B (phase diagram)."
            if win
            else "even on a routing task built to need it, the graph does not "
            "beat the controls. Strong evidence the negative result is "
            "fundamental; fall back to the strengthened negative paper."
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
