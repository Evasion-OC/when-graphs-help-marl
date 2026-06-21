"""Stage A smoke: does observability gate the graph's value? (pre-registered H4/H5)

Trains {QMIX, MLP-QMIX (param-matched no-graph control), GNN-QMIX} under
``obs_mode in {full, ego}`` on two CoordGrid cells:

  * ring N=4  (diameter 2)  -- the cell where the published null lives.
  * line N=6  (diameter 5)  -- where the graph's routing advantage should be
                              largest once observability is restricted.

Verdict (directional, smoke-grade -- NOT the final statistics):
  graph advantage  A(mode) = mean(GNN-QMIX) - max(mean(QMIX), mean(MLP-QMIX)).
H4 predicts A(ego) > 0; the existing result is A(full) <= 0. A sign flip
full->ego is the signal that the positive story is real.

Usage::

    python scripts/phaseA_obs_smoke.py --steps 20000 --seeds 3
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

ALGOS = ("qmix", "mlp_qmix", "gnn_qmix")
MODES = ("full", "ego")
# (graph, n_agents, grid_size, label)
CELLS = (
    ("ring", 4, 5, "ring-N4-d2"),
    ("line", 6, 7, "line-N6-d5"),
)


def final_window_return(run_dir: Path, frac: float = 0.2) -> float:
    df = pd.read_csv(Path(run_dir) / "episodes.csv")
    k = max(1, int(len(df) * frac))
    return float(df["return"].tail(k).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--log-dir", type=Path, default=REPO_ROOT / "results" / "phaseA_smoke")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    results: dict[tuple[str, str, str], list[float]] = {}
    t0 = time.monotonic()
    n_jobs = len(CELLS) * len(MODES) * len(ALGOS) * len(seeds)
    done = 0
    print(f"[smokeA] {n_jobs} runs x {args.steps} steps -> {args.log_dir}", flush=True)

    for graph, n_agents, grid_size, label in CELLS:
        for mode in MODES:
            for algo in ALGOS:
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
                        eps_start=1.0,
                        eps_end=0.05,
                        eps_anneal_steps=int(args.steps * 0.8),
                        seed=seed,
                        log_dir=args.log_dir / label / mode,
                        run_id=f"{algo}__{graph}__N{n_agents}__{mode}__seed{seed}",
                    )
                    vals.append(final_window_return(train(cfg)))
                    done += 1
                results[(label, mode, algo)] = vals
                print(
                    f"  [{done}/{n_jobs}] {label}/{mode}/{algo}: "
                    f"mean={np.mean(vals):.3f} sd={np.std(vals):.3f} "
                    f"({time.monotonic() - t0:.0f}s)",
                    flush=True,
                )

    print("\n=== SUMMARY: final-window mean return ===", flush=True)
    for label in (c[3] for c in CELLS):
        print(f"  [{label}]", flush=True)
        for mode in MODES:
            cells = " | ".join(
                f"{a}={np.mean(results[(label, mode, a)]):.3f}" for a in ALGOS
            )
            print(f"    {mode:4s}: {cells}", flush=True)

    print("\n=== GRAPH ADVANTAGE  A = GNN-QMIX - max(QMIX, MLP-QMIX) ===", flush=True)
    flip_seen = False
    for label in (c[3] for c in CELLS):
        def adv(mode: str) -> float:
            g = np.mean(results[(label, mode, "gnn_qmix")])
            ctrl = max(
                np.mean(results[(label, mode, "qmix")]),
                np.mean(results[(label, mode, "mlp_qmix")]),
            )
            return g - ctrl

        a_full, a_ego = adv("full"), adv("ego")
        flip = a_full <= 0 < a_ego
        flip_seen = flip_seen or flip
        print(
            f"  [{label}]  A(full)={a_full:+.3f}  A(ego)={a_ego:+.3f}  "
            f"{'<<< SIGN FLIP (graph wins under ego)' if flip else ''}",
            flush=True,
        )

    print(
        "\nVERDICT: "
        + (
            "at least one cell flips negative->positive under ego. "
            "The positive story has empirical support; proceed to Stage B."
            if flip_seen
            else "no sign flip yet. The graph does not win even under ego at this "
            "budget/cell; inspect curves before investing in Stage B."
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
