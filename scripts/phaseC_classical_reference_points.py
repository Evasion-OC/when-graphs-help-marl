"""Reference points for the classical-CG baseline pre-registration (design
``docs/CLASSICAL_CG_BASELINE_DESIGN.md`` sections 4/6).

Measures, in the *same style* as ``phaseC_reference_points.py`` (random
policy + full-information greedy oracle), the two floors/ceilings this
pre-reg's smoke gates are read against that are NOT already committed
anywhere in the repo:

  - **cell C (relay_routing)**: random floor and a full-information greedy
    oracle (the analytic ceiling / routing-headroom scale). Cell C's smoke
    gate (design section 4.1) needs this to judge whether ``gnn_true``
    clears the floor with real 2-hop routing.
  - **cell A (default co-location reward, ring, N=4, grid=5, full obs)**:
    random floor only (home-turf sanity; the oracle here is a
    coordination-search problem, not needed for the smoke gate).

Does NOT touch ``phaseC_reference_points.py`` or its output (the PairRouting
floor/oracle ~50 / ~149 the design's cell-B reasoning already relies on) --
that file and its CSV are frozen historical results.

Usage::
    python scripts/phaseC_classical_reference_points.py --episodes 500 --seeds 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnmarl.envs.coord_grid import _ACTION_DELTAS, CoordGrid  # noqa: E402


def toroidal_dist(a: np.ndarray, b: np.ndarray, gs: int) -> np.ndarray:
    raw = np.abs(a - b)
    return np.minimum(raw, gs - raw).sum(axis=-1)


def run_relay_episode(env: CoordGrid, rng: np.random.Generator, policy: str) -> float:
    env.reset()
    total = 0.0
    done = False
    role = env._role  # noqa: SLF001  -- 0=source,1=relay,2=sink; fixed per env
    n = env.n_agents
    while not done:
        if policy == "random":
            actions = rng.integers(0, env.n_actions, size=n)
        else:  # full-information greedy oracle
            # Relay already observes its own goal (no routing needed); sink is
            # PRIVILEGED here to see its chain's source's goal directly (the
            # routing-headroom ceiling). Source's own action never affects
            # reward, but give it a legal action (uniform) rather than 0-bias.
            actions = np.zeros(n, dtype=np.int64)
            sink_idx = np.where(role == 2)[0]
            source_idx_for_sink = sink_idx - 2
            relay_idx = np.where(role == 1)[0]
            source_idx = np.where(role == 0)[0]
            for i in relay_idx:
                cand = (env._positions[i] + _ACTION_DELTAS) % env.grid_size  # noqa: SLF001
                d = toroidal_dist(cand, env._goals[i][None, :], env.grid_size)  # noqa: SLF001
                actions[i] = int(np.argmin(d))
            for si, ti in zip(source_idx_for_sink, sink_idx, strict=True):
                cand = (env._positions[ti] + _ACTION_DELTAS) % env.grid_size  # noqa: SLF001
                d = toroidal_dist(cand, env._goals[si][None, :], env.grid_size)  # noqa: SLF001
                actions[ti] = int(np.argmin(d))
            actions[source_idx] = rng.integers(0, env.n_actions, size=len(source_idx))
        res = env.step(actions)
        total += res.reward
        done = res.done
    return total


def run_colocation_episode(env: CoordGrid, rng: np.random.Generator) -> float:
    env.reset()
    total = 0.0
    done = False
    while not done:
        actions = rng.integers(0, env.n_actions, size=env.n_agents)
        res = env.step(actions)
        total += res.reward
        done = res.done
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "results" / "phaseC_classical" / "reference_points.csv",
    )
    args = ap.parse_args()

    rows = []

    # --- Cell C: relay_routing, N=6, grid=3, episode_steps=25, ego obs ---
    for policy in ("random", "oracle"):
        for seed in range(args.seeds):
            env = CoordGrid(
                n_agents=6, grid_size=3, episode_steps=25, graph="relay",
                relay_routing=True, obs_mode="ego", max_neighbors_override=5,
            )
            env.reset(seed=seed)
            rng = np.random.default_rng(20_000 + seed)
            rets = [run_relay_episode(env, rng, policy) for _ in range(args.episodes)]
            rows.append({
                "cell": "C", "task": "relay_routing", "policy": policy, "seed": seed,
                "episodes": args.episodes, "mean_return": float(np.mean(rets)),
                "sd": float(np.std(rets, ddof=1)),
            })
            print(f"  [C relay ] {policy:<7} seed{seed}: {np.mean(rets):7.2f} "
                  f"(sd {np.std(rets, ddof=1):.2f})", flush=True)

    # --- Cell A: default co-location reward, ring, N=4, grid=5, full obs ---
    for seed in range(args.seeds):
        env = CoordGrid(
            n_agents=4, grid_size=5, episode_steps=25, graph="ring",
            obs_mode="full", max_neighbors_override=3,
        )
        env.reset(seed=seed)
        rng = np.random.default_rng(30_000 + seed)
        rets = [run_colocation_episode(env, rng) for _ in range(args.episodes)]
        rows.append({
            "cell": "A", "task": "colocation_ring", "policy": "random", "seed": seed,
            "episodes": args.episodes, "mean_return": float(np.mean(rets)),
            "sd": float(np.std(rets, ddof=1)),
        })
        print(f"  [A ring  ] random  seed{seed}: {np.mean(rets):7.2f} "
              f"(sd {np.std(rets, ddof=1):.2f})", flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")
    for cell, task, policy in (
        ("C", "relay_routing", "random"), ("C", "relay_routing", "oracle"),
        ("A", "colocation_ring", "random"),
    ):
        sub = df[(df.cell == cell) & (df.task == task) & (df.policy == policy)]["mean_return"]
        print(f"  cell {cell} {task:<15} {policy:<7} grand mean {sub.mean():7.2f} "
              f"(seed sd {sub.std(ddof=1):.3f}, n={len(sub)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
