"""Measure the two reference points that anchor the Stage C headline scale.

The paper's structure-boundary section quotes two reference points for
``PairRouting`` (CoordGrid ``pair_routing=True``, N=6, 3x3 grid, 25-step
episodes): the random/do-nothing return (~50) and a full-information greedy
oracle (~149, near the analytic ceiling of N * T = 150). This script measures
both so the anchors are a committed artifact rather than prose.

- Random policy: uniform over the 5 actions, every agent, every step.
- Greedy oracle: every agent knows its partner's private goal (full
  information, no routing needed) and picks the action minimising its
  next-step toroidal Manhattan distance to that goal (ties -> first).

Usage::
    python scripts/phaseC_reference_points.py --episodes 500 --seeds 12
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


def run_episode(env: CoordGrid, rng: np.random.Generator, policy: str) -> float:
    env.reset()
    total = 0.0
    done = False
    while not done:
        if policy == "random":
            actions = rng.integers(0, env.n_actions, size=env.n_agents)
        else:  # greedy full-information oracle
            targets = env._goals[env._partner]  # noqa: SLF001 [N, 2]
            actions = np.zeros(env.n_agents, dtype=np.int64)
            for i in range(env.n_agents):
                cand = (env._positions[i] + _ACTION_DELTAS) % env.grid_size  # noqa: SLF001
                d = toroidal_dist(cand, targets[i][None, :], env.grid_size)
                actions[i] = int(np.argmin(d))
        res = env.step(actions)
        total += res.reward
        done = res.done
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, default=6)
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--episode-steps", type=int, default=25)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "phaseC" / "reference_points.csv")
    args = ap.parse_args()

    rows = []
    for policy in ("random", "oracle"):
        for seed in range(args.seeds):
            env = CoordGrid(
                n_agents=args.n_agents, grid_size=args.grid,
                episode_steps=args.episode_steps, graph="matching",
                pair_routing=True, obs_mode="ego",
                max_neighbors_override=args.n_agents - 1,
            )
            env.reset(seed=seed)
            rng = np.random.default_rng(10_000 + seed)
            rets = [run_episode(env, rng, policy) for _ in range(args.episodes)]
            rows.append({
                "policy": policy, "seed": seed, "episodes": args.episodes,
                "mean_return": float(np.mean(rets)), "sd": float(np.std(rets, ddof=1)),
            })
            print(f"  {policy:<7} seed{seed}: {np.mean(rets):7.2f} "
                  f"(sd {np.std(rets, ddof=1):.2f})", flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")
    for policy in ("random", "oracle"):
        sub = df[df.policy == policy]["mean_return"]
        print(f"  {policy:<7} grand mean {sub.mean():7.2f} "
              f"(seed sd {sub.std(ddof=1):.3f}, n={len(sub)}); "
              f"analytic ceiling = {args.n_agents * args.episode_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
