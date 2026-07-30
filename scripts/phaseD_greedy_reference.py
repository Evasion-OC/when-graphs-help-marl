"""Scripted-policy headroom diagnostic for the Phase D oracle-headroom gate.

NOT a training run and NOT a budget extension -- a cheap analytic reference,
same style as ``phaseC_classical_reference_points.py`` /
``phaseD_reference_points.py``. Added to resolve an ambiguity the 30k-step
smoke produced (see ``results/phaseD_external/PREREGISTRATION.md`` section
4.1): the trained ``oracle`` arm plateaued at essentially the same
final-window mean as the trained ``mlp`` arm (-57.93 vs -58.26, n=3,
converged by ~40% of the smoke and flat for the remaining 60%). That single
fact is consistent with two very different diagnoses:

  (a) **no headroom in this env config** -- even a hand-scripted policy that
      is HANDED the exact target cannot do meaningfully better than a
      hand-scripted policy that only sees the (identity-blind) landmark
      positions and heads for their centroid. Genuine Branch-0 (design
      section 5): the vehicle lacks headroom at K=3 / local_ratio=0.5 /
      max_cycles=25, not a training bug.
  (b) **headroom exists but the trained oracle arm failed to exploit it** --
      the scripted greedy-to-true-target policy clearly beats the scripted
      greedy-to-centroid policy, yet the TRAINED oracle sat at the centroid
      level. That points at an oracle-wiring / training issue, not an
      intractable vehicle.

Two deterministic (no learning) policies on the E1 confirmatory config
(k=3, comm off, graph=true -- graph is irrelevant here since both policies
are centralized-omniscient scripts, not GNN/MLP-conditioned actors):

  - **greedy_target**: each local agent moves toward the landmark its
    partner's own ``goal_b`` specifies (exactly what ``oracle`` mode injects
    into that agent's observation -- see ``mpe_reference_pairs.py`` module
    docstring and this env's true reward decomposition:
    ``pair_reward = -0.5 * [dist(agent1, agent0.goal_b) +
    dist(agent0, agent1.goal_b)]``, so agent_0's own greedy target is
    ``agent_1.goal_b`` and vice versa).
  - **greedy_centroid**: each local agent moves toward the centroid of the
    3 visible landmark POSITIONS (identity-blind -- exactly what an ``mlp``
    agent can compute from its own observation without any target info).

Discrete action mapping verified directly against
``mpe2/_mpe_utils/simple_env.py::_set_action``: ``move=1/2/3/4`` ==
``-x/+x/-y/+y`` (``move=0`` no-op); the wrapper's ``move = action % 5``
encoding matches. Greedy action = the axis (x or y) with the larger absolute
displacement needed, signed toward the target; ties broken toward x.

Usage::

    python scripts/phaseD_greedy_reference.py --episodes 200 --seeds 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnmarl.envs.mpe_reference_pairs import (  # noqa: E402
    _FIXED_SAY,
    _MOVE_DIM,
    MPEReferencePairs,
)


def _greedy_move(delta: np.ndarray) -> int:
    """delta = target_pos - agent_pos (2,). Returns a move index in {0,1,2,3,4}."""
    dx, dy = float(delta[0]), float(delta[1])
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return 0  # no_action -- already there
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 1  # +x : -x
    return 4 if dy > 0 else 3  # +y : -y


def run_episode(env: MPEReferencePairs, policy: str) -> float:
    env.reset()
    total = 0.0
    done = False
    while not done:
        actions = np.zeros(env.n_agents, dtype=np.int64)
        for kk in range(env.k):
            sub = env._envs[kk]  # noqa: SLF001 -- diagnostic script, read-only
            world = sub.unwrapped.world
            a0, a1 = world.agents[0], world.agents[1]
            g0, g1 = 2 * kk, 2 * kk + 1
            if policy == "greedy_target":
                # agent p's own movement affects the OTHER agent's local
                # reward term, so agent p's greedy target is the PARTNER's
                # own goal_b (see module docstring's reward decomposition).
                tgt0 = a1.goal_b.state.p_pos
                tgt1 = a0.goal_b.state.p_pos
            elif policy == "greedy_centroid":
                centroid = np.mean([lm.state.p_pos for lm in world.landmarks], axis=0)
                tgt0 = tgt1 = centroid
            else:
                raise ValueError(policy)
            move0 = _greedy_move(tgt0 - a0.state.p_pos)
            move1 = _greedy_move(tgt1 - a1.state.p_pos)
            # comm disabled (comm_on=False): fix `say` to the same constant
            # the training harness uses; irrelevant since the comm slot is
            # zeroed in the observation regardless.
            actions[g0] = move0 + _MOVE_DIM * _FIXED_SAY
            actions[g1] = move1 + _MOVE_DIM * _FIXED_SAY
        res = env.step(actions)
        total += res.reward
        done = res.done
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "phaseD_external" / "greedy_reference.csv",
    )
    args = ap.parse_args()

    rows = []
    for policy in ("greedy_target", "greedy_centroid"):
        for seed in range(args.seeds):
            env = MPEReferencePairs(k=3, graph="true", comm_on=False, max_cycles=25, seed=seed)
            rets = [run_episode(env, policy) for _ in range(args.episodes)]
            env.close()
            rows.append(
                {
                    "cell": "E1",
                    "task": "mpe_reference_pairs",
                    "k": 3,
                    "policy": policy,
                    "seed": seed,
                    "episodes": args.episodes,
                    "mean_return": float(np.mean(rets)),
                    "sd": float(np.std(rets, ddof=1)),
                }
            )
            print(
                f"  [E1 k=3  ] {policy:<16} seed{seed}: {np.mean(rets):7.2f} "
                f"(sd {np.std(rets, ddof=1):.2f})",
                flush=True,
            )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")
    for policy in ("greedy_target", "greedy_centroid"):
        sub = df[df.policy == policy]["mean_return"]
        print(
            f"  {policy:<16} grand mean {sub.mean():7.2f} "
            f"(seed sd {sub.std(ddof=1):.3f}, n={len(sub)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
