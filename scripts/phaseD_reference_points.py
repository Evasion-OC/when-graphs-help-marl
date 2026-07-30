"""Reference points for the Phase D external-validity pre-registration
(``results/phaseD_external/PREREGISTRATION.md`` section 4.1).

Measures, on the exact E1 confirmatory env config (``k=3``, ``comm_on=False``,
``graph="true"``, non-oracle observation), a **random** policy and a
**do-nothing** (always no-op) policy over many episodes. These two together
define the "floor band" gate-1 (privacy/implementation gate) is read
against: ``mlp`` cannot observe its own target (structural absence, see
``mpe_reference_pairs.py`` module docstring), so a healthy ``mlp`` should sit
in or near ``[min(random, do_nothing), max(random, do_nothing)]``, not climb
toward ``oracle``. A single uniform-random floor is NOT used alone, because
continuous MPE physics lets a policy that does not know its own target still
beat pure noise by exploiting generic structure (drifting toward the
landmark centroid, standing still near spawn) -- that is still "floored," not
a privacy leak (design doc's analogue of the additive-reward caution already
on record for cell C in ``results/phaseC_classical/PREREGISTRATION.md``
section 2b).

Cell R (relay addendum) reuses the EXISTING
``results/phaseC_classical/reference_points.csv`` relay references
(``relay_routing`` random floor 33.37 / oracle 99.11) -- no new reference-
point run for cell R here.

Usage::

    python scripts/phaseD_reference_points.py --episodes 500 --seeds 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnmarl.envs.mpe_reference_pairs import MPEReferencePairs  # noqa: E402

_DO_NOTHING_ACTION = 0  # move=0 == "no_action" (see mpe_reference_pairs.py docstring)


def run_episode(env: MPEReferencePairs, rng: np.random.Generator, policy: str) -> float:
    env.reset()
    total = 0.0
    done = False
    while not done:
        if policy == "random":
            actions = rng.integers(0, env.n_actions, size=env.n_agents)
        elif policy == "do_nothing":
            actions = np.full(env.n_agents, _DO_NOTHING_ACTION, dtype=np.int64)
        else:
            raise ValueError(policy)
        res = env.step(actions)
        total += res.reward
        done = res.done
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "phaseD_external" / "reference_points.csv",
    )
    args = ap.parse_args()

    rows = []

    # E1 confirmatory config: k=3, comm off, graph=true, non-oracle obs.
    for policy in ("random", "do_nothing"):
        for seed in range(args.seeds):
            env = MPEReferencePairs(k=3, graph="true", comm_on=False, max_cycles=25, seed=seed)
            rng = np.random.default_rng(40_000 + seed)
            rets = [run_episode(env, rng, policy) for _ in range(args.episodes)]
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
                f"  [E1 k=3  ] {policy:<10} seed{seed}: {np.mean(rets):7.2f} "
                f"(sd {np.std(rets, ddof=1):.2f})",
                flush=True,
            )

    # E0 honesty-anchor config: k=1, comm ON, unmodified simple_reference --
    # cheap sanity floor for the non-confirmatory cell too.
    for policy in ("random", "do_nothing"):
        for seed in range(args.seeds):
            env = MPEReferencePairs(k=1, graph="true", comm_on=True, max_cycles=25, seed=seed)
            rng = np.random.default_rng(50_000 + seed)
            rets = [run_episode(env, rng, policy) for _ in range(args.episodes)]
            env.close()
            rows.append(
                {
                    "cell": "E0",
                    "task": "mpe_reference_pairs",
                    "k": 1,
                    "policy": policy,
                    "seed": seed,
                    "episodes": args.episodes,
                    "mean_return": float(np.mean(rets)),
                    "sd": float(np.std(rets, ddof=1)),
                }
            )
            print(
                f"  [E0 k=1  ] {policy:<10} seed{seed}: {np.mean(rets):7.2f} "
                f"(sd {np.std(rets, ddof=1):.2f})",
                flush=True,
            )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")
    for cell, k in (("E1", 3), ("E0", 1)):
        for policy in ("random", "do_nothing"):
            sub = df[(df.cell == cell) & (df.k == k) & (df.policy == policy)]["mean_return"]
            print(
                f"  cell {cell} k={k} {policy:<10} grand mean {sub.mean():7.2f} "
                f"(seed sd {sub.std(ddof=1):.3f}, n={len(sub)})"
            )
    print(
        "\nNote: relay-addendum (cell R) reuses the EXISTING "
        "results/phaseC_classical/reference_points.csv relay references "
        "(random 33.37 / oracle 99.11) -- not recomputed here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
