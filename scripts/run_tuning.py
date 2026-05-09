"""Phase 2A tuning sweep — focused grid for QMIX and GNN-QMIX.

Phase 1b's diagnostic showed that shrinking embed_dim alone fixed
GNN-QMIX (peak 0.10 → 0.30) but did NOT fix vanilla QMIX (peak stayed
~0.10 across embed_dim ∈ {8, 16}). So Phase 2A's tuning grid widens
the QMIX/GNN-QMIX configurations along two more axes:

  - mixer_lr ∈ {1e-4, 5e-4}  — smaller may stop the mixer's
                                state-conditioned bias from absorbing
                                the TD error before it flows back to
                                the Q-net.
  - mixer_init ∈ {default, orthogonal}  — orthogonal init with small
                                          gain (0.1) biases the mixer
                                          toward near-zero output at
                                          init, letting the Q-net
                                          dominate early training.

embed_dim ∈ {8, 16}: these were the two values Phase 1b verified
contain the divergence; embed_dim=32 is excluded (Phase 1b showed it
diverges).

Cells: 2 algos × 2 embeds × 2 mixer_lrs × 2 inits = 16 configs.
Seeds: 3 per cell. Budget: 10k env steps per run (matches Phase 1
pilot budget so peak-success metrics are directly comparable).

IQL and VDN are NOT in this grid because Phase 1 confirmed their
defaults work; they'll be re-run at full budget in Phase 2C with the
same lr=5e-4 used in Phase 1.

Total: 16 × 3 = 48 runs. Wall-clock estimate: ~75 min on M1 Pro / MPS.

Output: results/phase_2_tune/{algo}/{config_id}_seed{N}.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gnnmarl.train import Args, main  # noqa: E402

# ---- Common env / training settings (shared across all tuning runs) -------
COMMON = dict(
    env="coord_grid",
    grid_size=4,
    n_agents=2,
    max_steps=20,
    coordination_graph="full",
    shape_distance=True,
    total_steps=10_000,  # match Phase 1 budget for direct comparability
    warmup_steps=500,
    batch_size=64,
    gamma=0.99,
    hidden=64,
    target_update_every=200,
    grad_clip=10.0,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=5_000,
    eval_every_episodes=50,
    eval_episodes=20,
    device="auto",
    lr=5e-4,  # Q-net lr fixed at the Phase 1 default
    gnn_layers=2,  # fixed; Phase 4 ablates it
)

SEEDS = [0, 1, 2]


def cfg_id(algo: str, params: dict) -> str:
    """Stable, filesystem-safe id for a (algo, params) cell."""
    parts = [algo]
    for k in ("embed_dim", "mixer_lr", "mixer_init"):
        if k in params:
            v = params[k]
            if isinstance(v, float):
                parts.append(f"{k}={v:g}")
            else:
                parts.append(f"{k}={v}")
    return "__".join(parts)


def expand_grid() -> list[tuple[str, dict]]:
    """Build the (algo, hyperparams) cells: 2 × 2 × 2 × 2 = 16 cells."""
    cells: list[tuple[str, dict]] = []
    for algo in ("qmix", "gnn_qmix"):
        for embed in (8, 16):
            for mlr in (1e-4, 5e-4):
                for init in ("default", "orthogonal"):
                    p = {
                        "embed_dim": embed,
                        "mixer_lr": mlr,
                        "mixer_init": init,
                    }
                    cells.append((algo, p))
    return cells


def main_sweep() -> None:
    cells = expand_grid()
    runs = [(a, p, s) for a, p in cells for s in SEEDS]
    print(
        f"Phase 2A tuning: {len(cells)} cells × {len(SEEDS)} seeds = {len(runs)} runs"
    )
    print(f"  Env: 2 agents on 4×4, full graph, shaping=on, {COMMON['total_steps']} steps")
    print()

    t_total = time.time()
    for i, (algo, params, seed) in enumerate(runs):
        cid = cfg_id(algo, params)
        log_path = f"results/phase_2_tune/{algo}/{cid}_seed{seed}.csv"
        merged = {**COMMON, **params, "algo": algo, "seed": seed, "log_path": log_path}
        accepted = {k: v for k, v in merged.items() if k in Args.__annotations__}
        args = Args(**accepted)
        print(f"=== [{i+1}/{len(runs)}] {cid} seed={seed} ===")
        t0 = time.time()
        path = main(args)
        dt = time.time() - t0
        print(f"  done in {dt:.1f}s -> {path}\n")
    print(f"Phase 2A tuning total wall: {(time.time() - t_total) / 60:.1f} min")


if __name__ == "__main__":
    main_sweep()
