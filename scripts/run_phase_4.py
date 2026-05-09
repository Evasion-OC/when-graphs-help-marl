"""Phase 4 — depth × diameter ablation (H3).

Hypothesis 3 (paper version, Phase 4 form):
    GNN-QMIX's advantage over QMIX scales with how well the GNN's
    receptive field (depth L) matches the coordination-graph diameter d.
    Specifically, performance is roughly an inverted-U in (L − d):
    too few layers can't propagate information across the graph;
    too many cause over-smoothing.

To test this: hold the algorithm fixed (GNN-QMIX with Phase 2A's locked
embed_dim, mixer_lr, mixer_init), vary GNN depth L and graph diameter d
independently, and look for the predicted ridge along L ≈ d.

We use ring topologies in CoordGrid because their diameter has a
clean closed form (floor(N/2) for ring of N), so d is controllable
without constructing pathological graphs.

Sweep matrix:
    gnn_layers L ∈ {1, 2, 3, 4}       — 4 values (covers the "too
                                         few" through "matched" range
                                         for our N range)
    N ∈ {4, 6, 8}                      — 3 ring sizes giving
                                         diameters d ∈ {2, 3, 4}
    seeds 0..4                         — 5 seeds
    = 60 runs

Wall-clock estimate: ~5 hours on M1 Pro / MPS at 30k steps each.

Reference cell (no GNN): a parallel small QMIX sweep at the same N
× seed grid is included so we can plot GNN-QMIX vs QMIX deltas. That
adds 15 more runs for 75 total, ~6 hours wall.

Output: results/phase_4_ablation/{algo}/L{L}_N{N}_seed{S}.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gnnmarl.train import Args, main  # noqa: E402

LOCKED = ROOT / "configs" / "phase_2_locked.yaml"

GNN_LAYERS = (1, 2, 3, 4)  # L
N_AGENTS = (4, 6, 8)        # ring diameters d = floor(N/2) ∈ {2, 3, 4}
SEEDS = [0, 1, 2, 3, 4]

COMMON = dict(
    env="coord_grid",
    grid_size=6,             # gives 36 cells; enough for 8 agents + 8 targets
    max_steps=40,
    coordination_graph="ring",  # the controllable-diameter knob
    shape_distance=True,
    total_steps=30_000,
    warmup_steps=1_000,
    batch_size=64,
    gamma=0.99,
    hidden=64,
    target_update_every=200,
    grad_clip=10.0,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=15_000,
    eval_every_episodes=100,
    eval_episodes=20,
    device="auto",
    lr=5e-4,
)


def main_sweep() -> None:
    if not LOCKED.exists():
        sys.stderr.write(f"!! {LOCKED} not found. Run Phase 2A first.\n")
        sys.exit(2)

    locked = yaml.safe_load(LOCKED.read_text())
    gnn_locked = locked.get("gnn_qmix", {}) or {}
    qmix_locked = locked.get("qmix", {}) or {}
    print("Phase 4 — using locked Phase 2A configs:")
    print(f"  gnn_qmix: {gnn_locked}")
    print(f"  qmix    : {qmix_locked}")

    runs: list[tuple[str, dict]] = []
    # GNN-QMIX: depth × diameter × seed
    for L in GNN_LAYERS:
        for n in N_AGENTS:
            for s in SEEDS:
                runs.append((
                    "gnn_qmix",
                    dict(gnn_locked, gnn_layers=L, n_agents=n, seed=s,
                         log_path=f"results/phase_4_ablation/gnn_qmix/L{L}_N{n}_seed{s}.csv"),
                ))
    # QMIX reference: just diameter × seed (depth doesn't apply)
    for n in N_AGENTS:
        for s in SEEDS:
            runs.append((
                "qmix",
                dict(qmix_locked, n_agents=n, seed=s,
                     log_path=f"results/phase_4_ablation/qmix/N{n}_seed{s}.csv"),
            ))

    print(f"\nPhase 4 sweep: {len(runs)} runs")
    print(
        f"  - GNN-QMIX cells: {len(GNN_LAYERS)} L × {len(N_AGENTS)} N "
        f"× {len(SEEDS)} seeds = {len(GNN_LAYERS) * len(N_AGENTS) * len(SEEDS)}"
    )
    print(
        f"  - QMIX reference: {len(N_AGENTS)} N × {len(SEEDS)} seeds "
        f"= {len(N_AGENTS) * len(SEEDS)}"
    )
    print()

    t_total = time.time()
    for i, (algo, params) in enumerate(runs):
        merged = {**COMMON, **params, "algo": algo}
        accepted = {k: v for k, v in merged.items() if k in Args.__annotations__}
        args = Args(**accepted)
        cell_id = (
            f"L{params.get('gnn_layers', '-')}_N{params['n_agents']}"
            f"_seed{params['seed']}"
        )
        print(f"=== [{i+1}/{len(runs)}] {algo} {cell_id} ===")
        t0 = time.time()
        path = main(args)
        dt = time.time() - t0
        print(f"  done in {dt:.1f}s -> {path}\n")
    print(f"Phase 4 sweep total wall: {(time.time() - t_total) / 60:.1f} min")


if __name__ == "__main__":
    main_sweep()
