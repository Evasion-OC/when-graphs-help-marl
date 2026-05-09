"""Quick sanity test: does shrinking mixer capacity fix QMIX/GNN-QMIX?

Pilot found that with embed_dim=32, QMIX and GNN-QMIX fail on the small
2-agent / 4×4 task while VDN and IQL learn. Hypothesis: the hypernetwork
mixer is over-parameterized for an 8-dim global state and over-fits the
TD targets through its state-conditioned bias term.

Test: run QMIX and GNN-QMIX with embed_dim=8 on seed 0 only. If they
learn, the hypothesis holds.

This is NOT a tuned re-run of the pilot — it's a diagnostic. Real
re-tuning happens in Phase 2.

Run:
    python scripts/tune_mixer_sanity.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gnnmarl.train import Args, main  # noqa: E402

BASE = dict(
    env="coord_grid",
    grid_size=4,
    n_agents=2,
    max_steps=20,
    coordination_graph="full",
    shape_distance=True,
    total_steps=10_000,
    warmup_steps=500,
    batch_size=64,
    lr=5e-4,
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
    seed=0,
)


def run() -> None:
    cases = [
        ("qmix", 8),
        ("qmix", 16),
        ("gnn_qmix", 8),
        ("gnn_qmix", 16),
    ]
    print(f"Mixer-capacity sanity sweep: {len(cases)} runs (seed=0 only)")
    for i, (algo, ed) in enumerate(cases):
        log_path = f"results/pilot_tune/{algo}_emb{ed}.csv"
        args_dict = {**BASE, "algo": algo, "embed_dim": ed, "log_path": log_path}
        if algo == "gnn_qmix":
            args_dict["gnn_layers"] = 2
        accepted = {k: v for k, v in args_dict.items() if k in Args.__annotations__}
        args = Args(**accepted)
        print(f"\n=== [{i+1}/{len(cases)}] {algo} embed_dim={ed} ===")
        t0 = time.time()
        path = main(args)
        dt = time.time() - t0
        print(f"  done in {dt:.1f}s -> {path}")


if __name__ == "__main__":
    run()
