"""CLI entry point: `python -m gnnmarl.train --algo gnn_qmix --env coord_grid`.

Single-run trainer. Multi-seed sweeps are launched by `scripts/sweep.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from .algos.base import AlgoConfig
from .algos.gnn_qmix import GNNQMIX
from .algos.iql import IQL
from .algos.qmix import QMIX
from .algos.vdn import VDN
from .envs.coord_grid import CoordGridConfig, CoordGridEnv
from .training.loop import TrainConfig, train
from .utils.seeding import pick_device, set_global_seed

ALGO_CLASSES = {
    "iql": IQL,
    "vdn": VDN,
    "qmix": QMIX,
    "gnn_qmix": GNNQMIX,
}


@dataclass
class Args:
    algo: str = "gnn_qmix"
    env: str = "coord_grid"
    seed: int = 0
    grid_size: int = 6
    n_agents: int = 4
    max_steps: int = 40
    coordination_graph: str = "full"
    edge_p: float = 0.5
    knn_k: int = 2
    shape_distance: bool = False
    total_steps: int = 50_000
    warmup_steps: int = 1_000
    batch_size: int = 64
    lr: float = 5e-4
    gamma: float = 0.99
    hidden: int = 64
    embed_dim: int = 32
    gnn_layers: int = 2
    mixer_lr: float | None = None
    mixer_init: str = "default"
    init_scale: float = 0.1
    target_update_every: int = 200
    grad_clip: float = 10.0
    eps_decay_steps: int = 20_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    log_path: str = "results/run.csv"
    device: str = "auto"
    eval_every_episodes: int = 50
    eval_episodes: int = 10


def main(args: Args) -> Path:
    if args.algo not in ALGO_CLASSES:
        raise ValueError(f"Unknown algo: {args.algo}. Choices: {list(ALGO_CLASSES)}")
    if args.env != "coord_grid":
        raise ValueError(f"Phase 0 supports env='coord_grid' only; got {args.env!r}")

    set_global_seed(args.seed)
    device = pick_device(args.device)

    env_cfg = CoordGridConfig(
        grid_size=args.grid_size,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        coordination_graph=args.coordination_graph,
        edge_p=args.edge_p,
        knn_k=args.knn_k,
        shape_distance=args.shape_distance,
    )
    env = CoordGridEnv(env_cfg, seed=args.seed)

    algo_cfg = AlgoConfig(
        n_agents=args.n_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden=args.hidden,
        embed_dim=args.embed_dim,
        gnn_layers=args.gnn_layers,
        lr=args.lr,
        mixer_lr=args.mixer_lr,
        mixer_init=args.mixer_init,
        init_scale=args.init_scale,
        gamma=args.gamma,
        target_update_every=args.target_update_every,
        grad_clip=args.grad_clip,
    )
    agent = ALGO_CLASSES[args.algo](algo_cfg, device)

    train_cfg = TrainConfig(
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        log_path=args.log_path,
        eval_every_episodes=args.eval_every_episodes,
        eval_episodes=args.eval_episodes,
    )

    print(
        f"[{args.algo}] device={device} "
        f"agents={args.n_agents} grid={args.grid_size}x{args.grid_size} "
        f"graph={args.coordination_graph} (diam={env.graph_diameter}) "
        f"steps={args.total_steps} seed={args.seed}"
    )
    return train(agent, env, train_cfg, seed=args.seed)


if __name__ == "__main__":
    main(tyro.cli(Args))
