"""Generic Phase-0 trainer.

The trainer is algo-agnostic and env-agnostic: it holds an ``Algo`` instance and
a ``MultiAgentEnv`` and runs a standard off-policy episode loop with linear
epsilon decay and per-episode CSV logging.

A run is fully specified by a :class:`TrainConfig`. The :func:`train` entrypoint
returns the run directory it wrote to; callers (CLI, sweep scripts, tests) own
how they assemble the config and what they do with the resulting directory.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gnnmarl.algos import make_algo
from gnnmarl.envs import make_env
from gnnmarl.utils import CSVLogger, Transition, set_seed


@dataclass
class TrainConfig:
    """Full specification of a single training run.

    Note on dotted overrides: the CLI flattens ``env_kwargs`` / ``algo_kwargs``
    via tyro's nested-dataclass support. For YAML, callers can splat directly
    into these dicts.
    """

    # What to run.
    env: str = "coord_grid"
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    algo: str = "qmix"
    algo_kwargs: dict[str, Any] = field(default_factory=dict)

    # How long.
    total_env_steps: int = 50_000
    # Run one gradient step every ``update_every`` env steps once the buffer
    # is warm. Standard off-policy practice; 1 = update every step.
    update_every: int = 4
    # Number of env steps to collect before any gradient step is taken.
    warmup_steps: int = 200

    # Exploration schedule (linear).
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_anneal_steps: int = 50_000

    # Bookkeeping.
    seed: int = 0
    log_dir: Path = Path("results/runs")
    run_id: str | None = None
    # ``"auto"`` resolves to ``"cuda"`` if available, else ``"cpu"``. Phase 0
    # ran CPU-only; Colab runs benefit from auto-detect.
    device: str = "auto"

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        return f"{self.algo}__{self.env}__seed{self.seed}"


def _linear_eps(step: int, cfg: TrainConfig) -> float:
    if cfg.eps_anneal_steps <= 0:
        return cfg.eps_end
    frac = min(1.0, step / cfg.eps_anneal_steps)
    return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac


def train(cfg: TrainConfig) -> Path:
    """Run one training job end-to-end and return its run directory."""
    rng_action, _torch_gen = set_seed(cfg.seed)

    env = make_env(cfg.env, **cfg.env_kwargs)
    # First reset establishes the initial adjacency. We seed the env from the
    # training seed so different runs with the same cfg.seed see the same
    # initial graph (Erdős–Rényi resamples are then chained off that seed).
    result = env.reset(seed=cfg.seed)

    if cfg.device == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(cfg.device)

    algo_kwargs: dict[str, Any] = dict(cfg.algo_kwargs)
    # The algo needs to know the env dims — never trust the YAML to repeat them.
    algo_kwargs.setdefault("n_agents", env.n_agents)
    algo_kwargs.setdefault("obs_dim", env.obs_dim)
    algo_kwargs.setdefault("state_dim", env.state_dim)
    algo_kwargs.setdefault("n_actions", env.n_actions)
    algo_kwargs.setdefault("seed", cfg.seed)
    algo_kwargs.setdefault("device", resolved_device)
    algo = make_algo(cfg.algo, **algo_kwargs)

    run_dir = Path(cfg.log_dir) / cfg.resolved_run_id()
    logger = CSVLogger(
        run_dir=run_dir,
        run_id=cfg.resolved_run_id(),
        config={
            **{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)},
            # Surface the env metadata so downstream analysis doesn't need to
            # re-instantiate the env to know its shape.
            "n_agents": env.n_agents,
            "obs_dim": env.obs_dim,
            "state_dim": env.state_dim,
            "n_actions": env.n_actions,
        },
    )

    try:
        env_step = 0
        episode = 0
        obs, state = result.obs, result.state
        adj = env.adjacency()
        ep_return = 0.0
        ep_len = 0
        losses: list[float] = []
        grad_norms: list[float] = []

        while env_step < cfg.total_env_steps:
            eps = _linear_eps(env_step, cfg)
            actions = algo.act(obs, adj, epsilon=eps, rng=rng_action)
            step_result = env.step(actions)

            algo.observe(
                Transition(
                    obs=obs,
                    state=state,
                    actions=actions,
                    reward=step_result.reward,
                    next_obs=step_result.obs,
                    next_state=step_result.state,
                    done=step_result.done,
                    adj=adj,
                )
            )

            if env_step >= cfg.warmup_steps and (env_step % cfg.update_every == 0):
                metrics = algo.update()
                if metrics is not None:
                    losses.append(metrics["loss"])
                    grad_norms.append(metrics["grad_norm"])

            ep_return += step_result.reward
            ep_len += 1
            env_step += 1

            obs, state = step_result.obs, step_result.state

            if step_result.done or env_step >= cfg.total_env_steps:
                mean_loss = float(np.mean(losses)) if losses else None
                mean_grad = float(np.mean(grad_norms)) if grad_norms else None
                logger.log_episode(
                    episode=episode,
                    env_step=env_step,
                    ret=ep_return,
                    ep_len=ep_len,
                    epsilon=eps,
                    loss=mean_loss,
                    grad_norm=mean_grad,
                )
                episode += 1

                if env_step >= cfg.total_env_steps:
                    break

                # Reset for next episode. Seed each reset off (cfg.seed, episode)
                # so Erdős–Rényi adjacency is reproducible per episode.
                reset_result = env.reset(seed=cfg.seed * 1_000_003 + episode)
                obs, state = reset_result.obs, reset_result.state
                adj = env.adjacency()
                ep_return = 0.0
                ep_len = 0
                losses = []
                grad_norms = []
    finally:
        logger.close()
        env.close()

    return run_dir
