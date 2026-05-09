"""Training loop. Algorithm-agnostic: takes any BaseAgent + env."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..algos.base import BaseAgent
from ..envs.coord_grid import CoordGridEnv
from ..utils.logging import CSVLogger
from .replay import ReplayBuffer


@dataclass
class TrainConfig:
    total_steps: int = 50_000
    warmup_steps: int = 1_000
    batch_size: int = 64
    buffer_capacity: int = 50_000
    train_every: int = 1
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    log_path: str = "results/run.csv"
    eval_every_episodes: int = 50
    eval_episodes: int = 10
    progress_every_steps: int = 5_000


def _epsilon(step: int, cfg: TrainConfig) -> float:
    if step >= cfg.eps_decay_steps:
        return cfg.eps_end
    frac = step / max(cfg.eps_decay_steps, 1)
    return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac


def evaluate(agent: BaseAgent, env: CoordGridEnv, n_episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    returns = []
    successes = []
    lengths = []
    for ep in range(n_episodes):
        obs, _state = env.reset(seed=seed + 1000 + ep)
        ep_ret = 0.0
        ep_len = 0
        success = False
        done = False
        while not done:
            a = agent.act(obs, epsilon=0.0, rng=rng)  # greedy eval
            obs, _state, r, done, info = env.step(a)
            ep_ret += r
            ep_len += 1
            success = info.get("success", False) or success
        returns.append(ep_ret)
        successes.append(float(success))
        lengths.append(ep_len)
    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_length_mean": float(np.mean(lengths)),
    }


def train(
    agent: BaseAgent,
    env: CoordGridEnv,
    cfg: TrainConfig,
    seed: int,
) -> Path:
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(
        capacity=cfg.buffer_capacity,
        n_agents=env.cfg.n_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        seed=seed,
    )

    log_path = Path(cfg.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(log_path)

    obs, state = env.reset(seed=seed)
    ep_return = 0.0
    ep_len = 0
    ep_count = 0

    step = 0
    while step < cfg.total_steps:
        eps = _epsilon(step, cfg)
        a = agent.act(obs, epsilon=eps, rng=rng)
        next_obs, next_state, reward, done, info = env.step(a)
        buffer.push(obs, state, a, reward, next_obs, next_state, done)
        ep_return += reward
        ep_len += 1
        step += 1

        if (
            step >= cfg.warmup_steps
            and step % cfg.train_every == 0
            and len(buffer) >= cfg.batch_size
        ):
            batch = buffer.sample(cfg.batch_size)
            metrics = agent.learn(batch, env.graph)
        else:
            metrics = {}

        if done:
            ep_count += 1
            row = {
                "step": step,
                "episode": ep_count,
                "ep_return": ep_return,
                "ep_length": ep_len,
                "ep_success": int(info.get("success", False)),
                "epsilon": eps,
                **metrics,
            }
            logger.log(row)
            if ep_count % cfg.eval_every_episodes == 0:
                eval_metrics = evaluate(agent, env, cfg.eval_episodes, seed=seed + 9999)
                logger.log(
                    {
                        "step": step,
                        "episode": ep_count,
                        "ep_return": "",
                        "ep_length": "",
                        "ep_success": "",
                        "epsilon": eps,
                        **eval_metrics,
                    }
                )
            obs, state = env.reset()
            ep_return = 0.0
            ep_len = 0
        else:
            obs, state = next_obs, next_state

        if step % cfg.progress_every_steps == 0:
            print(f"  step {step}/{cfg.total_steps}  episodes={ep_count}  eps={eps:.2f}")

    logger.close()
    return log_path
