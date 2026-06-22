"""Unit tests for the TokenMatch out-of-harness env (Stage D)."""

from __future__ import annotations

import numpy as np
import pytest

from gnnmarl.envs import TokenMatch, make_env
from gnnmarl.envs.base import MultiAgentEnv, StepResult


def test_shapes_and_protocol() -> None:
    env = TokenMatch(n_agents=6, n_tokens=5, graph="matching", seed=0)
    assert isinstance(env, MultiAgentEnv)
    assert env.obs_dim == 5 and env.state_dim == 6 and env.n_actions == 5
    out = env.reset(seed=0)
    assert isinstance(out, StepResult)
    assert out.obs.shape == (6, 5) and out.obs.dtype == np.float32
    assert out.state.shape == (6,)


def test_make_env_factory() -> None:
    assert isinstance(make_env("token_match", n_agents=4), TokenMatch)


def test_obs_is_own_token_one_hot_only() -> None:
    env = TokenMatch(n_agents=6, n_tokens=5, graph="matching", seed=1)
    env.reset(seed=2)
    obs = env._compute_obs()
    # exactly one hot per agent, at its own token; nothing about any partner.
    assert np.all(obs.sum(axis=1) == 1.0)
    assert np.array_equal(obs.argmax(axis=1), env._tokens)


def test_partner_is_canonical_matching() -> None:
    env = TokenMatch(n_agents=6, graph="complete", seed=0)
    assert env._partner.tolist() == [1, 0, 3, 2, 5, 4]


def test_comm_graphs_disjoint_and_complete() -> None:
    a_true = TokenMatch(n_agents=6, graph="matching", seed=0).adjacency()
    a_wrong = TokenMatch(n_agents=6, graph="matching_wrong", seed=0).adjacency()
    a_comp = TokenMatch(n_agents=6, graph="complete", seed=0).adjacency()
    np.testing.assert_array_equal(a_true.sum(1), np.ones(6))
    np.testing.assert_array_equal(a_wrong.sum(1), np.ones(6))
    assert np.all(a_true * a_wrong == 0.0)          # edge-disjoint
    assert a_comp.sum() == 6 * 5


def test_reward_max_when_outputting_partner_token() -> None:
    env = TokenMatch(n_agents=6, n_tokens=5, graph="matching", seed=0)
    env.reset(seed=3)
    correct = env._tokens[env._partner]             # each agent's target = partner's token
    sr = env.step(correct)
    assert sr.reward == 6.0                          # all correct -> N
    wrong = (correct + 1) % env.n_tokens
    assert env.step(wrong).reward == 0.0


@pytest.mark.parametrize("kwargs,match", [
    ({"n_agents": 5}, "even"),
    ({"n_agents": 2, "graph": "matching_wrong"}, ">= 4"),
    ({"n_agents": 6, "n_tokens": 1}, "n_tokens"),
])
def test_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TokenMatch(seed=0, **kwargs)
