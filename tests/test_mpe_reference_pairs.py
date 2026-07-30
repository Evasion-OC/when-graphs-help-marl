"""Unit tests for ``gnnmarl.envs.mpe_reference_pairs.MPEReferencePairs``.

Covers the smoke gates docs/EXTERNAL_VALIDITY_DESIGN.md section 6 depends on:
privacy (own-target absence + comm zeroing; oracle exposes the partner goal),
adjacency properties (true/wrong edge-disjoint 1-regular, complete correct,
obs_dim constant across non-oracle graphs), reward wiring (sum of native
per-pair rewards), determinism, and short-training liveness.

The whole module is skipped if ``mpe2`` is not installed (the ``[mpe]``
extra) -- the core package must still import and the rest of the suite must
still pass without it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mpe2")

from gnnmarl.envs import make_env  # noqa: E402
from gnnmarl.envs.base import MultiAgentEnv, StepResult  # noqa: E402
from gnnmarl.envs.mpe_reference_pairs import MPEReferencePairs  # noqa: E402


# --------------------------------------------------------------------- shapes


@pytest.mark.parametrize("graph", ["true", "wrong", "complete"])
def test_shapes_and_obs_dim_constant_across_graphs(graph: str) -> None:
    env = MPEReferencePairs(k=3, graph=graph, seed=0)
    assert isinstance(env, MultiAgentEnv)
    assert env.n_agents == 6
    assert env.obs_dim == 21  # constant across non-oracle graphs
    assert env.state_dim == 21 * 6
    assert env.n_actions == 5  # movement-only with comm disabled

    out = env.reset(seed=0)
    assert isinstance(out, StepResult)
    assert out.obs.shape == (6, 21)
    assert out.obs.dtype == np.float32
    assert out.state.shape == (21 * 6,)

    actions = np.zeros(6, dtype=np.int64)
    step_out = env.step(actions)
    assert step_out.obs.shape == (6, 21)
    assert np.isfinite(step_out.reward)
    env.close()


def test_make_env_registration() -> None:
    env = make_env("mpe_reference_pairs", k=3, graph="true", seed=0)
    assert isinstance(env, MPEReferencePairs)
    assert env.n_agents == 6
    env.close()


def test_k_escalation_cell_5_and_honesty_anchor_cell_1() -> None:
    env5 = MPEReferencePairs(k=5, graph="true", seed=0)
    assert env5.n_agents == 10
    assert env5.obs_dim == 21
    env5.reset(seed=0)
    env5.step(np.zeros(10, dtype=np.int64))
    env5.close()

    env1 = MPEReferencePairs(k=1, graph="true", comm_on=True, seed=0)
    assert env1.n_agents == 2
    assert env1.n_actions == 50  # full native action space
    env1.reset(seed=0)
    env1.step(np.zeros(2, dtype=np.int64))
    env1.close()


# ------------------------------------------------------------------- privacy


def test_comm_slot_zeroed_by_default() -> None:
    env = MPEReferencePairs(k=3, graph="true", seed=0)
    out = env.reset(seed=0)
    assert np.all(out.obs[:, 11:21] == 0.0)

    rng = np.random.default_rng(1)
    for _ in range(5):
        actions = rng.integers(0, env.n_actions, size=env.n_agents)
        step_out = env.step(actions)
        assert np.all(step_out.obs[:, 11:21] == 0.0), "comm slot leaked with comm_on=False"
    env.close()


def test_comm_slot_not_zeroed_when_comm_on() -> None:
    env = MPEReferencePairs(k=1, graph="true", comm_on=True, seed=0)
    env.reset(seed=0)
    # move=0 for both agents; agent_0 says symbol 7 (action = move + 5*say).
    actions = np.array([0 + 5 * 7, 0], dtype=np.int64)
    out = env.step(actions)
    # agent_1's comm slot should reflect agent_0's chosen say symbol.
    expected = np.zeros(10, dtype=np.float32)
    expected[7] = 1.0
    assert np.array_equal(out.obs[1, 11:21], expected)
    env.close()


def test_own_target_absent_partner_holds_it() -> None:
    """The [8:11] "goal_id" slot is the PARTNER's target, never the agent's
    own -- verified against the underlying mpe2 world state, not just the
    wrapper's bookkeeping."""
    env = MPEReferencePairs(k=1, graph="true", seed=0)
    out = env.reset(seed=0)

    world = env._envs[0].unwrapped.world
    agent0, agent1 = world.agents[0], world.agents[1]

    # agent0's own obs[8:11] must equal the landmark agent0 KNOWS about
    # (agent0.goal_b, which agent1 -- agent0.goal_a -- must reach), i.e. the
    # PARTNER's target, not agent0's own.
    assert np.allclose(out.obs[0, 8:11], agent0.goal_b.color)
    assert np.allclose(out.obs[1, 8:11], agent1.goal_b.color)

    # agent0's OWN target (the landmark agent0 itself must reach) is held by
    # agent1 (agent1.goal_b, since agent1.goal_a == agent0) -- and it is
    # nowhere in agent0's own observation.
    own_target_for_agent0 = agent1.goal_b.color
    assert not np.allclose(out.obs[0, 8:11], own_target_for_agent0)
    # It IS present in agent1's own slot (that's the point: only the
    # partner holds it).
    assert np.allclose(out.obs[1, 8:11], own_target_for_agent0)
    env.close()


def test_true_graph_neighbor_is_exactly_who_holds_your_target_all_pairs() -> None:
    """Locks the two halves the privacy and adjacency tests verify separately:
    for EVERY agent g (not just pair 0), its `graph="true"` neighbor is
    exactly the agent whose own `[8:11]` slot holds g's target."""
    env = MPEReferencePairs(k=3, graph="true", seed=0)
    out = env.reset(seed=0)
    adj = env.adjacency()

    for kk in range(env.k):
        world = env._envs[kk].unwrapped.world
        agent0, agent1 = world.agents[0], world.agents[1]
        g0, g1 = 2 * kk, 2 * kk + 1

        # true-graph neighbor of g0 is g1, and vice versa.
        assert adj[g0, g1] == 1.0
        assert adj[g1, g0] == 1.0

        # g0's own target (what g0 itself must reach) is held in g1's own
        # [8:11] slot -- i.e. exactly g0's true-graph neighbor holds it.
        own_target_for_g0 = agent1.goal_b.color  # agent1.goal_a == agent0
        own_target_for_g1 = agent0.goal_b.color  # agent0.goal_a == agent1
        assert np.allclose(out.obs[g1, 8:11], own_target_for_g0)
        assert np.allclose(out.obs[g0, 8:11], own_target_for_g1)
    env.close()


def test_oracle_mode_exposes_partner_goal_and_widens_obs() -> None:
    env = MPEReferencePairs(k=3, graph="true", oracle=True, seed=0)
    assert env.obs_dim == 24
    out = env.reset(seed=0)
    assert out.obs.shape == (6, 24)

    world = env._envs[0].unwrapped.world
    agent0, agent1 = world.agents[0], world.agents[1]
    own_target_for_agent0 = agent1.goal_b.color  # what agent0 itself must reach
    own_target_for_agent1 = agent0.goal_b.color

    # The oracle concatenation is exactly the partner's own goal_id slot,
    # i.e. this agent's own target -- routing done for free.
    assert np.allclose(out.obs[0, 21:24], own_target_for_agent0)
    assert np.allclose(out.obs[1, 21:24], own_target_for_agent1)
    # And the non-oracle mlp control never sees this: without oracle=True
    # the base 21-dim block never contains it (already checked above).
    env.close()


def test_oracle_position_mode_injects_relative_position_not_color() -> None:
    """AMENDMENT 1 (results/phaseD_external/PREREGISTRATION.md): diagnostic-
    only oracle_mode="position" injects the same own-target landmark as
    oracle_mode="color", but its RELATIVE POSITION (read from the mpe2 world
    state) rather than its color -- obs_dim shrinks to 23 (2 floats) instead
    of 24 (3 floats), and the injected values must match
    partner.goal_b.state.p_pos - self.state.p_pos exactly (the same
    computation scripts/phaseD_greedy_reference.py's greedy_target policy
    uses for its own-target displacement)."""
    env = MPEReferencePairs(k=3, graph="true", oracle=True, oracle_mode="position", seed=0)
    assert env.obs_dim == 23
    out = env.reset(seed=0)
    assert out.obs.shape == (6, 23)

    for kk in range(env.k):
        world = env._envs[kk].unwrapped.world
        a0, a1 = world.agents[0], world.agents[1]
        g0, g1 = 2 * kk, 2 * kk + 1

        expected_g0 = np.asarray(a1.goal_b.state.p_pos) - np.asarray(a0.state.p_pos)
        expected_g1 = np.asarray(a0.goal_b.state.p_pos) - np.asarray(a1.state.p_pos)
        assert np.allclose(out.obs[g0, 21:23], expected_g0, atol=1e-5)
        assert np.allclose(out.obs[g1, 21:23], expected_g1, atol=1e-5)

        # Sanity: this must NOT equal the color-mode injection (different
        # quantity, different units) -- guards against a copy-paste bug that
        # silently reused the color slice.
        own_target_color_g0 = a1.goal_b.color
        assert not np.allclose(out.obs[g0, 21:23], own_target_color_g0[:2], atol=1e-3)

    # Position mode must still respect the comm-disable contract.
    assert np.all(out.obs[:, 11:21] == 0.0)

    step_out = env.step(np.zeros(6, dtype=np.int64))
    assert step_out.obs.shape == (6, 23)
    assert np.all(np.isfinite(step_out.obs))
    env.close()


def test_oracle_mode_default_is_color_and_unchanged() -> None:
    """Default oracle_mode="color" must reproduce the pre-existing
    obs_dim=24 behavior byte-for-byte -- no change to the already-authorized
    confirmatory `oracle` arm."""
    env_default = MPEReferencePairs(k=3, graph="true", oracle=True, seed=0)
    env_explicit = MPEReferencePairs(
        k=3, graph="true", oracle=True, oracle_mode="color", seed=0
    )
    assert env_default.obs_dim == env_explicit.obs_dim == 24
    out_default = env_default.reset(seed=0)
    out_explicit = env_explicit.reset(seed=0)
    assert np.array_equal(out_default.obs, out_explicit.obs)
    env_default.close()
    env_explicit.close()


def test_invalid_oracle_mode_raises() -> None:
    with pytest.raises(ValueError, match="oracle_mode must be one of"):
        MPEReferencePairs(k=3, graph="true", oracle=True, oracle_mode="nonsense", seed=0)


def test_oracle_mode_comm_still_zeroed_by_default() -> None:
    env = MPEReferencePairs(k=3, graph="true", oracle=True, seed=0)
    out = env.reset(seed=0)
    assert np.all(out.obs[:, 11:21] == 0.0)
    env.close()


# ---------------------------------------------------------------- adjacency


def test_adjacency_true_is_block_diagonal_matching() -> None:
    env = MPEReferencePairs(k=3, graph="true", seed=0)
    adj = env.adjacency()
    expected = np.zeros((6, 6), dtype=np.float32)
    for k in range(3):
        expected[2 * k, 2 * k + 1] = 1.0
        expected[2 * k + 1, 2 * k] = 1.0
    assert np.array_equal(adj, expected)
    assert np.array_equal(adj, adj.T)
    assert np.all(np.diag(adj) == 0.0)
    env.close()


def test_adjacency_true_and_wrong_are_edge_disjoint_and_1_regular() -> None:
    true_env = MPEReferencePairs(k=3, graph="true", seed=0)
    wrong_env = MPEReferencePairs(k=3, graph="wrong", seed=0)
    a_true = true_env.adjacency()
    a_wrong = wrong_env.adjacency()

    assert np.all(a_true.sum(axis=1) == 1.0), "true graph must be 1-regular"
    assert np.all(a_wrong.sum(axis=1) == 1.0), "wrong graph must be 1-regular"
    assert not np.any((a_true > 0) & (a_wrong > 0))  # edge-disjoint
    assert not np.array_equal(a_true, a_wrong)
    true_env.close()
    wrong_env.close()


def test_adjacency_complete_is_all_to_all() -> None:
    env = MPEReferencePairs(k=3, graph="complete", seed=0)
    adj = env.adjacency()
    expected = np.ones((6, 6), dtype=np.float32)
    np.fill_diagonal(expected, 0.0)
    assert np.array_equal(adj, expected)
    env.close()


def test_graph_wrong_requires_k_at_least_2() -> None:
    with pytest.raises(ValueError, match="k >= 2"):
        MPEReferencePairs(k=1, graph="wrong", seed=0)


def test_invalid_graph_raises() -> None:
    with pytest.raises(ValueError, match="graph must be one of"):
        MPEReferencePairs(k=3, graph="nonsense", seed=0)


# -------------------------------------------------------------- reward wiring


def test_reward_is_sum_of_native_pair_rewards() -> None:
    """The composed reward must equal the sum of the K independent
    sub-environments' own (local_ratio-mixed) pair rewards, reproduced
    standalone with the same derived sub-seeds and actions."""
    from mpe2 import simple_reference_v3

    from gnnmarl.envs.mpe_reference_pairs import _FIXED_SAY, _MOVE_DIM

    k = 2
    env = MPEReferencePairs(k=k, graph="true", seed=0)
    env.reset(seed=0)  # re-derives self._rng from seed=0

    # Replicate the sub-seed derivation exactly as the wrapper does it.
    rng = np.random.default_rng(0)
    sub_seeds = rng.integers(0, 2**31 - 1, size=k)

    standalone_envs = [
        simple_reference_v3.parallel_env(local_ratio=0.5, max_cycles=25) for _ in range(k)
    ]
    for e, s in zip(standalone_envs, sub_seeds):
        e.reset(seed=int(s))

    rng_actions = np.random.default_rng(2)
    actions = rng_actions.integers(0, env.n_actions, size=env.n_agents)

    step_out = env.step(actions)

    expected_total = 0.0
    for i, e in enumerate(standalone_envs):
        a0 = int(actions[2 * i]) + _MOVE_DIM * _FIXED_SAY
        a1 = int(actions[2 * i + 1]) + _MOVE_DIM * _FIXED_SAY
        _obs, rewards, _term, _trunc, _info = e.step({"agent_0": a0, "agent_1": a1})
        expected_total += float(np.mean(list(rewards.values())))
        e.close()

    assert step_out.reward == pytest.approx(expected_total, abs=1e-5)
    env.close()


# --------------------------------------------------------------- determinism


def test_determinism_same_seed_same_first_episode_trajectory() -> None:
    def rollout(seed: int, actions_seq: list[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
        env = MPEReferencePairs(k=3, graph="true", seed=seed)
        res = env.reset(seed=seed)
        obs_list = [res.obs.copy()]
        rewards = []
        for a in actions_seq:
            step_out = env.step(a)
            obs_list.append(step_out.obs.copy())
            rewards.append(step_out.reward)
        env.close()
        return obs_list, rewards

    rng = np.random.default_rng(123)
    actions_seq = [rng.integers(0, 5, size=6) for _ in range(5)]

    obs1, rw1 = rollout(42, actions_seq)
    obs2, rw2 = rollout(42, actions_seq)

    assert all(np.array_equal(a, b) for a, b in zip(obs1, obs2))
    assert rw1 == pytest.approx(rw2)


# ----------------------------------------------------------------- liveness


@pytest.mark.parametrize("algo", ["gnn_qmix", "mlp_qmix"])
def test_short_training_liveness_k3(algo: str, tmp_path) -> None:
    """A few hundred env steps of gnn_true / mlp on K=3 trains without
    crashing and produces finite losses (docs/EXTERNAL_VALIDITY_DESIGN.md
    section 6's harness-trains-MPE sanity, cheap version)."""
    from gnnmarl.training import TrainConfig, train

    cfg = TrainConfig(
        env="mpe_reference_pairs",
        env_kwargs={"k": 3, "graph": "true", "max_cycles": 10, "seed": 0},
        algo=algo,
        algo_kwargs={
            "buffer_capacity": 500,
            "batch_size": 16,
            "target_update_interval": 50,
            "gnn_layers": 2,
            "gnn_hidden": 32,
        },
        total_env_steps=300,
        warmup_steps=50,
        eps_start=1.0,
        eps_end=0.1,
        eps_anneal_steps=250,
        seed=3,
        log_dir=tmp_path,
    )
    run_dir = train(cfg)

    import pandas as pd

    df = pd.read_csv(run_dir / "episodes.csv")
    assert len(df) > 0
    losses = df["loss"].dropna()
    assert len(losses) > 0, "no gradient steps were taken"
    assert np.all(np.isfinite(losses.to_numpy(dtype=float)))
