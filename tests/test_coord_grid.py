"""Unit tests for ``gnnmarl.envs.coord_grid.CoordGrid``."""

from __future__ import annotations

import numpy as np
import pytest

from gnnmarl.envs import CoordGrid, make_env
from gnnmarl.envs.base import MultiAgentEnv, StepResult


GRAPH_CASES = [
    ("ring", {"n_agents": 6}),
    ("line", {"n_agents": 6}),
    ("complete", {"n_agents": 6}),
    ("erdos_renyi", {"n_agents": 6, "er_prob": 0.5}),
    ("grid2d", {"n_agents": 9}),
]


@pytest.mark.parametrize("graph,kwargs", GRAPH_CASES)
def test_shapes(graph: str, kwargs: dict) -> None:
    env = CoordGrid(graph=graph, seed=0, **kwargs)
    assert isinstance(env, MultiAgentEnv)
    n = env.n_agents

    out = env.reset(seed=0)
    assert isinstance(out, StepResult)
    assert out.obs.shape == (n, env.obs_dim)
    assert out.obs.dtype == np.float32
    assert out.state.shape == (env.state_dim,)
    assert out.state.dtype == np.float32
    assert out.reward == 0.0
    assert out.done is False
    assert out.info["episode_step"] == 0

    adj = env.adjacency()
    assert adj.shape == (n, n)
    assert adj.dtype == np.float32

    # A step preserves shapes.
    rng = np.random.default_rng(123)
    actions = rng.integers(0, env.n_actions, size=n)
    nxt = env.step(actions)
    assert nxt.obs.shape == (n, env.obs_dim)
    assert nxt.state.shape == (env.state_dim,)


def test_determinism() -> None:
    env_a = CoordGrid(n_agents=4, grid_size=5, graph="ring", seed=42)
    env_b = CoordGrid(n_agents=4, grid_size=5, graph="ring", seed=42)
    a0 = env_a.reset(seed=42)
    b0 = env_b.reset(seed=42)
    np.testing.assert_array_equal(a0.obs, b0.obs)
    np.testing.assert_array_equal(a0.state, b0.state)
    assert a0.reward == b0.reward
    assert a0.done == b0.done

    action_rng = np.random.default_rng(2024)
    for _ in range(10):
        actions = action_rng.integers(0, env_a.n_actions, size=env_a.n_agents)
        ra = env_a.step(actions)
        rb = env_b.step(actions)
        np.testing.assert_array_equal(ra.obs, rb.obs)
        np.testing.assert_array_equal(ra.state, rb.state)
        assert ra.reward == rb.reward
        assert ra.done == rb.done


@pytest.mark.parametrize("graph,kwargs", GRAPH_CASES)
def test_adjacency_properties(graph: str, kwargs: dict) -> None:
    env = CoordGrid(graph=graph, seed=7, **kwargs)
    env.reset(seed=7)
    adj = env.adjacency()
    # Symmetric.
    np.testing.assert_array_equal(adj, adj.T)
    # Zero diagonal.
    np.testing.assert_array_equal(np.diag(adj), np.zeros(env.n_agents, dtype=adj.dtype))
    # 0/1 valued.
    unique = np.unique(adj)
    assert set(unique.tolist()).issubset({0.0, 1.0})


def test_ring_topology() -> None:
    env = CoordGrid(n_agents=6, graph="ring", seed=0)
    env.reset(seed=0)
    adj = env.adjacency()
    # Six edges → upper-triangular sum is 6.
    assert int(np.triu(adj, k=1).sum()) == 6
    # Every node has degree 2.
    np.testing.assert_array_equal(adj.sum(axis=1), np.full(6, 2.0, dtype=np.float32))


def test_reward_when_colocated() -> None:
    env = CoordGrid(n_agents=2, grid_size=5, graph="ring", seed=0)
    env.reset(seed=0)
    # Force colocation, then take stay actions so they remain colocated.
    env._place_agents(np.array([[2, 2], [2, 2]]))
    out = env.step(np.zeros(2, dtype=np.int64))  # 0 = stay
    assert out.reward > 0.0
    assert out.reward == pytest.approx(1.0)


def test_reward_zero_when_apart() -> None:
    env = CoordGrid(n_agents=2, grid_size=5, graph="ring", seed=0)
    env.reset(seed=0)
    env._place_agents(np.array([[0, 0], [3, 3]]))
    out = env.step(np.zeros(2, dtype=np.int64))  # both stay
    assert out.reward == 0.0


def test_episode_termination() -> None:
    steps = 5
    env = CoordGrid(n_agents=3, grid_size=4, episode_steps=steps, graph="ring", seed=0)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    dones = []
    for _ in range(steps):
        actions = rng.integers(0, env.n_actions, size=env.n_agents)
        out = env.step(actions)
        dones.append(out.done)
    assert dones[:-1] == [False] * (steps - 1)
    assert dones[-1] is True


def test_grid2d_requires_square() -> None:
    with pytest.raises(ValueError):
        CoordGrid(n_agents=5, graph="grid2d")


def test_make_env_factory() -> None:
    env = make_env("coord_grid", n_agents=4, graph="ring", seed=0)
    assert isinstance(env, CoordGrid)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        make_env("unknown_env")


# --------------------------------------------------------------- obs_mode (Stage A)

def _obs_blocks(env: CoordGrid) -> tuple[slice, slice, slice]:
    """Slices for the own-position, neighbour, and agent-id blocks of an obs row."""
    mn = env.max_neighbors
    return slice(0, 2), slice(2, 2 + 2 * mn), slice(2 + 2 * mn, env.obs_dim)


@pytest.mark.parametrize("mode", ["full", "ego", "radius"])
def test_obs_mode_keeps_obs_dim_constant(mode: str) -> None:
    # Every algorithm must see the same input shape regardless of observability,
    # so the GNN-vs-control contrast is not confounded with input dimension.
    extra = {"obs_radius": 1} if mode == "radius" else {}
    env = CoordGrid(n_agents=4, grid_size=5, graph="ring", seed=0, obs_mode=mode, **extra)
    baseline = CoordGrid(n_agents=4, grid_size=5, graph="ring", seed=0)
    assert env.obs_dim == baseline.obs_dim
    out = env.reset(seed=0)
    assert out.obs.shape == (4, env.obs_dim)


def test_ego_withholds_neighbours_but_keeps_self_and_id() -> None:
    env = CoordGrid(n_agents=4, grid_size=5, graph="ring", seed=0, obs_mode="ego")
    env.reset(seed=0)
    env._place_agents([[0, 0], [1, 0], [2, 0], [3, 0]])
    obs = env._compute_obs()
    own, nbr, idblk = _obs_blocks(env)
    # The graph is now the only channel to neighbours: neighbour block is empty.
    np.testing.assert_array_equal(obs[:, nbr], 0.0)
    # Self-position survives for off-origin agents, and the id one-hot is intact.
    assert np.any(obs[:, own] != 0.0)
    np.testing.assert_array_equal(obs[:, idblk].sum(axis=1), np.ones(4, dtype=np.float32))


def test_full_reveals_neighbours() -> None:
    env = CoordGrid(n_agents=4, grid_size=5, graph="ring", seed=0, obs_mode="full")
    env.reset(seed=0)
    env._place_agents([[0, 0], [1, 0], [2, 0], [3, 0]])
    obs = env._compute_obs()
    _, nbr, _ = _obs_blocks(env)
    assert np.any(obs[:, nbr] != 0.0)


def test_radius_hides_far_neighbours_only() -> None:
    # Ring N=4: agent 0's neighbours are agents 1 and 3.
    layout = [[0, 0], [1, 0], [4, 4], [4, 0]]  # nbr 1 is 1 step away, nbr 3 is far
    env = CoordGrid(n_agents=4, grid_size=9, graph="ring", seed=0,
                    obs_mode="radius", obs_radius=1)
    env.reset(seed=0)
    env._place_agents(layout)
    obs = env._compute_obs()
    full = CoordGrid(n_agents=4, grid_size=9, graph="ring", seed=0, obs_mode="full")
    full.reset(seed=0)
    full._place_agents(layout)
    fobs = full._compute_obs()
    _, nbr, _ = _obs_blocks(env)
    # The near neighbour is visible; the far one is masked -> strictly fewer
    # non-zeros than full, but at least the near neighbour shows through.
    assert np.count_nonzero(obs[0, nbr]) >= 1
    assert np.count_nonzero(obs[0, nbr]) < np.count_nonzero(fobs[0, nbr])


def test_invalid_obs_mode_raises() -> None:
    with pytest.raises(ValueError):
        CoordGrid(obs_mode="bogus")


def test_radius_mode_requires_valid_radius() -> None:
    with pytest.raises(ValueError):
        CoordGrid(obs_mode="radius")  # missing obs_radius
    with pytest.raises(ValueError):
        CoordGrid(obs_mode="radius", obs_radius=-1)


def test_make_env_threads_obs_mode() -> None:
    env = make_env("coord_grid", n_agents=4, graph="ring", obs_mode="ego")
    assert env.obs_mode == "ego"


# ----------------------------------------------------- goal-routing task (Stage A)

def test_goal_routing_adds_three_obs_slots() -> None:
    base = CoordGrid(n_agents=4, grid_size=5, graph="ring", seed=0)
    gr = CoordGrid(n_agents=4, grid_size=5, graph="ring", seed=0, goal_routing=True)
    assert gr.obs_dim == base.obs_dim + 3


def test_only_source_agent_sees_goal() -> None:
    env = CoordGrid(n_agents=4, grid_size=5, graph="ring", seed=0, goal_routing=True)
    env.reset(seed=0)
    obs = env._compute_obs()
    goal = obs[:, -3:]  # [has_goal, gx, gy]
    # Source agent (0) has the flag set; everyone else's goal block is zero.
    assert goal[0, 0] == 1.0
    np.testing.assert_array_equal(goal[1:], np.zeros((env.n_agents - 1, 3), dtype=np.float32))


def test_goal_routing_reward_counts_agents_at_goal() -> None:
    env = CoordGrid(n_agents=4, grid_size=5, graph="ring", seed=0, goal_routing=True)
    env.reset(seed=0)
    g = env._goal
    # Put two agents on the goal cell, two elsewhere.
    other = (g + np.array([1, 0])) % env.grid_size
    env._place_agents([g, g, other, other])
    assert env._compute_reward() == 2.0
    # Goal is a real cell, so a co-located non-goal reward path is not triggered.
    env._place_agents([other, other, other, other])
    assert env._compute_reward() == 0.0


def test_goal_dense_reward_decreases_with_distance() -> None:
    env = CoordGrid(n_agents=2, grid_size=5, graph="ring", seed=0,
                    goal_routing=True, goal_dense=True)
    env.reset(seed=0)
    g = env._goal
    env._place_agents([g, g])
    assert env._compute_reward() == 2.0  # both on goal -> max
    far = (g + np.array([2, 2])) % env.grid_size
    env._place_agents([g, far])
    r = env._compute_reward()
    assert 1.0 <= r < 2.0  # one on goal (1.0) + one farther (<1.0)


# --------------------------------------------------------------- pair_routing
# Stage C structure test: comm graph must match the task pairing (see docs/STAGE_C.md).


def test_pair_routing_obs_dim_constant_across_comm_graphs() -> None:
    # With max_neighbors_override fixed, the true/wrong/complete comm graphs all
    # yield the SAME obs_dim, so the structure arms share byte-identical I/O.
    dims = {
        g: CoordGrid(n_agents=6, grid_size=3, graph=g, pair_routing=True,
                     obs_mode="ego", max_neighbors_override=5, seed=0).obs_dim
        for g in ("matching", "matching_wrong", "complete")
    }
    assert len(set(dims.values())) == 1, dims


def test_pair_routing_partner_is_canonical_matching_regardless_of_graph() -> None:
    # The reward partner is the fixed matching (0,1),(2,3),(4,5) for ANY comm graph.
    for g in ("matching", "matching_wrong", "complete"):
        env = CoordGrid(n_agents=6, grid_size=3, graph=g, pair_routing=True, seed=0)
        assert env._partner.tolist() == [1, 0, 3, 2, 5, 4]


def test_matching_graphs_are_disjoint_and_one_regular() -> None:
    a_true = CoordGrid(n_agents=6, grid_size=3, graph="matching", seed=0).adjacency()
    a_wrong = CoordGrid(n_agents=6, grid_size=3, graph="matching_wrong", seed=0).adjacency()
    # Both are perfect matchings (each node degree 1) ...
    np.testing.assert_array_equal(a_true.sum(axis=1), np.ones(6))
    np.testing.assert_array_equal(a_wrong.sum(axis=1), np.ones(6))
    # ... and share no edge, so "wrong" pairs every agent with a non-partner.
    assert np.all(a_true * a_wrong == 0.0)
    env = CoordGrid(n_agents=6, grid_size=3, graph="matching", pair_routing=True, seed=0)
    for i in range(6):
        assert np.where(a_true[i] > 0)[0].tolist() == [env._partner[i]]
        assert env._partner[i] not in np.where(a_wrong[i] > 0)[0].tolist()


def test_pair_routing_every_agent_sees_only_its_own_goal_under_ego() -> None:
    env = CoordGrid(n_agents=6, grid_size=3, graph="matching", pair_routing=True,
                    obs_mode="ego", max_neighbors_override=5, seed=0)
    env.reset(seed=1)
    obs = env._compute_obs()
    goal_block = obs[:, -3:]  # [has_goal, gx, gy] per agent
    for i in range(env.n_agents):
        assert goal_block[i, 0] == 1.0  # every agent has its own goal
        np.testing.assert_allclose(goal_block[i, 1:], env._goals[i] / env.grid_size)
    # Neighbour block is withheld under ego (no agent sees another's position).
    nbr = obs[:, 2:2 + 2 * env.max_neighbors]
    np.testing.assert_array_equal(nbr, np.zeros_like(nbr))


def test_pair_routing_reward_max_when_all_on_partner_goal() -> None:
    env = CoordGrid(n_agents=6, grid_size=3, graph="matching", pair_routing=True, seed=0)
    env.reset(seed=2)
    env._place_agents(env._goals[env._partner])  # each agent on its partner's goal
    assert env._compute_reward() == pytest.approx(6.0)  # == N, maximal
    # Sitting all on a single corner over a fixed goal set is below max.
    env._place_agents(np.zeros((6, 2), dtype=np.int64))
    assert env._compute_reward() < 6.0


@pytest.mark.parametrize("kwargs,match", [
    ({"n_agents": 5, "pair_routing": True}, "even"),
    ({"n_agents": 5, "graph": "matching"}, "even"),
    ({"n_agents": 2, "graph": "matching_wrong"}, ">= 4"),
    ({"n_agents": 4, "pair_routing": True, "goal_routing": True}, "mutually exclusive"),
])
def test_pair_routing_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CoordGrid(grid_size=3, seed=0, **kwargs)
