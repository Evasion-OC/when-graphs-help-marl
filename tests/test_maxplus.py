"""Tests for the DCG classical coordination-graph baseline (``algos/maxplus.py``).

Covers the validity gates from ``docs/CLASSICAL_CG_BASELINE_DESIGN.md`` section
6: max-plus exactness on tree graphs (brute force), message-norm convergence on
loopy graphs, DCG's parameter count vs GNN-QMIX at the matched PairRouting
config, the ``act()``/update contracts, and end-to-end training on both
``pair_routing`` and ``relay_routing``.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from gnnmarl.algos import make_algo
from gnnmarl.algos.maxplus import (
    DCG,
    count_params,
    max_plus_action_select,
    q_tot_from_actions,
)
from gnnmarl.training import TrainConfig, train

N_AGENTS = 6
OBS_DIM = 21
STATE_DIM = 12
N_ACTIONS = 5


def _make(name: str = "dcg", **kw):
    base = dict(
        n_agents=N_AGENTS,
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        device=torch.device("cpu"),
        seed=0,
    )
    base.update(kw)
    return make_algo(name, **base)


def _symmetrize(qp_raw: torch.Tensor) -> torch.Tensor:
    return 0.5 * (qp_raw + qp_raw.permute(0, 2, 1, 4, 3))


def _adj_matching(n: int) -> torch.Tensor:
    a = torch.zeros(n, n)
    for k in range(n // 2):
        a[2 * k, 2 * k + 1] = 1.0
        a[2 * k + 1, 2 * k] = 1.0
    return a


def _adj_chain(n: int) -> torch.Tensor:
    a = torch.zeros(n, n)
    for i in range(n - 1):
        a[i, i + 1] = 1.0
        a[i + 1, i] = 1.0
    return a


def _adj_ring(n: int) -> torch.Tensor:
    a = torch.zeros(n, n)
    for i in range(n):
        a[i, (i + 1) % n] = 1.0
        a[(i + 1) % n, i] = 1.0
    return a


def _adj_complete(n: int) -> torch.Tensor:
    return torch.ones(n, n) - torch.eye(n)


def _brute_force_best_q_tot(U: torch.Tensor, Qp: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """Exhaustive joint-action search; returns the max achievable Q_tot per batch."""
    b, n, a = U.shape
    best = torch.full((b,), float("-inf"))
    for joint in itertools.product(range(a), repeat=n):
        actions = torch.tensor(joint, dtype=torch.int64).unsqueeze(0).expand(b, n)
        q = q_tot_from_actions(U, Qp, actions, adj)
        best = torch.maximum(best, q)
    return best


# --------------------------------------------------------------- max-plus exactness


@pytest.mark.parametrize(
    "adj_fn,n,a",
    [
        (_adj_matching, 4, 3),  # disjoint matching: two independent 2-cliques
        (_adj_chain, 3, 3),  # 3-chain
    ],
)
def test_max_plus_matches_brute_force_on_trees(adj_fn, n, a) -> None:
    torch.manual_seed(0)
    b = 3
    U = torch.randn(b, n, a)
    qp_raw = torch.randn(b, n, n, a, a)
    Qp = _symmetrize(qp_raw)
    adj = adj_fn(n).unsqueeze(0).expand(b, n, n).contiguous()

    actions, _ = max_plus_action_select(U, Qp, adj, rounds=8)
    achieved = q_tot_from_actions(U, Qp, actions, adj)
    best = _brute_force_best_q_tot(U, Qp, adj)

    # Robust to ties in the argmax: compare the VALUE achieved, not the indices
    # (design section 6 / advisor guidance: exact on trees means exact value).
    torch.testing.assert_close(achieved, best, atol=1e-4, rtol=1e-4)


def test_max_plus_matches_brute_force_multiple_seeds() -> None:
    torch.manual_seed(123)
    n, a, b = 3, 3, 4
    adj = _adj_chain(n).unsqueeze(0).expand(b, n, n).contiguous()
    for _ in range(5):
        U = torch.randn(b, n, a)
        Qp = _symmetrize(torch.randn(b, n, n, a, a))
        actions, _ = max_plus_action_select(U, Qp, adj, rounds=8)
        achieved = q_tot_from_actions(U, Qp, actions, adj)
        best = _brute_force_best_q_tot(U, Qp, adj)
        torch.testing.assert_close(achieved, best, atol=1e-4, rtol=1e-4)


# ------------------------------------------------------- message-norm convergence


@pytest.mark.parametrize(
    "adj_fn,n",
    [(_adj_ring, 5), (_adj_complete, 4)],
)
def test_max_plus_message_norm_converges_on_loopy_graphs(adj_fn, n) -> None:
    # On graphs with cycles max-plus is not guaranteed to reach a fixed
    # point (design section 1: "loopy-graph stability" via mean-subtracted
    # messages, not exact convergence) -- it can settle into a stable
    # limit cycle whose level is unrelated to the initial transient. The
    # correct convergence statistic (design section 6: "message-norm
    # plateau") is therefore that the message-norm VARIANCE shrinks (a
    # plateau, whatever its level), not that the level itself decreases.
    torch.manual_seed(0)
    a, b = 3, 2
    U = torch.randn(b, n, a)
    Qp = _symmetrize(torch.randn(b, n, n, a, a))
    adj = adj_fn(n).unsqueeze(0).expand(b, n, n).contiguous()

    _, _, norms = max_plus_action_select(U, Qp, adj, rounds=20, return_message_norms=True)
    assert len(norms) == 20
    assert all(math.isfinite(x) for x in norms)
    early_std = float(np.std(norms[:5]))
    late_std = float(np.std(norms[-5:]))
    assert late_std <= early_std + 1e-6, (
        f"message norm did not plateau: early_std={early_std} late_std={late_std}"
    )


def test_max_plus_shape_validation() -> None:
    b, n, a = 2, 3, 4
    U = torch.randn(b, n, a)
    Qp = torch.randn(b, n, n, a, a)
    adj = torch.zeros(b, n, n)
    with pytest.raises(ValueError, match="U\\[B,N,A\\]"):
        max_plus_action_select(torch.randn(n, a), Qp, adj)
    with pytest.raises(ValueError, match="Qp\\[B,N,N,A,A\\]"):
        max_plus_action_select(U, torch.randn(b, n, n, a, a + 1), adj)
    with pytest.raises(ValueError, match="adj\\[B,N,N\\]"):
        max_plus_action_select(U, Qp, torch.randn(b, n, n + 1))
    with pytest.raises(ValueError, match="rounds must be >= 1"):
        max_plus_action_select(U, Qp, adj, rounds=0)


# --------------------------------------------------------------------- param counts


def test_dcg_conditioning_validation() -> None:
    with pytest.raises(ValueError, match="conditioning must be one of"):
        _make(conditioning="bogus")
    with pytest.raises(ValueError, match="mp_rounds must be >= 1"):
        _make(mp_rounds=0)


@pytest.mark.parametrize("conditioning", ["jointobs", "canonical", "oracle_state"])
def test_dcg_has_at_least_as_many_params_as_gnn_qmix(conditioning: str) -> None:
    # The matched PairRouting-ego config (N=6, max_neighbors_override=5):
    # obs_dim=21, state_dim=12 (see tests/test_coord_grid.py). "Generosity
    # toward DCG" (design section 3): total DCG params >= GNN-QMIX at
    # gnn_hidden=64, gnn_layers=2 for EVERY conditioning arm.
    dcg = _make("dcg", conditioning=conditioning, gnn_hidden=64)
    gnn = _make("gnn_qmix", gnn_layers=2, gnn_hidden=64)
    assert count_params(dcg) >= count_params(gnn), (
        f"conditioning={conditioning}: dcg={count_params(dcg)} < "
        f"gnn_qmix={count_params(gnn)}"
    )


def test_dcg_has_no_mixer_module() -> None:
    # DCG's Q_tot is the factored average (U + pairwise); unlike QMIX/GNN-QMIX
    # there is no separate learned monotonic mixer.
    dcg = _make("dcg")
    assert not hasattr(dcg, "mixer")


# --------------------------------------------------------------------- act() contract


def _chain_adj(n_agents: int = N_AGENTS) -> np.ndarray:
    a = np.zeros((n_agents, n_agents), dtype=np.float32)
    for k in range(n_agents // 3):
        s, r, t = 3 * k, 3 * k + 1, 3 * k + 2
        a[s, r] = a[r, s] = 1.0
        a[r, t] = a[t, r] = 1.0
    return a


@pytest.mark.parametrize("conditioning", ["jointobs", "canonical", "oracle_state"])
def test_dcg_act_contract(conditioning: str) -> None:
    algo = _make(conditioning=conditioning)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(N_AGENTS, OBS_DIM)).astype(np.float32)
    adj = _chain_adj()
    a = algo.act(obs, adj, epsilon=0.0, rng=rng)
    assert isinstance(a, np.ndarray)
    assert a.dtype == np.int64
    assert a.shape == (N_AGENTS,)
    assert a.min() >= 0 and a.max() < N_ACTIONS


def test_dcg_eps_one_is_purely_rng_determined() -> None:
    algo = _make()
    obs = np.zeros((N_AGENTS, OBS_DIM), dtype=np.float32)
    adj = _chain_adj()

    rng = np.random.default_rng(42)
    actions_1 = algo.act(obs, adj, epsilon=1.0, rng=rng)
    actions_2 = algo.act(obs, adj, epsilon=1.0, rng=rng)

    expected_rng = np.random.default_rng(42)
    mask_1 = expected_rng.random(N_AGENTS) < 1.0
    rand_1 = expected_rng.integers(low=0, high=N_ACTIONS, size=N_AGENTS)
    mask_2 = expected_rng.random(N_AGENTS) < 1.0
    rand_2 = expected_rng.integers(low=0, high=N_ACTIONS, size=N_AGENTS)

    assert mask_1.all() and mask_2.all()
    assert np.array_equal(actions_1, rand_1.astype(np.int64))
    assert np.array_equal(actions_2, rand_2.astype(np.int64))


def test_dcg_act_shape_validation() -> None:
    algo = _make()
    rng = np.random.default_rng(0)
    adj = _chain_adj()
    with pytest.raises(ValueError, match="act\\(\\) expects obs"):
        algo.act(np.zeros((N_AGENTS + 1, OBS_DIM), dtype=np.float32), adj, epsilon=0.0, rng=rng)
    with pytest.raises(ValueError, match="act\\(\\) expects adj"):
        algo.act(
            np.zeros((N_AGENTS, OBS_DIM), dtype=np.float32),
            np.zeros((N_AGENTS, N_AGENTS + 1), dtype=np.float32),
            epsilon=0.0,
            rng=rng,
        )


# ------------------------------------------------------------------------ update()


def _fake_transition(rng: np.random.Generator, adj: np.ndarray):
    from gnnmarl.utils.replay import Transition

    obs = rng.normal(size=(N_AGENTS, OBS_DIM)).astype(np.float32)
    next_obs = rng.normal(size=(N_AGENTS, OBS_DIM)).astype(np.float32)
    state = rng.normal(size=(STATE_DIM,)).astype(np.float32)
    next_state = rng.normal(size=(STATE_DIM,)).astype(np.float32)
    actions = rng.integers(low=0, high=N_ACTIONS, size=N_AGENTS).astype(np.int64)
    reward = float(rng.normal())
    return Transition(
        obs=obs, state=state, actions=actions, reward=reward,
        next_obs=next_obs, next_state=next_state, done=False, adj=adj,
    )


@pytest.mark.parametrize("conditioning", ["jointobs", "canonical", "oracle_state"])
def test_dcg_update_runs(conditioning: str) -> None:
    algo = _make(conditioning=conditioning, batch_size=4)
    rng = np.random.default_rng(0)
    adj = _chain_adj()
    for _ in range(10):
        algo.observe(_fake_transition(rng, adj))

    saw_metrics = None
    for _ in range(5):
        metrics = algo.update()
        if metrics is not None:
            saw_metrics = metrics
            break

    assert saw_metrics is not None, "update() never produced a metrics dict"
    assert math.isfinite(saw_metrics["loss"])
    assert math.isfinite(saw_metrics["grad_norm"])


def test_dcg_target_update() -> None:
    target_update_interval = 3
    algo = _make(batch_size=4, target_update_interval=target_update_interval)
    rng = np.random.default_rng(1)
    adj = _chain_adj()
    for _ in range(10):
        algo.observe(_fake_transition(rng, adj))

    successful = 0
    safety = 0
    while successful < target_update_interval:
        if algo.update() is not None:
            successful += 1
        safety += 1
        if safety > 100:
            raise AssertionError("update() failed to produce metrics in 100 calls")

    online_flat = torch.cat([p.detach().reshape(-1) for p in algo.parameters()])
    target_flat = torch.cat([p.detach().reshape(-1) for p in algo._target.parameters()])
    assert online_flat.shape == target_flat.shape
    assert torch.allclose(online_flat, target_flat, atol=0.0, rtol=0.0)


def test_dcg_construct_and_factory() -> None:
    algo = make_algo(
        "dcg",
        n_agents=N_AGENTS, obs_dim=OBS_DIM, state_dim=STATE_DIM, n_actions=N_ACTIONS,
        device=torch.device("cpu"), seed=0,
    )
    assert isinstance(algo, DCG)
    assert algo.name == "dcg"
    assert algo.conditioning == "jointobs"  # default per design section 1


# --------------------------------------------------------------- end-to-end training


@pytest.mark.parametrize(
    "env_kwargs",
    [
        {
            "n_agents": 6, "grid_size": 3, "episode_steps": 10, "graph": "matching",
            "pair_routing": True, "obs_mode": "ego", "max_neighbors_override": 5,
        },
        {
            "n_agents": 6, "grid_size": 3, "episode_steps": 10, "graph": "relay",
            "relay_routing": True, "obs_mode": "ego", "max_neighbors_override": 5,
        },
    ],
    ids=["pair_routing", "relay_routing"],
)
def test_dcg_trains_without_error(env_kwargs: dict, tmp_path: Path) -> None:
    cfg = TrainConfig(
        env="coord_grid",
        env_kwargs=env_kwargs,
        algo="dcg",
        algo_kwargs={
            "buffer_capacity": 1_000, "batch_size": 16, "target_update_interval": 50,
            "mp_rounds": 4, "conditioning": "jointobs",
        },
        total_env_steps=300,
        warmup_steps=20,
        update_every=2,
        eps_start=1.0, eps_end=0.1, eps_anneal_steps=250,
        seed=3,
        log_dir=tmp_path,
    )
    run_dir = train(cfg)
    csv_path = run_dir / "episodes.csv"
    assert csv_path.exists()

    import pandas as pd

    df = pd.read_csv(csv_path)
    assert len(df) > 0
    assert df["return"].apply(math.isfinite).all()
    assert df["loss"].dropna().apply(math.isfinite).all()
