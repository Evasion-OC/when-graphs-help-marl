r"""MPEReferencePairs — K composed ``simple_reference`` pairs, comm disabled.

Implements the vehicle from ``docs/EXTERNAL_VALIDITY_DESIGN.md`` (section 9):
run the isolator family (``gnn_true`` / ``gnn_wrong`` / ``gnn_complete`` /
``mlp`` / ``oracle``) on K independent PettingZoo/``mpe2`` ``simple_reference``
sub-environments (N = 2K agents) with the scripted communication channel
disabled, so the coordination graph is the sole route for a partner's private
goal. This module imports ``mpe2`` (falling back to ``pettingzoo.mpe`` for
older installs) lazily — it lives behind the ``[mpe]`` extra so the core
package still imports without it.

**Why ``simple_reference`` gives us private-partner information natively.**
Each of the 2 agents in one ``simple_reference`` sub-environment observes a
``goal_id`` block that is the color of the landmark its *partner* must reach
(the partner's own reward is scored against that landmark). Crucially the
scenario's ``observation()`` never concatenates ``entity_color`` at all, only
relative landmark *positions* — so an agent's own target is not merely masked
by us, it is *structurally absent* from its own observation upstream in
mpe2; only the partner (who computes it as its own ``goal_id`` block) holds
it. Verified directly against ``mpe2/simple_reference/simple_reference.py``
(``Scenario.observation``) and empirically:
``obs['agent_0'][8:11] == world.agents[0].goal_b.color`` and
``world.agents[0].goal_b`` is landmark that ``world.agents[1]`` (agent_0's
``goal_a``) must reach — i.e. agent_0 knows agent_1's target, not its own.
Agent_1's own ``[8:11]`` slot is agent_0's own target.

**The exact per-sub-env observation layout** (21 floats, confirmed against
``Scenario.observation`` and by direct inspection of the constructed env)::

    obs = concat([self_vel(2), landmark_rel_pos(3 landmarks * 2 = 6),
                  goal_color(3), comm(10)])

Slice indices used throughout this module (0-based, half-open)::

    [0:2]   self_vel            -- own physical velocity, always kept.
    [2:8]   landmark_rel_pos    -- relative positions of the 3 landmarks
                                   (never their colors/identity -- own target
                                   color is genuinely nowhere in this obs).
    [8:11]  goal_color          -- "goal_id": color of the landmark the
                                   PARTNER must reach. This is the payload
                                   that must be routed to the partner over
                                   the coordination graph. NEVER zeroed.
    [11:21] comm                -- the partner's broadcast ``say`` symbol
                                   (one-hot, dim_c=10). ZEROED unless
                                   ``comm_on=True`` -- this is the whole
                                   point: with comm disabled, [8:11] can only
                                   reach the partner via the GNN's message
                                   passing over ``adjacency()``.

The underlying discrete action space is ``Discrete(50) = move(5) x say(10)``,
decoded by mpe2's ``simple_env.py`` as ``move = action % 5``,
``say = action // 5``. With comm disabled we expose only the 5-way movement
sub-action to the trainer and internally fix ``say`` to a constant symbol
(the exact constant is irrelevant since ``[11:21]`` is zeroed regardless of
what any agent actually said).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gnnmarl.envs.base import MultiAgentEnv, StepResult

# --- simple_reference obs layout (see module docstring; verified against
# mpe2/simple_reference/simple_reference.py::Scenario.observation). ---------
_SUBENV_OBS_DIM = 21
_GOAL_SLICE = slice(8, 11)  # partner's target landmark color ("goal_id").
_COMM_SLICE = slice(11, 21)  # native `say` broadcast; zeroed unless comm_on.

# --- simple_reference action layout (see mpe2/_mpe_utils/simple_env.py
# ``_execute_world_step``: ``scenario_action.append(action % mdim);
# action //= mdim`` then the remainder is the say index). -------------------
_MOVE_DIM = 5  # no_action/left/right/down/up
_SAY_DIM = 10  # world.dim_c
_NATIVE_N_ACTIONS = _MOVE_DIM * _SAY_DIM  # 50
_FIXED_SAY = 0  # constant say symbol when comm is off

_VALID_GRAPHS = ("true", "wrong", "complete")


class MPEReferencePairs(MultiAgentEnv):
    """Compose K independent ``simple_reference`` pairs into one 2K-agent env.

    Parameters
    ----------
    k
        Number of independent ``simple_reference`` sub-environments composed;
        ``n_agents = 2 * k``. Primary confirmatory cell is ``k=3``; ``k=5`` is
        the pre-specified escalation cell; ``k=1`` (N=2) is the non-confirmatory
        honesty-anchor cell, normally paired with ``comm_on=True``.
    graph
        Coordination graph exposed via :meth:`adjacency`, one of:

        * ``"true"``     — the K native pair edges (block-diagonal 1-regular
          matching ``2k <-> 2k+1``): each agent's edge is exactly its actual
          partner, i.e. the agent who holds its private target.
        * ``"wrong"``    — a 1-regular matching sharing NO edge with
          ``"true"``, same density. Reuses ``CoordGrid``'s ``matching_wrong``
          formula verbatim (``i = 2k+1, j = (2k+2) % N``): edge-disjoint from
          ``"true"`` for any ``k >= 2``; undefined at ``k=1`` (raises).
        * ``"complete"`` — all-to-all over the ``2k`` agents.

        ``obs_dim`` is identical across all three (adjacency is a channel
        fed to the GNN's message passing, entirely separate from the
        observation vector; with comm disabled the observation is byte-
        identical across ``graph`` values).
    oracle
        If ``True``, concatenate the partner's ``goal_id`` block (i.e. this
        agent's *own* target, which the partner holds — see module
        docstring) directly into each agent's observation, short-circuiting
        the need to route it over any graph. This is a descriptive headroom
        ceiling, not a controlled arm: ``obs_dim`` is ``24`` instead of
        ``21`` in this mode (that is expected and documented — see design
        doc section 3, "param-match its nets accordingly").
    comm_on
        If ``True``, leave the native ``say`` channel and the full
        ``Discrete(50)`` action space intact (``n_actions=50``) and do NOT
        zero the comm observation slot. Used only for the N=2 honesty-anchor
        cell (design doc section 8) — an unmodified ``simple_reference``
        sanity check, not part of the confirmatory family.
    max_cycles
        Episode length forwarded to each ``simple_reference`` sub-env.
    local_ratio
        ``simple_reference``'s local/global reward mixing weight; default
        0.5 (upstream default, matches ``mpe_adapter.py``'s convention).
    seed
        Forwarded to the first :meth:`reset`.
    """

    n_actions: int

    def __init__(
        self,
        k: int = 3,
        *,
        graph: str = "true",
        oracle: bool = False,
        comm_on: bool = False,
        max_cycles: int = 25,
        local_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if graph not in _VALID_GRAPHS:
            raise ValueError(f"graph must be one of {_VALID_GRAPHS}, got {graph!r}")
        if graph == "wrong" and k < 2:
            raise ValueError(
                "graph='wrong' requires k >= 2 (n_agents >= 4): a matching edge-"
                f"disjoint from the true pairing does not exist at k={k} (N=2, "
                "true and complete coincide -- see design doc section 2a)"
            )

        try:
            # simple_reference moved out of pettingzoo into the standalone
            # mpe2 package; prefer it, fall back to the old in-tree location
            # for older installs (mirrors mpe_adapter.py's convention).
            try:
                from mpe2 import simple_reference_v3
            except ImportError:
                from pettingzoo.mpe import simple_reference_v3
        except ImportError as e:
            raise ImportError(
                "MPEReferencePairs requires the 'mpe' extra; install with "
                "`pip install mpe2` (or `pip install -e .[mpe]`)"
            ) from e

        self.k = int(k)
        self.graph = graph
        self.oracle = bool(oracle)
        self.comm_on = bool(comm_on)
        self.max_cycles = int(max_cycles)
        self.local_ratio = float(local_ratio)

        self.n_agents = 2 * self.k
        self.obs_dim = _SUBENV_OBS_DIM + (3 if self.oracle else 0)
        self.state_dim = self.obs_dim * self.n_agents
        self.n_actions = _NATIVE_N_ACTIONS if self.comm_on else _MOVE_DIM

        self._local_agents = ("agent_0", "agent_1")
        self._envs = [
            simple_reference_v3.parallel_env(
                local_ratio=self.local_ratio,
                max_cycles=self.max_cycles,
                continuous_actions=False,
            )
            for _ in range(self.k)
        ]

        self._adj = self._build_adjacency()
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._last_obs_dicts: list[dict[str, Any]] = [dict() for _ in range(self.k)]
        self._episode_step = 0
        self._closed = False

        # Establish an initial observation (and validate the sub-env's obs
        # layout matches _SUBENV_OBS_DIM) so obs_dim / state_dim are backed
        # by a real reset before the caller's first explicit reset().
        self.reset(seed=seed)

    # ------------------------------------------------------------------ API

    def reset(self, *, seed: int | None = None) -> StepResult:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Derive K deterministic-but-distinct sub-seeds from the top-level
        # seed so a fresh instance with the same seed reproduces the same
        # first-episode trajectory (docs/REPRO.md).
        sub_seeds = self._rng.integers(0, 2**31 - 1, size=self.k)

        obs_dicts: list[dict[str, Any]] = []
        for env, sub_seed in zip(self._envs, sub_seeds):
            obs_dict, _info = env.reset(seed=int(sub_seed))
            obs_dicts.append(obs_dict)
        self._last_obs_dicts = obs_dicts
        self._episode_step = 0

        obs = self._compose_obs(obs_dicts)
        return StepResult(
            obs=obs,
            state=obs.reshape(-1).astype(np.float32),
            reward=0.0,
            done=False,
            info={"episode_step": 0},
        )

    def step(self, actions: np.ndarray) -> StepResult:
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions; got {len(actions)}")
        if np.any(actions < 0) or np.any(actions >= self.n_actions):
            raise ValueError(f"actions must be in [0, {self.n_actions}); got {actions.tolist()}")

        total_reward = 0.0
        done = False
        for i, env in enumerate(self._envs):
            a0, a1 = int(actions[2 * i]), int(actions[2 * i + 1])
            if not self.comm_on:
                # Expose only the movement sub-action; fix `say` to a
                # constant. The comm slot is zeroed in the observation
                # regardless, so the constant's value is irrelevant.
                a0 = a0 + _MOVE_DIM * _FIXED_SAY
                a1 = a1 + _MOVE_DIM * _FIXED_SAY
            action_dict = {self._local_agents[0]: a0, self._local_agents[1]: a1}
            next_obs, rewards, terms, truncs, _infos = env.step(action_dict)

            # simple_reference truncates the agent list on termination; keep
            # the previous obs to preserve shape (mirrors mpe_adapter.py).
            if next_obs:
                self._last_obs_dicts[i] = next_obs

            pair_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
            total_reward += pair_reward
            done = done or bool(any(terms.values()) or any(truncs.values()))

        self._episode_step += 1
        obs = self._compose_obs(self._last_obs_dicts)
        return StepResult(
            obs=obs,
            state=obs.reshape(-1).astype(np.float32),
            reward=total_reward,
            done=done,
            info={"episode_step": self._episode_step},
        )

    def adjacency(self) -> np.ndarray:
        return self._adj.copy()

    def close(self) -> None:
        if self._closed:
            return
        for env in self._envs:
            try:
                env.close()
            except Exception:
                pass
        self._closed = True

    # ------------------------------------------------------------ internals

    def _build_adjacency(self) -> np.ndarray:
        n = self.n_agents
        a = np.zeros((n, n), dtype=np.float32)
        if self.graph == "true":
            # The K native pair edges: block-diagonal 1-regular matching
            # (2k, 2k+1) -- exactly each agent's real partner.
            for kk in range(self.k):
                a[2 * kk, 2 * kk + 1] = 1.0
                a[2 * kk + 1, 2 * kk] = 1.0
        elif self.graph == "wrong":
            # Reuses CoordGrid's `matching_wrong` formula verbatim: a 1-
            # regular matching sharing NO edge with `true` (mispair across
            # sub-envs). E.g. at k=3 (N=6): true=(0,1),(2,3),(4,5);
            # wrong=(1,2),(3,4),(5,0).
            for kk in range(n // 2):
                i = 2 * kk + 1
                j = (2 * kk + 2) % n
                a[i, j] = 1.0
                a[j, i] = 1.0
        elif self.graph == "complete":
            a[:] = 1.0
            np.fill_diagonal(a, 0.0)
        else:  # pragma: no cover -- guarded in __init__.
            raise ValueError(self.graph)
        np.fill_diagonal(a, 0.0)
        return a

    def _compose_obs(self, obs_dicts: list[dict[str, Any]]) -> np.ndarray:
        n = self.n_agents
        raw = np.zeros((n, _SUBENV_OBS_DIM), dtype=np.float32)
        for i, obs_dict in enumerate(obs_dicts):
            for p, name in enumerate(self._local_agents):
                g = 2 * i + p
                o = np.asarray(obs_dict[name], dtype=np.float32).reshape(-1)
                if o.shape[0] != _SUBENV_OBS_DIM:
                    # Defensive: fail loudly rather than silently zeroing the
                    # wrong slot if a future mpe2 release changes the layout.
                    raise RuntimeError(
                        f"simple_reference obs_dim drifted: got {o.shape[0]}, "
                        f"expected {_SUBENV_OBS_DIM} (see module docstring for "
                        "the slice layout this wrapper depends on)"
                    )
                raw[g] = o

        obs = raw.copy()
        if not self.comm_on:
            obs[:, _COMM_SLICE] = 0.0

        if not self.oracle:
            return obs

        # Oracle: concatenate the partner's goal_color block -- which is
        # this agent's OWN target, held only by the partner (see module
        # docstring) -- so the routing problem is solved for free.
        out = np.zeros((n, self.obs_dim), dtype=np.float32)
        out[:, :_SUBENV_OBS_DIM] = obs
        for kk in range(self.k):
            g0, g1 = 2 * kk, 2 * kk + 1
            out[g0, _SUBENV_OBS_DIM:] = raw[g1, _GOAL_SLICE]
            out[g1, _SUBENV_OBS_DIM:] = raw[g0, _GOAL_SLICE]
        return out
