"""DCG — deep sparse cooperative Q-learning with max-plus action selection.

The classical coordination-graph baseline from
``docs/CLASSICAL_CG_BASELINE_DESIGN.md`` (Böhmer, Kurin & Whiteson 2020's
neural, function-approximated descendant of Kok & Vlassis max-plus). It is
the steelman answer to "does payoff-propagation over the same factored
reward already give you what the GNN gives you?" -- so it must be a genuine
coordination-graph learner, not a strawman.

Value model (design section 1)::

    Q_tot(o,a) = (1/|V|) sum_i U_i(o_i,a_i) + (1/|E|) sum_{(i,j) in E} Q_ij(o_i,o_j,a_i,a_j)

* ``U_i``: the shared per-agent :class:`~gnnmarl.algos.networks.AgentQNet`
  (obs + agent-id one-hot -> |A|), the same utility head QMIX/GNN-QMIX use.
* ``Q_ij``: a shared pairwise payoff net over ``env.adjacency()`` edges,
  symmetrized ``Q_ij(a_i,a_j) = 1/2 (f(x_i,x_j)[a_i,a_j] + f(x_j,x_i)[a_j,a_i])``.
  What ``x`` carries is the pre-registered structure variable
  (``conditioning``):

  - ``"jointobs"`` (steelman, HEADLINE arm): ``x_i = o_i`` -- the pairwise
    factor sees the full joint edge observation, the maximal information an
    edge legally carries at decentralised execution.
  - ``"canonical"`` (mechanistic control): ``x_i`` is empty -- the classic
    Kok-Vlassis coordination-game factor, a learned payoff over the action
    pair only (per agent-id, no observation of either endpoint). All
    observation-dependence lives in ``U_i``; isolates action-coupling from
    observation-routing.
  - ``"oracle_state"`` (privileged ceiling arm, never the comparison):
    ``x_i`` = the env's global state, shared by both endpoints. See the
    "Known deviation" note on :meth:`DCG.act` -- the privileged state is
    only available where the harness's ``Batch``/``Transition`` thread it
    (i.e. at TD-target time inside :meth:`DCG._td_step`); the live rollout
    policy in :meth:`DCG.act` cannot see it (the ``Algo.act(obs, adj, ...)``
    contract carries no state argument) and falls back to an all-zero state
    proxy of the same width, so it explores/acts as ``QMIX``-with-canonical-
    payoffs while still learning a state-aware ``Q_tot``.

Greedy joint action selection is **max-plus** (loopy belief propagation on
the coordination graph, ``T`` rounds, mean-subtracted messages for loopy
stability -- design section 1), implemented as the batched, differentiable
-free function :func:`max_plus_action_select` (kept module-level, not a
method, so it can be unit-tested against brute force without constructing a
full algorithm).

Training reuses the harness's Double-DQN TD skeleton conceptually (online
action-select, target value, Huber loss, hard target sync every
``target_update_interval`` updates) but :meth:`DCG._td_step` and
:meth:`DCG.act` are full overrides -- DCG does not fit
:class:`~gnnmarl.algos.base.BaseAlgo`'s per-agent-argmax IGM skeleton (see
design section 1: "must override act() and the next-action selection inside
_td_step"). ``_q_values``/``_total_q`` are still implemented (they are
``@abstractmethod`` on :class:`BaseAlgo`) but are utility-only stand-ins
never exercised by DCG's own act()/_td_step path -- see their docstrings.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from gnnmarl.utils.replay import Batch

from .base import BaseAlgo
from .networks import AgentQNet


# ---------------------------------------------------------------------------
# Free functions: max-plus + the factored Q_tot. Kept module-level (no
# ``self``) so they are directly unit-testable against brute force.
# ---------------------------------------------------------------------------


def max_plus_action_select(
    U: torch.Tensor,
    Qp: torch.Tensor,
    adj: torch.Tensor,
    rounds: int = 8,
    *,
    return_message_norms: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Batched max-plus greedy joint action selection.

    Runs ``rounds`` of synchronous (parallel) max-plus message passing over
    the (dense, batched) adjacency, then returns the greedy joint action.
    Exact on tree-structured graphs (matchings, chains) once ``rounds`` is
    at least the graph diameter; approximate (loopy belief propagation) on
    graphs with cycles (rings, complete graphs) -- design section 1.

    Message convention: ``M[b, i, j, a]`` is ``m_{i -> j}(a_j = a)``, the
    message FROM sender ``i`` TO receiver ``j`` about receiver ``j``'s
    action. Update rule (design section 1)::

        m_{i->j}(a_j) = max_{a_i} [ U_i(a_i)
                                     + sum_{k in N(i)\\j} m_{k->i}(a_i)
                                     + Q_ij(a_i,a_j) ]  -  const

    where the ``- const`` is a mean-subtraction over the message's own
    action axis each round (loopy-graph stability). Non-edges are always
    kept at exactly zero.

    Normalization: the greedy action must maximize the SAME objective
    :func:`q_tot_from_actions` scores, which is the *normalized*
    ``Q_tot = (1/|V|) sum_i U_i(a_i) + (1/|E|) sum_{edges} Q_ij(a_i,a_j)``
    (design section 1), not the raw unnormalized sum. Since ``1/|V|`` and
    ``1/|E|`` are different constants, running max-plus on raw ``U``/``Qp``
    would greedily optimize a *differently-weighted* objective and can pick
    a suboptimal joint action even on a tree. We therefore divide ``U`` by
    ``|V|=N`` and ``Qp`` by the batch's edge count ``|E|`` (unique i<j edges,
    matching :func:`q_tot_from_actions`'s convention) before running the
    message-passing recursion, so the fixed point is exact for the
    normalized ``Q_tot``.

    Args:
        U: ``[B, N, A]`` per-agent utilities ``U_i(a_i)``.
        Qp: ``[B, N, N, A, A]`` symmetrized pairwise payoffs, indexed
            ``[b, i, j, a_i, a_j]`` (only entries where ``adj[b,i,j]>0`` are
            read; non-edge entries are ignored regardless of their value).
        adj: ``[B, N, N]`` adjacency (0/1, symmetric, no self-loops).
        rounds: number of synchronous message-passing rounds (``T`` in the
            design doc; 6-8 is the pre-registered range).
        return_message_norms: if True, also return the per-round L2 norm of
            the message *update* (``||M_t - M_{t-1}||``) -- a convergence
            diagnostic used by the unit tests, not needed at train/act time.

    Returns:
        ``(actions[B, N] int64, final_vals[B, N, A])``, plus the norms list
        if ``return_message_norms``.
    """
    if U.dim() != 3:
        raise ValueError(f"max_plus_action_select expects U[B,N,A]; got {tuple(U.shape)}")
    b, n, a = U.shape
    if Qp.shape != (b, n, n, a, a):
        raise ValueError(
            f"max_plus_action_select expects Qp[B,N,N,A,A]={(b, n, n, a, a)}; "
            f"got {tuple(Qp.shape)}"
        )
    if adj.shape != (b, n, n):
        raise ValueError(
            f"max_plus_action_select expects adj[B,N,N]={(b, n, n)}; got {tuple(adj.shape)}"
        )
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")

    n_edges = (torch.triu(adj, diagonal=1) > 0).to(U.dtype).sum(dim=(1, 2)).clamp(min=1.0)
    u_n = U / n  # (1/|V|) U_i -- |V| is a fixed python int, same for every batch item.
    qp_n = Qp / n_edges.view(b, 1, 1, 1, 1)  # (1/|E|) Q_ij -- |E| varies per batch item.

    M = U.new_zeros(b, n, n, a)  # M[b, sender, receiver, a_receiver]
    norms: list[float] = []
    for _ in range(rounds):
        # Total incoming to i, evaluated at a_i: sum_k adj[k,i] * m_{k->i}(a_i).
        total_incoming = (adj.unsqueeze(-1) * M).sum(dim=1)  # [B, N, A]  (i, a_i)
        # m_{j->i}(a_i), for every (i, j): M[b, j, i, a] = M.transpose(1,2)[b,i,j,a].
        m_from_j = M.transpose(1, 2)  # [B, N, N, A]  (i, j, a_i)
        # Exclude j's own contribution to i's incoming sum (the "\j" in N(i)\j).
        incoming_minus_j = total_incoming.unsqueeze(2) - m_from_j  # [B, N, N, A] (i,j,a_i)

        val = (
            u_n.unsqueeze(2).unsqueeze(-1)  # [B, N, 1, A, 1]  (1/|V|) U_i(a_i)
            + incoming_minus_j.unsqueeze(-1)  # [B, N, N, A, 1]
            + qp_n  # [B, N, N, A, A]  (1/|E|) Q_ij(a_i, a_j)
        )  # [B, N, N, A_i, A_j]
        new_m = val.max(dim=3).values  # max over a_i -> [B, N, N, A_j]
        new_m = new_m - new_m.mean(dim=-1, keepdim=True)  # mean-subtract (stability)
        new_m = new_m * adj.unsqueeze(-1)  # keep non-edges exactly at zero

        if return_message_norms:
            norms.append(float((new_m - M).norm().item()))
        M = new_m

    total_incoming = (adj.unsqueeze(-1) * M).sum(dim=1)  # [B, N, A]
    final_vals = u_n + total_incoming  # a_i* = argmax_a [(1/|V|)U_i(a) + sum_k m_{k->i}(a)]
    actions = final_vals.argmax(dim=-1)  # [B, N] int64

    if return_message_norms:
        return actions, final_vals, norms
    return actions, final_vals


def q_tot_from_actions(
    U: torch.Tensor,
    Qp: torch.Tensor,
    actions: torch.Tensor,
    adj: torch.Tensor,
) -> torch.Tensor:
    """Factored team value at a given joint action.

    ``Q_tot(o,a) = (1/|V|) sum_i U_i(a_i) + (1/|E|) sum_{(i,j) in E, i<j} Q_ij(a_i,a_j)``

    Args:
        U: ``[B, N, A]``.
        Qp: ``[B, N, N, A, A]`` symmetrized pairwise payoffs.
        actions: ``[B, N]`` int64, the joint action to evaluate.
        adj: ``[B, N, N]``.

    Returns:
        ``[B]`` team Q-values.
    """
    b, n, a = U.shape
    chosen_u = U.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # [B, N]
    mean_u = chosen_u.mean(dim=1)  # [B]  -- (1/|V|) sum_i U_i(a_i)

    ai = actions.view(b, n, 1, 1, 1).expand(b, n, n, 1, a)
    q_at_ai = Qp.gather(dim=3, index=ai).squeeze(3)  # [B, N, N, A]  (i, j, a_j) at a_i=actions[i]
    aj = actions.view(b, 1, n, 1).expand(b, n, n, 1)
    q_ij = q_at_ai.gather(dim=3, index=aj).squeeze(3)  # [B, N, N]  Q_ij(actions[i], actions[j])

    mask = (torch.triu(adj, diagonal=1) > 0).to(q_ij.dtype)  # unique edges i<j
    n_edges = mask.sum(dim=(1, 2)).clamp(min=1.0)  # [B]
    mean_pairwise = (q_ij * mask).sum(dim=(1, 2)) / n_edges  # [B]

    return mean_u + mean_pairwise


def count_params(module: nn.Module) -> int:
    """Total learnable (deduplicated) parameter count.

    Mirrors the ``_nparams`` helper used across the test suite
    (``tests/test_gat_control.py`` etc.) so param-count comparisons between
    DCG and GNN-QMIX use the same convention.
    """
    seen: set[int] = set()
    total = 0
    for p in module.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


# ---------------------------------------------------------------------------
# Pairwise payoff network
# ---------------------------------------------------------------------------


class _PairwisePayoffNet(nn.Module):
    """Shared pairwise payoff net producing symmetrized ``Q_ij`` for every pair.

    Computed densely for ALL ordered pairs ``(i, j)`` (N is small in MARL,
    same convention as :class:`~gnnmarl.algos.networks.GCNStack`); callers
    mask to actual graph edges (see :func:`q_tot_from_actions`,
    :func:`max_plus_action_select`).

    Architecture: ``MLP([x_i; id_i; x_j; id_j]) -> hidden -> hidden ->
    n_actions*n_actions``, reshaped to ``[.., A, A]`` and symmetrized
    ``0.5*(f(x_i,id_i,x_j,id_j)[a_i,a_j] + f(x_j,id_j,x_i,id_i)[a_j,a_i])``.
    ``x`` is empty (``cond_dim=0``) for the ``"canonical"`` conditioning, so
    the whole pairwise factor collapses to a per-edge-identity (via the id
    one-hots) action-pair payoff table with zero observation dependence.
    """

    def __init__(self, cond_dim: int, n_agents: int, n_actions: int, hidden: int):
        super().__init__()
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        self.cond_dim = cond_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        in_dim = 2 * (cond_dim + n_agents)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions * n_actions),
        )
        self.register_buffer(
            "_agent_eye", torch.eye(n_agents, dtype=torch.float32), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x[B, N, cond_dim]`` (per-agent conditioning vector) -> ``[B,N,N,A,A]``."""
        if x.dim() != 3:
            raise ValueError(f"_PairwisePayoffNet expects x[B,N,cond_dim]; got {x.shape}")
        b, n, d = x.shape
        if n != self.n_agents or d != self.cond_dim:
            raise ValueError(
                f"_PairwisePayoffNet got (N,cond_dim)=({n},{d}); "
                f"expected ({self.n_agents},{self.cond_dim})"
            )

        ids = self._agent_eye.unsqueeze(0).expand(b, n, n)  # [B, N, n_agents]
        feat = torch.cat([x, ids], dim=-1)  # [B, N, cond_dim+n_agents]
        f_dim = feat.shape[-1]
        f_i = feat.unsqueeze(2).expand(b, n, n, f_dim)  # [B,N,N,F]  (i varies, const over j)
        f_j = feat.unsqueeze(1).expand(b, n, n, f_dim)  # [B,N,N,F]  (j varies, const over i)
        pair_in = torch.cat([f_i, f_j], dim=-1)  # [B,N,N,2F]

        raw = self.net(pair_in)  # [B,N,N,A*A]
        raw = raw.view(b, n, n, self.n_actions, self.n_actions)  # [B,N,N,A_i,A_j]
        # raw[b,i,j,a,c] holds f(x_i,id_i,x_j,id_j)[a,c]; symmetrize with the
        # swapped-identity, swapped-action term so Q_ij(a_i,a_j)==Q_ji(a_j,a_i).
        swapped = raw.permute(0, 2, 1, 4, 3)  # swapped[b,i,j,a,c] = raw[b,j,i,c,a]
        return 0.5 * (raw + swapped)


# ---------------------------------------------------------------------------
# DCG algorithm
# ---------------------------------------------------------------------------

_CONDITIONINGS = ("jointobs", "canonical", "oracle_state")


class DCG(BaseAlgo):
    """Deep coordination graph with max-plus action selection.

    See module docstring for the value model and the ``conditioning`` arms.
    Does not use :class:`~gnnmarl.algos.networks.QMixerHypernet` -- the
    factored ``Q_tot`` (average utility + average pairwise payoff) is
    already the mixing function; there is no separate learned mixer.
    """

    name = "dcg"

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        *,
        conditioning: str = "jointobs",
        mp_rounds: int = 8,
        payoff_hidden: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            conditioning: one of ``"jointobs" | "canonical" | "oracle_state"``
                (see module docstring). Pre-registered headline arm is
                ``"jointobs"``.
            mp_rounds: max-plus message-passing rounds (``T`` in the design
                doc; 6-8 pre-registered, default 8).
            payoff_hidden: pairwise-payoff-net hidden width. Defaults to
                ``2 * gnn_hidden`` -- wide enough that total DCG parameters
                exceed GNN-QMIX's at the matched ``gnn_hidden`` (design
                section 3: "generosity toward DCG"; see
                :func:`count_params` and ``tests/test_maxplus.py``).

        These are set BEFORE ``super().__init__()`` (i.e. before
        :class:`torch.nn.Module`'s own ``__init__`` has run) because
        :meth:`BaseAlgo.__init__` calls :meth:`_register_extra_modules`
        internally, which needs them. Plain (non-Parameter/Module) attribute
        assignment on an ``nn.Module`` is safe before ``nn.Module.__init__``
        runs -- ``nn.Module.__setattr__`` only special-cases values that ARE
        a ``Parameter``/``Module``/registered buffer name, and falls through
        to ``object.__setattr__`` otherwise.
        """
        if conditioning not in _CONDITIONINGS:
            raise ValueError(
                f"conditioning must be one of {_CONDITIONINGS}; got {conditioning!r}"
            )
        if mp_rounds < 1:
            raise ValueError(f"mp_rounds must be >= 1, got {mp_rounds}")
        self.conditioning = conditioning
        self.mp_rounds = int(mp_rounds)
        self._payoff_hidden_override = payoff_hidden

        super().__init__(
            n_agents=n_agents,
            obs_dim=obs_dim,
            state_dim=state_dim,
            n_actions=n_actions,
            **kwargs,
        )

    # --------------------------------------------------------------- setup
    def _register_extra_modules(self) -> None:
        # Drop the BaseAlgo-installed AgentQNet and rebuild it as ``U`` (same
        # architecture; renamed so the module tree reads as U_i / Q_ij).
        if hasattr(self, "agent_q"):
            del self.agent_q
        self.U = AgentQNet(obs_dim=self.obs_dim, n_agents=self.n_agents, n_actions=self.n_actions)

        cond_dim = {
            "jointobs": self.obs_dim,
            "canonical": 0,
            "oracle_state": self.state_dim,
        }[self.conditioning]
        payoff_hidden = self._payoff_hidden_override or (2 * self.gnn_hidden)
        self.payoff = _PairwisePayoffNet(
            cond_dim=cond_dim,
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            hidden=payoff_hidden,
        )

    # ---------------------------------------------------------- conditioning
    def _cond_input(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        """Build the pairwise net's per-agent conditioning vector ``x``."""
        b, n, _ = obs.shape
        if self.conditioning == "jointobs":
            return obs
        if self.conditioning == "canonical":
            return obs.new_zeros(b, n, 0)
        # "oracle_state": broadcast the (batch-level) global state to every
        # agent. ``state`` is None only from :meth:`act` (see its docstring
        # for the documented rollout-time limitation) -- fall back to zeros
        # of the same width so downstream shapes never depend on the caller.
        if state is None:
            state = obs.new_zeros(b, self.state_dim)
        return state.unsqueeze(1).expand(b, n, self.state_dim)

    def _utilities_and_payoffs(
        self,
        u_net: nn.Module,
        payoff_net: nn.Module,
        obs: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u = u_net(obs)  # [B, N, A]
        x = self._cond_input(obs, state)  # [B, N, cond_dim]
        qp = payoff_net(x)  # [B, N, N, A, A]
        return u, qp

    # ------------------------------------------------------------- act()
    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        adj: np.ndarray,
        *,
        epsilon: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Epsilon-greedy action selection with a max-plus greedy branch.

        Mirrors :meth:`BaseAlgo.act`'s contract exactly (per-agent
        independent Bernoulli epsilon mask; ``epsilon=1.0`` never touches
        the network) but the greedy branch is the JOINT max-plus action,
        not a per-agent argmax -- DCG does not fit the per-agent-argmax IGM
        skeleton (design section 1).

        Known deviation (``conditioning="oracle_state"`` only): the
        ``Algo.act(obs, adj, *, epsilon, rng)`` contract (unchanged, so the
        trainer consumes DCG like every other algo) carries no global-state
        argument, so the live rollout policy cannot see the privileged
        state here and uses an all-zero proxy of the same width instead
        (i.e. acts as canonical-payoff QMIX). The privileged state IS used
        correctly for both the current-and-next ``Q_tot`` and the Double-DQN
        max-plus bootstrap inside :meth:`_td_step`, since
        ``Batch.state``/``Batch.next_state`` are threaded through the replay
        buffer regardless of algorithm -- so ``oracle_state`` still learns a
        state-aware value function even though its own exploration policy
        does not see the state at collection time.
        """
        if obs.ndim != 2 or obs.shape != (self.n_agents, self.obs_dim):
            raise ValueError(
                f"act() expects obs of shape ({self.n_agents}, {self.obs_dim}); "
                f"got {obs.shape}"
            )
        if adj.shape != (self.n_agents, self.n_agents):
            raise ValueError(
                f"act() expects adj of shape ({self.n_agents}, {self.n_agents}); "
                f"got {adj.shape}"
            )
        eps = float(epsilon)
        random_mask = rng.random(self.n_agents) < eps
        random_actions = rng.integers(low=0, high=self.n_actions, size=self.n_agents)

        if random_mask.all():
            return random_actions.astype(np.int64)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        adj_t = torch.as_tensor(adj, dtype=torch.float32, device=self.device).unsqueeze(0)

        u, qp = self._utilities_and_payoffs(self.U, self.payoff, obs_t, None)
        greedy_t, _ = max_plus_action_select(u, qp, adj_t, rounds=self.mp_rounds)
        greedy = greedy_t.squeeze(0).cpu().numpy().astype(np.int64)

        out = np.where(random_mask, random_actions, greedy).astype(np.int64)
        return out

    # --------------------------------------------------------- TD training
    def _td_step(self, batch: Batch) -> tuple[float, float]:
        """Double-DQN TD step on the factored ``Q_tot``, max-plus bootstrap.

        ``y = r + gamma * (1-done) * Q_tot^target(o', a'*)``, where ``a'*``
        is the max-plus greedy joint action from the ONLINE net at ``o'``
        (Double-DQN action-select-online/value-from-target, mirroring
        :meth:`BaseAlgo._td_step` exactly, just with max-plus in place of
        the per-agent argmax).
        """
        u_on, qp_on = self._utilities_and_payoffs(self.U, self.payoff, batch.obs, batch.state)
        q_tot = q_tot_from_actions(u_on, qp_on, batch.actions, batch.adj)  # [B]

        with torch.no_grad():
            u_on_next, qp_on_next = self._utilities_and_payoffs(
                self.U, self.payoff, batch.next_obs, batch.next_state
            )
            next_actions, _ = max_plus_action_select(
                u_on_next, qp_on_next, batch.adj, rounds=self.mp_rounds
            )

            u_tgt_next, qp_tgt_next = self._utilities_and_payoffs(
                self._target.U, self._target.payoff, batch.next_obs, batch.next_state
            )
            next_q_tot = q_tot_from_actions(u_tgt_next, qp_tgt_next, next_actions, batch.adj)

            y = batch.reward + self.gamma * (1.0 - batch.done) * next_q_tot

        loss = F.smooth_l1_loss(q_tot, y)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online_parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.detach().item(), float(grad_norm)

    # ------------------------------------------------- BaseAlgo abstract API
    def _q_values(self, obs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Per-agent utilities ``U_i(o_i, a_i)``.

        Satisfies :class:`BaseAlgo`'s abstract contract (shape ``[B,N,A]``);
        NOT used by :meth:`act` or :meth:`_td_step` (both fully overridden),
        since ``U_i`` alone omits the pairwise coordination term.
        """
        del adj
        return self.U(obs)

    def _total_q(
        self,
        q_per_agent: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """Utility-only fallback; NOT DCG's real ``Q_tot``.

        The real factored ``Q_tot`` needs per-agent observations for the
        ``"jointobs"``/``"canonical"`` pairwise factor, which this method's
        ``BaseAlgo``-mandated signature does not carry (only ``state``,
        privileged and global). It exists solely so ``DCG`` is a concrete
        (non-abstract) subclass; the real training path is
        :meth:`_td_step` (see :func:`q_tot_from_actions`) and the real
        greedy action-value is computed inside :func:`max_plus_action_select`.
        This stub returns just ``(1/|V|) sum_i U_i(a_i)``.
        """
        del state, adj
        chosen = q_per_agent.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # [B, N]
        return chosen.mean(dim=-1)  # [B]
