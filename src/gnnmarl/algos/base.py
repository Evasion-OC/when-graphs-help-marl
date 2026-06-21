"""Base classes for Phase-0 algorithms.

This module defines two things:

1. :class:`Algo` — the structural Protocol algorithms must satisfy. The
   trainer is written against this Protocol so any concrete algo (IQL, VDN,
   QMIX, GNN-QMIX, ...) can be dropped in.
2. :class:`BaseAlgo` — an ``nn.Module`` ABC that implements the shared
   plumbing: replay buffer ownership, target network bookkeeping,
   epsilon-greedy action selection, Adam optimizer, and the Double-DQN
   update skeleton. Concrete algorithms only need to override
   :meth:`_q_values` (per-agent Q-values from a batch of obs+adj) and
   :meth:`_total_q` (how those Q-values are mixed into a team Q-value given
   actions, state and adjacency).

Keeping the shared boilerplate in one place is what makes the ablation in
Phase 4 clean: when we vary the mixer or the GNN we change exactly two
methods on the subclass and nothing else.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnnmarl.utils.replay import Batch, ReplayBuffer, Transition

from .networks import AgentQNet


# ---------------------------------------------------------------------------
# Structural Protocol (what the trainer calls)
# ---------------------------------------------------------------------------


@runtime_checkable
class Algo(Protocol):
    """Structural interface every algorithm must satisfy.

    Mirrors ``docs/INTERFACES.md``. The trainer holds an ``Algo`` instance
    and never reaches inside it.
    """

    name: str

    def act(
        self,
        obs: np.ndarray,
        adj: np.ndarray,
        *,
        epsilon: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        ...

    def observe(self, transition: Transition) -> None: ...

    def update(self) -> dict[str, float] | None:
        """One gradient step if the buffer has at least ``batch_size`` items.

        Returns a ``{"loss", "grad_norm", ...}`` dict, or ``None`` if the
        buffer was too small to sample.
        """

    def state_dict(self) -> dict: ...

    def load_state_dict(self, d: dict) -> None: ...


# ---------------------------------------------------------------------------
# Shared ABC
# ---------------------------------------------------------------------------


class BaseAlgo(nn.Module, ABC):
    """Shared scaffolding for value-decomposition algorithms.

    Subclasses must:

    * Set ``self.name``.
    * Implement :meth:`_q_values` — per-agent Q-values for a batch of
      observations and (per-batch) adjacency matrices.
    * Implement :meth:`_total_q` — fold per-agent Q-values into a single
      scalar team Q given the (selected or argmax) actions, state, and
      adjacency. For IQL this is the sum of selected Q's (no actual mixing
      — IQL is a per-agent TD); for VDN it's a sum; for QMIX it's the
      hypernet mixer; for GNN-QMIX it's the same mixer.

    Subclasses may also override :meth:`_register_extra_modules` to declare
    additional modules (e.g. the GCN or the mixer) that should participate
    in the target-network mirror.
    """

    # Override in subclass — set in __init__.
    name: str = "base"

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        *,
        lr: float = 5e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 50_000,
        batch_size: int = 32,
        target_update_interval: int = 200,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
        # GNN-specific (ignored by non-GNN algos).
        gnn_layers: int = 2,
        gnn_hidden: int = 64,
        # Anti-over-smoothing knobs for the GCN encoder (off by default so the
        # paper's depth ablation keeps its plain-GCN semantics). When on, each
        # GCN layer becomes ``h + ReLU(LayerNorm(A_hat h W))`` (GCNII-flavoured).
        gnn_residual: bool = False,
        gnn_layernorm: bool = False,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.device = device
        self.seed = seed
        self.gnn_layers = gnn_layers
        self.gnn_hidden = gnn_hidden
        self.gnn_residual = gnn_residual
        self.gnn_layernorm = gnn_layernorm

        # Deterministic init: seed torch globally before module construction so
        # the per-component generators below derive from a known state.
        torch.manual_seed(seed)

        # Per-agent Q-net (shared parameters across agents).
        self.agent_q = AgentQNet(obs_dim=obs_dim, n_agents=n_agents, n_actions=n_actions)

        # Subclasses register extra modules (e.g. mixer, GNN) BEFORE we snapshot
        # the target nets and build the optimizer.
        self._register_extra_modules()

        # Move all owned modules onto the target device.
        self.to(device)

        # Target network: a frozen deep copy of self. We attach it as a plain
        # Python attribute (NOT a registered submodule) so that:
        #   * the optimizer doesn't see target params via ``self.parameters()``,
        #   * recursive ``state_dict`` / ``parameters`` calls don't descend into
        #     it (we serialize the target explicitly in ``state_dict``), and
        #   * the deep-copied target — which is itself a ``BaseAlgo`` subclass —
        #     doesn't acquire its own ``_target`` and trigger infinite recursion.
        target = self._build_target()
        target.to(device)
        object.__setattr__(self, "_target", target)
        for p in self._target.parameters():
            p.requires_grad_(False)

        # Optimizer over the ONLINE parameters only.
        self.optimizer = torch.optim.Adam(self.online_parameters(), lr=lr)

        # Replay buffer.
        self.buffer = ReplayBuffer(capacity=buffer_capacity, device=device)

        # Counters.
        self._update_count: int = 0

    # --------------------------------------------------------------- hooks
    def _register_extra_modules(self) -> None:
        """Subclasses register additional modules here (mixer, GCN, ...)."""
        return

    def _build_target(self) -> nn.Module:
        """Deep-copy self into a frozen target network.

        We copy the entire module tree so the target shares the exact
        architecture of the online net — including any subclass extensions
        like the mixer or the GNN.
        """
        tgt = copy.deepcopy(self._target_source_module())
        tgt.eval()
        return tgt

    def _target_source_module(self) -> nn.Module:
        """Return the module to mirror into the target. Default is ``self``.

        Subclasses can override if they need to exclude some children from
        the target (we don't in Phase 0, but the hook is here for symmetry).
        """
        return self

    def online_parameters(self):
        """Parameters that should be trained.

        Excludes the target network params (which are frozen). The default
        is ``self.parameters()`` because the target is held in a separate
        attribute and not registered as a submodule.
        """
        return self.parameters()

    # ------------------------------------------------------------ abstract
    @abstractmethod
    def _q_values(self, obs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Per-agent Q-values.

        Args:
            obs: ``[B, N, obs_dim]``
            adj: ``[B, N, N]`` adjacency (may be ignored by non-GNN algos).

        Returns:
            ``[B, N, n_actions]``
        """

    @abstractmethod
    def _total_q(
        self,
        q_per_agent: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """Fold per-agent Q-values into team Q given the selected actions.

        Args:
            q_per_agent: ``[B, N, n_actions]`` (the full Q tensor; subclasses
                gather the selected action's Q themselves).
            actions: ``[B, N]`` int64.
            state: ``[B, state_dim]``.
            adj: ``[B, N, N]``.

        Returns:
            ``[B]`` team Q.
        """

    # Target-net counterparts: we keep one indirection so subclasses can
    # reuse the online-net implementation by default but still override.
    def _q_values_target(self, obs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self._target._q_values(obs, adj)  # type: ignore[attr-defined]

    def _total_q_target(
        self,
        q_per_agent: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        return self._target._total_q(q_per_agent, actions, state, adj)  # type: ignore[attr-defined]

    # ------------------------------------------------------------- public
    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        adj: np.ndarray,
        *,
        epsilon: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Epsilon-greedy action selection.

        With probability ``epsilon`` *each agent independently* samples a
        uniform random action from ``rng``; otherwise it takes the argmax
        Q-value action. The random branch never touches the network — this
        is what makes ``epsilon=1.0`` purely RNG-determined.
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
        # Per-agent Bernoulli: which agents act randomly this step.
        random_mask = rng.random(self.n_agents) < eps
        random_actions = rng.integers(low=0, high=self.n_actions, size=self.n_agents)

        if random_mask.all():
            # Skip the network entirely — purely RNG-determined actions.
            return random_actions.astype(np.int64)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        adj_t = torch.as_tensor(adj, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self._q_values(obs_t, adj_t)                # [1, N, n_actions]
        greedy = q.argmax(dim=-1).squeeze(0).cpu().numpy().astype(np.int64)  # [N]

        out = np.where(random_mask, random_actions, greedy).astype(np.int64)
        return out

    def observe(self, transition: Transition) -> None:
        self.buffer.add(transition)

    def update(self) -> dict[str, float] | None:
        """One Double-DQN gradient step. Returns a metrics dict or None."""
        if len(self.buffer) < self.batch_size:
            return None
        # Sample. We derive the sampler RNG from torch's CPU global so update()
        # is reproducible under a single seed without threading another RNG
        # through every call site. (Algorithms requiring an isolated RNG can
        # override this, but Phase 0 doesn't need it.)
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        batch = self.buffer.sample(self.batch_size, rng)

        loss, grad_norm = self._td_step(batch)
        self._update_count += 1
        if self._update_count % self.target_update_interval == 0:
            self._hard_update_target()

        return {"loss": float(loss), "grad_norm": float(grad_norm)}

    # ----------------------------------------------------------- internal
    def _td_step(self, batch: Batch) -> tuple[float, float]:
        """Run one Huber TD step with Double-DQN targets.

        Returns ``(loss_scalar, pre_clip_grad_norm)``.
        """
        # Online Q for the chosen actions.
        q = self._q_values(batch.obs, batch.adj)                       # [B, N, A]
        q_tot = self._total_q(q, batch.actions, batch.state, batch.adj)  # [B]

        # Double-DQN: action argmax from online next-net, value from target.
        with torch.no_grad():
            next_q_online = self._q_values(batch.next_obs, batch.adj)       # [B, N, A]
            next_actions = next_q_online.argmax(dim=-1)                     # [B, N]

            next_q_target = self._q_values_target(batch.next_obs, batch.adj)  # [B, N, A]
            next_q_tot = self._total_q_target(
                next_q_target, next_actions, batch.next_state, batch.adj
            )                                                                # [B]

            y = batch.reward + self.gamma * (1.0 - batch.done) * next_q_tot

        loss = F.smooth_l1_loss(q_tot, y)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # clip_grad_norm_ returns the pre-clip total norm, which is what we log.
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online_parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.detach().item(), float(grad_norm)

    def _hard_update_target(self) -> None:
        """Copy online parameters into the target network.

        We bypass our custom ``state_dict`` override (which returns the
        rich {"online", "target", ...} dict) and use the base ``nn.Module``
        state dict directly so that the target receives a plain
        ``OrderedDict`` of tensors and loads cleanly.
        """
        online_sd = nn.Module.state_dict(self._target_source_module())
        nn.Module.load_state_dict(self._target, online_sd)

    # ------------------------------------------------------------- pickle
    def state_dict(self, *args, **kwargs) -> dict:  # type: ignore[override]
        """Serialize online net, target net, optimizer, and update counter.

        When called recursively by ``torch.nn.Module.state_dict`` (which
        passes ``destination``/``prefix``/``keep_vars``) we fall through to
        the parent implementation so the standard Module format is
        preserved. The user-facing top-level call (no args/kwargs) returns
        a richer dict containing the target net and the optimizer.
        """
        is_recursive_call = (
            len(args) > 0
            or "destination" in kwargs
            or "prefix" in kwargs
            or "keep_vars" in kwargs
        )
        if is_recursive_call:
            return super().state_dict(*args, **kwargs)
        # Serialize the target via the *base* nn.Module implementation to
        # avoid recursing into our own custom dict format.
        target_sd = nn.Module.state_dict(self._target)
        return {
            "online": super().state_dict(),
            "target": target_sd,
            "optimizer": self.optimizer.state_dict(),
            "update_count": self._update_count,
        }

    def load_state_dict(self, d, *args, **kwargs):  # type: ignore[override]
        """Counterpart to :meth:`state_dict`.

        A top-level call passes our custom dict (with an ``"online"`` key);
        a recursive call from torch's machinery passes a flat OrderedDict.
        We disambiguate on the presence of the ``"online"`` key.
        """
        if isinstance(d, dict) and "online" in d and "target" in d:
            super().load_state_dict(d["online"])
            self._target.load_state_dict(d["target"])
            self.optimizer.load_state_dict(d["optimizer"])
            self._update_count = int(d["update_count"])
            return None
        return super().load_state_dict(d, *args, **kwargs)
