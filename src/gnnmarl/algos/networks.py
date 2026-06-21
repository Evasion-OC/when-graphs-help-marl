"""Neural network building blocks shared by the Phase-0 algorithms.

Three modules:

* :class:`AgentQNet` — shared per-agent MLP Q-head. Agent identity is appended
  to the local observation as a one-hot so the same MLP can serve every agent
  (standard CTDE parameter sharing).
* :class:`GCNStack` — pure-PyTorch graph convolutional network. We avoid the
  ``torch_geometric`` dependency in Phase 0 because Windows installs are flaky
  (the conda/pip wheels for the C++ extensions are not always available); the
  GCN here is the textbook Kipf & Welling formulation,
  :math:`H' = \\sigma(\\hat{D}^{-1/2}\\hat{A}\\hat{D}^{-1/2} H W)`, computed
  per batch element so adjacency may vary across the batch.
* :class:`QMixerHypernet` — standard QMIX monotonic mixer with
  hypernetwork-produced absolute-value weights.

All modules conform to the shapes pinned in ``docs/INTERFACES.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Per-agent Q network (shared parameters across agents)
# ---------------------------------------------------------------------------


class AgentQNet(nn.Module):
    """Shared per-agent MLP Q-net.

    Architecture (per the interface contract):
        ``Linear(obs_dim + n_agents, 64) -> ReLU -> Linear(64, 64) -> ReLU
        -> Linear(64, n_actions)``

    The same parameters are applied to every agent. Agent identity is
    injected as a one-hot appended to the observation so the network can
    specialize per agent if needed.
    """

    def __init__(self, obs_dim: int, n_agents: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        in_dim = obs_dim + n_agents
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

        # Cached identity one-hot to avoid reallocating each forward pass; the
        # buffer moves with the module via ``.to(device)``.
        self.register_buffer(
            "_agent_eye",
            torch.eye(n_agents, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map ``obs[B, N, obs_dim]`` -> ``q[B, N, n_actions]``."""
        if obs.dim() != 3:
            raise ValueError(f"AgentQNet expects obs with 3 dims (B, N, D); got {obs.shape}")
        b, n, d = obs.shape
        if n != self.n_agents:
            raise ValueError(f"AgentQNet got N={n}, expected {self.n_agents}")
        if d != self.obs_dim:
            raise ValueError(f"AgentQNet got obs_dim={d}, expected {self.obs_dim}")

        ids = self._agent_eye.unsqueeze(0).expand(b, n, n)  # [B, N, n_agents]
        x = torch.cat([obs, ids], dim=-1)                   # [B, N, obs_dim+n_agents]
        return self.net(x)                                  # [B, N, n_actions]


# ---------------------------------------------------------------------------
# Pure-torch GCN stack
# ---------------------------------------------------------------------------


class GCNStack(nn.Module):
    """Stacked Kipf-Welling GCN layers with optional residual connections.

    Each layer computes :math:`H' = \\mathrm{ReLU}(\\tilde{A} H W)` where
    :math:`\\tilde{A} = \\hat{D}^{-1/2} (A + I) \\hat{D}^{-1/2}` with
    :math:`\\hat{D}_{ii} = \\sum_j (A+I)_{ij}`.

    The normalization is computed per batch element because adjacency may
    differ across the batch (e.g. when each episode samples a fresh
    Erdős–Rényi graph). The cost is negligible — N is small in MARL.

    From layer 2 onward we add a residual connection ``H' = H' + H`` since the
    hidden dim is constant after the first projection; this stabilizes deeper
    stacks (helpful in the Phase-4 depth ablation).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_layers: int,
        residual: bool = False,
        layernorm: bool = False,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"GCNStack requires n_layers >= 1, got {n_layers}")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.residual = bool(residual)
        self.layernorm = bool(layernorm)

        layers: list[nn.Linear] = []
        for layer_idx in range(n_layers):
            din = in_dim if layer_idx == 0 else hidden_dim
            layers.append(nn.Linear(din, hidden_dim, bias=True))
        self.layers = nn.ModuleList(layers)
        self.norms = (
            nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
            if self.layernorm
            else None
        )

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        """Symmetric normalization of ``A + I``.

        Args:
            adj: ``[B, N, N]`` float adjacency (0/1, symmetric, no self-loops).

        Returns:
            ``[B, N, N]`` float normalized adjacency.
        """
        if adj.dim() != 3:
            raise ValueError(f"GCNStack expects adj with 3 dims (B, N, N); got {adj.shape}")
        b, n, m = adj.shape
        if n != m:
            raise ValueError(f"GCNStack expects square adjacency; got {adj.shape}")

        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(b, n, n)
        a_hat = adj + eye
        deg = a_hat.sum(dim=-1)                              # [B, N]
        # Avoid div-by-zero; any isolated node still has self-loop so deg>=1,
        # but clamp just in case the user passes an unusual graph.
        deg_inv_sqrt = torch.clamp(deg, min=1e-6).pow(-0.5)  # [B, N]
        # D^{-1/2} A D^{-1/2}
        norm = a_hat * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
        return norm

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Run the GCN stack.

        Args:
            h: ``[B, N, in_dim]`` per-agent features.
            adj: ``[B, N, N]`` adjacency (0/1, symmetric, no self-loops).

        Returns:
            ``[B, N, hidden_dim]`` per-agent features after message passing.

        Note: no residual connection. The depth-vs-diameter ablation
        (paper Sec. 6.4 / H3) requires the only difference between an L=1
        and an L=2 model to be one extra message-passing step; a residual
        that kicks in at L>=2 would confound the depth effect with a
        residual-only effect. If you need a residual variant, wrap the
        stack — do not add it inside.
        """
        norm = self._normalize_adj(adj)
        for idx, lin in enumerate(self.layers):
            # Aggregate neighbours then project: A_hat @ H @ W
            agg = torch.bmm(norm, h)        # [B, N, in/hidden]
            out = lin(agg)
            if self.norms is not None:
                out = self.norms[idx](out)  # LayerNorm before the nonlinearity
            out = F.relu(out)               # [B, N, hidden]
            # Residual skip when shapes line up (anti-over-smoothing). With the
            # default flags this branch never runs, so plain-GCN behaviour and
            # the depth ablation are unchanged.
            h = h + out if (self.residual and out.shape == h.shape) else out
        return h


class MLPStack(nn.Module):
    """Parameter-matched no-graph control for :class:`GCNStack`.

    Byte-for-byte identical to :class:`GCNStack` in parameters and output
    shape — the same ``n_layers`` ``Linear`` modules with the same in/out
    dims — but with the neighbour-aggregation step removed. Each layer is
    just ``H' = ReLU(H W)``; there is no ``A_hat @ H`` mixing, so per-agent
    features never exchange information across the coordination graph.

    The ``forward`` signature accepts ``adj`` for drop-in interface parity
    with :class:`GCNStack`, but ignores it. This makes MLP-QMIX the clean
    control that isolates *graph structure* from *extra capacity*: at a
    matched ``(hidden_dim, n_layers)`` it has the same parameter count as
    the GCN stack, so any GNN-QMIX vs MLP-QMIX difference is the graph,
    not the parameters.
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"MLPStack requires n_layers >= 1, got {n_layers}")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        layers: list[nn.Linear] = []
        for layer_idx in range(n_layers):
            din = in_dim if layer_idx == 0 else hidden_dim
            layers.append(nn.Linear(din, hidden_dim, bias=True))
        self.layers = nn.ModuleList(layers)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        """Per-node MLP; ``adj`` accepted for interface parity, ignored."""
        for lin in self.layers:
            h = F.relu(lin(h))              # [B, N, hidden] — no aggregation
        return h


class GATStack(nn.Module):
    r"""Graph-attention stack: :class:`GCNStack` with *learned* aggregation.

    Mirrors :class:`GCNStack` one-for-one --- the same ``n_layers``
    ``Linear(din, hidden)`` modules, the same ReLU activation, the same
    self-loop convention (attend over :math:`A+I`) --- but replaces the
    *fixed* symmetric-normalized aggregation weights with single-head graph
    attention (Veličković et al., 2018): for projected features
    :math:`z_i = W h_i`, the edge weight is
    :math:`\alpha_{ij} = \mathrm{softmax}_j\big(\mathrm{LeakyReLU}
    (a_{\mathrm{src}}^\top z_i + a_{\mathrm{dst}}^\top z_j)\big)` over the
    neighbourhood (including self), and :math:`h_i' = \mathrm{ReLU}
    (\sum_j \alpha_{ij} z_j)`.

    This is the attention mechanism used by the graph-MARL methods the
    paper actually critiques (DGN, G2ANet), so GAT-QMIX tests whether a
    *learned* relational weighting rescues the graph where the fixed GCN
    aggregation does not. Per layer it adds two attention vectors
    ``a_src, a_dst`` of size ``hidden`` (``2*hidden`` params/layer; ~1%
    over GCN at ``hidden=64``). GAT therefore has marginally *more*
    capacity than the parameter-matched MLP control, so any GAT-vs-MLP
    deficit is a conservative estimate of the graph penalty.

    As with :class:`GCNStack`, there is no residual connection, so the
    only difference between an ``L=1`` and ``L=2`` model is one extra
    attention-aggregation step (keeping the depth ablation clean).
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int,
                 leaky_slope: float = 0.2):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"GATStack requires n_layers >= 1, got {n_layers}")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.leaky_slope = leaky_slope

        layers: list[nn.Linear] = []
        for layer_idx in range(n_layers):
            din = in_dim if layer_idx == 0 else hidden_dim
            layers.append(nn.Linear(din, hidden_dim, bias=True))
        self.layers = nn.ModuleList(layers)

        # Single-head additive attention: one source + one target vector per
        # layer, scoring the projected features.
        self.att_src = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_layers)]
        )
        self.att_dst = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_layers)]
        )
        # Xavier-uniform bound for an effectively [hidden, 1] attention vector.
        bound = (6.0 / (hidden_dim + 1)) ** 0.5
        for p in list(self.att_src) + list(self.att_dst):
            nn.init.uniform_(p, -bound, bound)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Run the GAT stack.

        Args:
            h: ``[B, N, in_dim]`` per-agent features.
            adj: ``[B, N, N]`` adjacency (0/1, symmetric, no self-loops).

        Returns:
            ``[B, N, hidden_dim]`` per-agent features after attention.
        """
        if adj.dim() != 3:
            raise ValueError(f"GATStack expects adj[B,N,N]; got {adj.shape}")
        b, n, m = adj.shape
        if n != m:
            raise ValueError(f"GATStack expects square adjacency; got {adj.shape}")

        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        edge = (adj + eye) > 0                                 # [B, N, N] incl. self
        neg_inf = torch.finfo(h.dtype).min
        for layer_idx, lin in enumerate(self.layers):
            z = lin(h)                                          # [B, N, hidden]
            s = (z * self.att_src[layer_idx]).sum(-1, keepdim=True)   # [B, N, 1]
            t = (z * self.att_dst[layer_idx]).sum(-1, keepdim=True)   # [B, N, 1]
            # scores[b,i,j] = LeakyReLU(a_src . z_i + a_dst . z_j)
            scores = F.leaky_relu(s + t.transpose(1, 2), self.leaky_slope)
            scores = scores.masked_fill(~edge, neg_inf)
            alpha = torch.softmax(scores, dim=-1)               # [B, N, N]
            h = F.relu(torch.bmm(alpha, z))                     # [B, N, hidden]
        return h


class DGNStack(nn.Module):
    r"""Multi-head dot-product graph attention --- the DGN/G2ANet mechanism.

    A faithful realisation, within the shared harness, of the multi-head
    scaled-dot-product graph attention that serves as the relational
    "convolution" in the attention-based graph-MARL methods this paper
    discusses (DGN, Jiang et al. 2020; G2ANet, Liu et al. 2020) --- as
    opposed to the fixed Kipf--Welling aggregation (:class:`GCNStack`) or
    single-head additive attention (:class:`GATStack`). Each layer runs
    ``n_heads`` attention heads over the neighbourhood (including self),
    concatenates them, and projects back to ``hidden_dim``:

        Q, K, V = W_q h, W_k h, W_v h   (per head, dim hidden/n_heads)
        alpha_{ij} = softmax_j( Q_i . K_j / sqrt(d_head) )   over A+I
        h_i' = ReLU( W_o [ concat_heads ( sum_j alpha_{ij} V_j ) ] )

    This is the most expressive graph mechanism we test, and --- with its
    Q/K/V/output projections --- the one with by far the *most* parameters,
    so a DGN-QMIX-below-MLP-QMIX result is the most conservative possible
    form of the graph-penalty claim (more capacity *and* the strongest
    relational bias, yet still a net liability). No residual connection,
    matching the other stacks, so the depth ablation stays clean.
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int,
                 n_heads: int = 4):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"DGNStack requires n_layers >= 1, got {n_layers}")
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.q = nn.ModuleList()
        self.k = nn.ModuleList()
        self.v = nn.ModuleList()
        self.o = nn.ModuleList()
        for layer_idx in range(n_layers):
            din = in_dim if layer_idx == 0 else hidden_dim
            self.q.append(nn.Linear(din, hidden_dim, bias=True))
            self.k.append(nn.Linear(din, hidden_dim, bias=True))
            self.v.append(nn.Linear(din, hidden_dim, bias=True))
            self.o.append(nn.Linear(hidden_dim, hidden_dim, bias=True))

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Map ``h[B,N,in_dim]`` -> ``[B,N,hidden_dim]`` via multi-head
        graph attention over ``adj`` (self-loops added)."""
        if adj.dim() != 3:
            raise ValueError(f"DGNStack expects adj[B,N,N]; got {adj.shape}")
        b, n, m = adj.shape
        if n != m:
            raise ValueError(f"DGNStack expects square adjacency; got {adj.shape}")

        heads, d = self.n_heads, self.head_dim
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        mask = ((adj + eye) > 0).unsqueeze(1)                 # [B, 1, N, N] incl. self
        neg_inf = torch.finfo(h.dtype).min
        scale = d ** -0.5
        for layer_idx in range(self.n_layers):
            q = self.q[layer_idx](h).view(b, n, heads, d).transpose(1, 2)  # [B,H,N,d]
            k = self.k[layer_idx](h).view(b, n, heads, d).transpose(1, 2)
            v = self.v[layer_idx](h).view(b, n, heads, d).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale          # [B,H,N,N]
            scores = scores.masked_fill(~mask, neg_inf)
            alpha = torch.softmax(scores, dim=-1)                          # [B,H,N,N]
            ctx = torch.matmul(alpha, v)                                   # [B,H,N,d]
            ctx = ctx.transpose(1, 2).reshape(b, n, heads * d)             # [B,N,hidden]
            h = F.relu(self.o[layer_idx](ctx))                             # [B,N,hidden]
        return h


# ---------------------------------------------------------------------------
# QMIX monotonic mixer (hypernetwork)
# ---------------------------------------------------------------------------


class QMixerHypernet(nn.Module):
    """Standard QMIX mixer.

    Mixes per-agent Q-values into a single team Q via a two-layer feedforward
    network whose weights are produced by a hypernetwork conditioned on the
    privileged global state. Weights are forced nonneg by taking absolute
    value, guaranteeing :math:`\\partial Q_{tot}/\\partial Q_i \\ge 0`
    (the monotonicity that makes joint argmax tractable).

    Sizing follows the original Rashid et al. (2018) defaults:
        * ``embed_dim`` = 32 (mixer hidden width)
        * ``hyper_hidden`` = 32 (hypernet hidden width)
    The hypernetwork for the first layer's weights is a 2-layer MLP because
    it has the largest fan-out (``n_agents * embed_dim``); the second layer
    uses linear hypernets, matching the reference implementation.
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        hyper_hidden: int = 32,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Weights of the first mixing layer: state -> [n_agents, embed_dim].
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * embed_dim),
        )
        # Weights of the second mixing layer: state -> [embed_dim, 1].
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, embed_dim),
        )

        # Biases: unconstrained.
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        # Final bias goes through a small MLP (the canonical implementation
        # uses a 2-layer MLP terminating in a single scalar).
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, q_per_agent: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Mix per-agent Q into team Q.

        Args:
            q_per_agent: ``[B, N]``
            state: ``[B, state_dim]``

        Returns:
            ``[B]`` team Q-values.
        """
        if q_per_agent.dim() != 2:
            raise ValueError(
                f"QMixerHypernet expects q_per_agent[B, N]; got {q_per_agent.shape}"
            )
        if state.dim() != 2:
            raise ValueError(f"QMixerHypernet expects state[B, state_dim]; got {state.shape}")
        b, n = q_per_agent.shape
        if n != self.n_agents:
            raise ValueError(f"QMixerHypernet got N={n}, expected {self.n_agents}")
        if state.shape[0] != b:
            raise ValueError(
                f"QMixerHypernet batch mismatch: q has B={b}, state has B={state.shape[0]}"
            )

        # First layer: [B, 1, N] @ [B, N, E] -> [B, 1, E]
        w1 = torch.abs(self.hyper_w1(state)).view(b, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(b, 1, self.embed_dim)
        q = q_per_agent.view(b, 1, self.n_agents)
        h = F.elu(torch.bmm(q, w1) + b1)                  # [B, 1, E]

        # Second layer: [B, 1, E] @ [B, E, 1] -> [B, 1, 1]
        w2 = torch.abs(self.hyper_w2(state)).view(b, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(b, 1, 1)
        y = torch.bmm(h, w2) + b2                          # [B, 1, 1]
        return y.view(b)
