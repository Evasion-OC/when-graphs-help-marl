# Phase-0 module interfaces

This contract is binding for every module landing in Phase 0. All algorithms,
environments, the trainer, and the logger must conform exactly. Changes require
PR review — drift breaks the cross-algorithm comparability that the whole study
depends on.

## Conventions

- Tensors are PyTorch `torch.Tensor`; NumPy arrays are only at the env boundary.
- Per-agent parameter sharing is the default for IQL/VDN/QMIX/GNN-QMIX (standard
  CTDE practice). Agent identity is provided as a one-hot appended to the obs.
- Discrete actions, shared `n_actions` across agents.
- Cooperative shared scalar reward.
- Device: pass a `torch.device` into every algo constructor; default `cpu`. No
  CUDA-only code paths.
- Seeding: every randomized object (env, replay sampler, algo init, eps-greedy)
  takes an explicit `np.random.Generator` or `torch.Generator`. No global RNG.

## Env interface — `gnnmarl.envs.base.MultiAgentEnv`

```python
class MultiAgentEnv(Protocol):
    n_agents: int        # N
    obs_dim: int         # per-agent local observation dimension
    state_dim: int       # global state dimension (privileged, for mixer)
    n_actions: int       # discrete action space size (shared)

    def reset(self, *, seed: int | None = None) -> StepResult: ...
    def step(self, actions: np.ndarray) -> StepResult: ...
    def adjacency(self) -> np.ndarray:        # shape [N, N], 0/1 float32, symmetric, no self-loops
        ...
    def close(self) -> None: ...
```

`StepResult` is a dataclass with fields:

| field      | shape / type     | notes                                            |
|------------|------------------|--------------------------------------------------|
| `obs`      | `[N, obs_dim]` f32 | per-agent local observation                    |
| `state`    | `[state_dim]` f32  | privileged global state for mixer/critic       |
| `reward`   | `float`          | scalar cooperative reward (0.0 on reset)         |
| `done`     | `bool`           | episode termination flag (False on reset)        |
| `info`     | `dict`           | free-form; must include `episode_step: int`     |

On `reset` the result's `reward` is `0.0` and `done` is `False`.

The adjacency is **static within an episode** but may differ across episodes
(e.g. random Erdős–Rényi resampled each reset). Algorithms read it via
`env.adjacency()` after every reset.

## Transition record — `gnnmarl.utils.replay.Transition`

```python
@dataclass
class Transition:
    obs:        np.ndarray   # [N, obs_dim] f32
    state:      np.ndarray   # [state_dim] f32
    actions:    np.ndarray   # [N] int64
    reward:     float
    next_obs:   np.ndarray   # [N, obs_dim] f32
    next_state: np.ndarray   # [state_dim] f32
    done:       bool
    adj:        np.ndarray   # [N, N] f32
```

`ReplayBuffer.sample(batch_size, rng) -> Batch` returns torch tensors with a
leading batch dim. The `done` flag is `float32` 0/1 in the batch.

## Algorithm interface — `gnnmarl.algos.base.Algo`

```python
class Algo(Protocol):
    name: str                # "iql" | "vdn" | "qmix" | "gnn_qmix"

    def act(
        self,
        obs: np.ndarray,         # [N, obs_dim]
        adj: np.ndarray,         # [N, N]
        *,
        epsilon: float,
        rng: np.random.Generator,
    ) -> np.ndarray: ...        # [N] int64

    def observe(self, transition: Transition) -> None: ...

    def update(self) -> dict[str, float] | None:
        """One gradient step if buffer has >= batch_size. Returns loss/grad-norm dict, or None."""

    def state_dict(self) -> dict: ...
    def load_state_dict(self, d: dict) -> None: ...
```

All four algorithms share:
- Per-agent Q-net `MLP(obs_dim + n_agents → n_actions)` with hidden dim 64×2,
  parameters shared across agents; agent-id one-hot appended to obs.
- Double-DQN target with target network hard-updated every `target_update_interval` steps.
- Adam optimizer, `lr=5e-4` default.
- Replay buffer capacity 50 000, batch size 32, γ=0.99, ε linearly annealed
  1.0 → 0.05 over 50 000 env steps.

Algorithm-specific differences:
- **IQL**: per-agent TD on independent Q.
- **VDN**: `Q_tot = Σ_i Q_i(o_i, a_i)`; one shared TD loss against `r + γ Σ_i max_a Q_i^-(o'_i,a)`.
- **QMIX**: `Q_tot = Mixer(state)(Q_1,…,Q_N)`. Mixer is a 2-layer hypernetwork
  producing nonnegative weights (`|W|`) and arbitrary biases; embedding dim 32,
  mixer hidden dim 32.
- **GNN-QMIX**: per-agent obs → `L`-layer GCN over `adj` (L default = 2,
  hidden 64), then per-agent Q heads. Mixer is identical to QMIX. The GNN is
  the only structural change relative to QMIX — keep the mixer fixed so the
  ablation in Phase 4 is clean.

## Logger interface — `gnnmarl.utils.logging.CSVLogger`

Constructor: `CSVLogger(run_dir: Path, run_id: str, config: dict)`.
On init writes `run_dir/config.json` and opens `run_dir/episodes.csv` with header:

```
run_id,algo,env,seed,episode,env_step,return,episode_len,epsilon,loss,grad_norm,wallclock_s
```

API:

```python
logger.log_episode(
    episode: int, env_step: int, ret: float, ep_len: int,
    epsilon: float, loss: float | None, grad_norm: float | None,
)
logger.close()
```

`loss` and `grad_norm` are mean-over-episode-updates; `None` is written as empty
string. The logger flushes after every row so partial runs are recoverable.

## Trainer entry — `gnnmarl.training.loop.train`

```python
def train(cfg: TrainConfig) -> Path: ...
```

`TrainConfig` is a dataclass holding env name + kwargs, algo name + kwargs,
seed, total env steps, eval interval, logging dir. Returns the run directory.

A minimal CLI entrypoint at `gnnmarl/__main__.py` parses `TrainConfig` via
`tyro` so `python -m gnnmarl --algo qmix --env coord_grid ...` works.
