# Phase 3 CPU smoke — unexpected finding

## TL;DR

The QMIX-family monotonic mixer with state-conditioned hypernetwork
**diverges** on `simple_spread` under every configuration we tested,
including Phase 2A's locked configs that worked on coord_grid. VDN
(parameter-free sum mixer) and IQL (no mixer) work without retuning.

This is a substantial finding that changes the Phase 3 design.

## What we ran

CPU-only smoke on Phase 3's target benchmark:
`mpe2.simple_spread_v3`, 3 agents, 10k env steps, 1 seed, default
trainer schedule (`eps_decay=5000`, `batch=64`, `lr=5e-4`).

## Results table

| Config | Final return | Late loss | Late `q_tot` | Status |
|---|---|---|---|---|
| VDN (parameter-free sum mixer) | -21 | 0.25 | -3.5 | ✅ healthy |
| IQL (no mixer at all) | -21 | 1.1 | -8.6 | ✅ healthy |
| QMIX locked Phase 2A: emb=16 mixer_lr=1e-4 ortho(0.1) | -38 | 25.7 | **+28** | ❌ wrong sign |
| GNN-QMIX locked Phase 2A: emb=8 mixer_lr=1e-4 ortho(0.1) | -62 | 124 | **+80** | ❌ diverges |
| GNN-QMIX with mixer_lr=5e-4 ortho(0.1) | -52 | 880 | **+224** | ❌ catastrophic |
| GNN-QMIX Phase 1 defaults: emb=32 mixer_lr=5e-4 default-init | -28 | 2078 | **+413** | ❌ exploding |

`q_tot` should be **negative** on `simple_spread` (rewards are negative).
The QMIX-family produces large positive `q_tot` instead — the
state-conditioned hypernetwork bias path is moving in the wrong
direction.

## Diagnosis

`simple_spread` has:
- 54-dim global state (vs coord_grid's 8-dim)
- Per-step reward in roughly $[-5, -1]$
- Episodic returns in roughly $[-125, -25]$

Q-values need to span $\approx -100$ to $0$. The QMIX hypernetwork
mixer's behaviour depends on initialization in a way the literature
does not document:

- **Orthogonal init with gain 0.1** (Phase 2A's locked choice for
  coord_grid): produces tiny initial mixer output. The mixer must
  scale up to $\approx -100$, but with `mixer_lr=1e-4` it can't
  catch up to the Q-net's updates. Mixer drifts in the wrong
  direction; loss explodes.

- **Orthogonal init with gain 0.1, `mixer_lr=5e-4`**: faster mixer
  updates but also faster divergence. Catastrophic.

- **Default init, large `embed_dim`** (Phase 1's broken-on-coord_grid
  config): large initial bias-path output. Loss explodes worst of all.

In short, *no* set of mixer hyperparameters we tested makes the QMIX
family stable on `simple_spread`. The mixer's state-conditioned bias
is the failure mode in both directions.

VDN's sum mixer is parameter-free, so it has no such failure mode —
the per-agent Q-values are simply summed, and as long as the Q-net
can scale, $Q_\text{tot}$ scales with it.

## Why this matters for the paper

The original Phase 3 plan was: take Phase 2A's locked configs, run on
`simple_spread`, report whether the conclusions transfer. The smoke
tells us the conclusions do *not* transfer cleanly — and this is the
paper-relevant finding, not a setup error.

The reframed Phase 3 contribution becomes:

> Phase 2A's tuning grid identified configurations that fix the
> QMIX-family failure on small-state-dim coordination tasks. These
> configurations do not transfer to higher-dim reward magnitudes
> (`simple_spread`'s 54-dim state and order-100 Q-values). VDN, which
> the field often dismisses as too simple, is the most robust
> algorithm in the high-dim regime. The failure mode is the
> state-conditioned hypernetwork bias path of QMIX, not the GNN
> component of GNN-QMIX (because vanilla QMIX fails identically).

This is a stronger, more honest paper claim than "GNN-QMIX is
better": the headline is *the QMIX-family mixer has hyperparameter
transferability issues that VDN doesn't share*.

## Phase 3 design options going forward

1. **Run sweep with locked configs anyway, report the failure.**
   Honest, low-effort, and the paper section reads as a clean
   negative-transfer finding. No additional tuning.

2. **Add Phase 3a tuning sub-sweep on `simple_spread`.** Try wider
   axes (gain ∈ {0.5, 1.0, 1.4} for orthogonal, larger embed_dim,
   per-mixer normalization). Risk: the QMIX-family may still fail.
   If so, we report two failed tuning rounds — also defensible but
   noisier.

3. **Run Phase 3 with VDN/IQL only.** Skip the QMIX-family on
   `simple_spread`. Simpler, but loses the "what the paper sets out
   to test" symmetry.

The cleanest paper story is **(1)**: predict, run, report. The locked
configs were chosen by a documented procedure on coord_grid; we
observe what happens when we apply them unmodified to a new env.
Negative transfer is itself a finding.

## Decision needed

Pick one of the three options above before launching the overnight
Phase 3 sweep.
