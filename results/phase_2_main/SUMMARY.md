# Phase 2C — main sweep summary

**Setup:** 4 algorithms × 3 task scales × 5 seeds = 60 runs at 30k env
steps each. Algorithms locked to Phase 2A's tuned configurations
(`configs/phase_2_locked.yaml`). Coordination graph fixed at `full`
(Phase 4 ablates graph structure). Shaping on, `eps_decay=15000`-`20000`
depending on scale.

**Wall:** 6.4 hours on M1 Pro / MPS, started 2026-05-09 18:00 UTC,
finished 2026-05-10 00:23 UTC.

## Headline results (peak / final-window eval success rate)

| Scale | N, grid | IQL | VDN | QMIX | **GNN-QMIX** |
|---|---|---|---|---|---|
| small | 2 agents on 4×4 | 0.68 / 0.32 | 0.47 / 0.25 | 0.59 / 0.38 | **0.63 / 0.43** |
| medium | 3 agents on 5×5 | **0.20 / 0.12** | 0.01 / 0.00 | 0.04 / 0.01 | 0.04 / 0.01 |
| large | 4 agents on 6×6 | 0.02 / 0.01 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

Bold = the highest value in each row.

## H1 statistical tests

Welch t and Mann-Whitney U with Holm correction across the 6
pairwise contrasts per (scale, metric) cell, n=5 seeds.

| Scale | Pair | Δ peak | Welch p (Holm) | Significant at α=0.05 |
|---|---|---|---|---|
| small | iql vs vdn | +0.21 | 0.026 | **yes** |
| small | iql vs qmix | +0.09 | 0.335 | no |
| small | iql vs gnn_qmix | +0.05 | 0.489 | no |
| small | vdn vs qmix | -0.12 | 0.049 | **yes** |
| small | vdn vs gnn_qmix | -0.16 | 0.023 | **yes** |
| small | **qmix vs gnn_qmix** | **-0.04** | **0.489** | **no** |
| medium | iql vs vdn | +0.19 | 0.007 | **yes** |
| medium | iql vs qmix | +0.16 | 0.013 | **yes** |
| medium | iql vs gnn_qmix | +0.16 | 0.013 | **yes** |
| medium | vdn vs qmix | -0.03 | 0.200 | no |
| medium | vdn vs gnn_qmix | -0.03 | 0.200 | no |
| medium | qmix vs gnn_qmix | 0.00 | 1.000 | no |
| large | all pairs | ≈ 0 | 1.000 | no |

Full table in `h1_tests.md`.

## What we can claim

1. **Scale dominates over algorithm choice.** At small (8-dim state),
   all four algorithms learn comparably; at medium (12-dim state),
   only IQL learns above-zero; at large (16-dim state), none of the
   four algorithms learn at all in this 30k step budget.

2. **GNN-QMIX vs QMIX: no significant difference at any scale.** The
   GNN aggregation does not give a statistically distinguishable
   advantage over the vanilla QMIX hypernetwork mixer once both are
   tuned with Phase 2A's grid. This is *contrary* to the literature
   narrative that graph augmentation systematically helps.

3. **IQL — the simplest baseline — is the most robust to scale.**
   At medium, IQL is the only algorithm with non-zero success and is
   significantly better than all three coordination-mechanism
   algorithms (VDN, QMIX, GNN-QMIX). This contradicts the textbook
   "value-decomposition fixes IQL's instability" framing for this
   regime.

4. **VDN underperforms relative to QMIX/GNN-QMIX at small.** The sum
   mixer is significantly worse than the hypernetwork mixers when the
   task is small enough that everyone learns. So the "VDN is too
   simple" claim does have a regime where it bites — but VDN is *not*
   uniformly the weakest algorithm (see Phase 3 simple_spread smoke,
   where VDN is the *only* robust algorithm).

## What we cannot claim

- That GNN-QMIX is worse than QMIX. The p-values do not separate them.
- That value decomposition methods are categorically broken for
  cooperative MARL. They work at small scale.
- That this generalizes beyond coord_grid. Phase 3's simple_spread
  smoke already showed the Phase 2A configs don't transfer to a
  benchmark with larger state dim and reward magnitudes. Phase 3's
  full sweep (which we run despite knowing the configs fail) will
  document the negative-transfer pattern with n=5 statistical
  evidence.

## What this tells us about the paper's thesis

The originally-proposed thesis was "GNN-QMIX outperforms QMIX in
tasks with graph-structured coordination demand." Phase 2C falsifies
that claim at α=0.05 on the env we tested. The data instead supports
a more nuanced and (I think) more interesting statement:

> The benefit of graph-structured coordination in cooperative MARL is
> **scale-dependent**. At task scales where all algorithms learn,
> hypernetwork mixers (with or without GNN augmentation) are
> statistically indistinguishable. At task scales where most
> algorithms fail to learn, the simplest baseline (IQL) is the most
> robust. The paper should report this scale-dependent picture
> rather than the "GNN always wins" framing.

The Phase 3 simple_spread results (when they arrive) will add a
second dimension to this picture: the *transferability* of any given
hyperparameter configuration across env scales is poor for the QMIX
family but fine for VDN and IQL.

## Figures

- `learning_curves_small.png` — small task; all four algos clustered around 0.5-0.7
- `learning_curves_medium.png` — medium task; IQL pulls ahead, others flatline near zero
- `learning_curves_large.png` — large task; everything pinned at zero
- `forest_plot.png` — peak success per (algo × scale) with 95% CIs

## Reproducibility

All 60 raw run CSVs are in `iql/`, `vdn/`, `qmix/`, `gnn_qmix/`.
Per-seed and aggregate tables in `per_seed.csv` and `summary.csv`.
H1 statistical test details in `h1_tests.csv` / `h1_tests.md`.
Locked hyperparameters in `configs/phase_2_locked.yaml` (from Phase 2A).
Total wall: 6.4 hours on Apple M1 Pro / MPS, single GPU.
