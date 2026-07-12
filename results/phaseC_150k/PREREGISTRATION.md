# Pre-registration — Attention-over-complete at matched 150k budget (F1 check)

**Written and git-committed BEFORE any 150k result was observed.** This is the
decisive check demanded by adversarial review finding F1: the headline claim
"the right graph *structure* helps — not merely the communication channel or
capacity" rests on learned-attention arms (`gat_complete`, `dgn_complete`)
failing to recover the partner signal over an all-to-all graph. In the submitted
paper those arms were measured only at a **30k**-step budget, while the *negative*
endpoint used **150k**. A reviewer will say the attention floor is a
budget artifact, not a structural fact. This experiment removes that asymmetry.

## Experiment (frozen)

- Command: `phaseC_structure_sweep.py --task pair --suite attention
  --steps 150000 --seeds 12 --n-agents 6 --obs ego --log-dir results/phaseC_150k`
- **Identical** to the existing 30k attention run in every respect except
  `--steps 30000 → 150000` (and `eps_anneal_steps` = 0.8·steps, as the harness
  ties them). Same arms, seeds 0–11, grid=3, episode_steps=25, all algo_kwargs
  (lr 5e-4, batch 32, buffer 50k, target 200, gnn_hidden 64, 2 layers,
  residual+LayerNorm on all GNN arms). Written to a **separate** dir so the 30k
  data is never touched.
- Arms: mlp, gnn_true, gcn_complete, gat_complete, dgn_complete, gat_true, dgn_true.

## Primary contrasts (frozen)

`gnn_true − gat_complete` and `gnn_true − dgn_complete`, on the final-20%
evaluation window, seeds 0–11, via the existing pipeline: Welch's t with Holm
correction across the contrast family + Mann–Whitney U. Significance threshold
α = 0.05 after Holm.

## Validity gate (must pass or the run is INVALID)

`gat_true` and `dgn_true` must still train well at 150k (≈ their 30k means of
67 / 71, i.e. clearly above the ~50 floor). This proves 150k is an *adequate*
budget for these architectures given the right graph. If they collapse,
something broke; the run is discarded, not interpreted.

## Decision rule (frozen — pre-committed interpretation of all three outcomes)

Let the complete-arm means be C = {gat_complete, dgn_complete}, floor ≈ 49.9 (mlp),
gnn_true ≈ 66 at 30k.

1. **C stays at floor AND both primary contrasts significant** → claim
   **EARNED**: structure is required, not budget. Report as the headline.
2. **C climbs materially but both contrasts remain significant** → claim
   **HOLDS, RESTATED**: "the advantage persists at 5× budget; learned attention
   narrows but does not close the gap."
3. **C reaches parity (either contrast n.s. after Holm)** → claim **REFUTED**:
   rescope to sample-efficiency — "the channel suffices given enough training" —
   and rewrite the abstract/discussion accordingly. This is reported honestly,
   not buried.

## Direction-of-generosity rule (anti-p-hacking)

Any additional method variation (e.g. learning-rate or attention-head sweep) is
permitted ONLY to help the `*_complete` attention arms **win** (give the
baseline its best shot). We report best-attention-complete vs. vanilla
`gnn_true`. We do NOT vary `gnn_true` to widen the gap. Generosity flows toward
the arm that threatens the thesis, never toward the thesis.

## Learning-curve evidence (planned regardless of outcome)

Plot per-episode return for the complete arms vs. gat_true/dgn_true. If the
complete arms have plateaued (flat final third) while the true-graph arms
climbed, that is direct evidence the floor is structural, not under-training —
this is reported even in outcome (1).

## Scope rule (applies to ALL outcomes, incl. a clean win)

150k is still a finite budget. The strongest honest phrasing on a win is
"up to 150k steps / 5× the main comparison," never "in principle." The
categorical abstract sentence ("Even learned attention recovers the partner's
signal only on the right graph") is rescoped to the measured budget regardless
of result.
