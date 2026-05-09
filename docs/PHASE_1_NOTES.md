# Phase 1 — pilot notes

## What this phase is for

A sanity gate before any real experiment. Three questions to answer:

1. **Does every algorithm learn at all on a coordination task?**
   If even VDN fails, the harness has a bug.
2. **Do the four algorithms produce *different* learning curves?**
   If they all converge identically, the env doesn't discriminate the
   algorithmic differences and Phase 2 needs a harder env.
3. **Are the curves stable across seeds?**
   If a single seed dominates the noise, Phase 2 needs more seeds.

This phase does **not** try to test any hypothesis from the paper.
H1, H2, H3 wait for Phase 2 (where we have enough seeds and step budget
for confidence intervals that don't overlap by accident).

## Pilot env config

Selected after a tighter sanity sweep showed:
- 4 agents on 6×6 with shaping is too hard inside a 15k step budget.
  All algorithms converge to "stay put", returns ≈ pure step penalty.
- 3 agents on 5×5 with shaping shows movement-toward-target but no
  successful joint coordination inside 10k steps.
- **2 agents on 4×4 with shaping** shows VDN reaching ~0.7 success
  in ~3k steps and is reproducible. This is the right pilot regime.

The harder regimes will be revisited in Phase 2 with 5 seeds and
30–50k step budgets.

## Found and fixed during this phase

- `CSVLogger` froze its schema to the first row's keys, silently
  dropping every column unique to evaluation rows (e.g.
  `eval_return_mean`). Refactored to take an explicit schema list at
  construction time and raise on unknown keys. Eval rows now persist.
- Training loop logged a mix of train and eval rows with no
  distinguishing field. Added a `kind` column (`'train'` / `'eval'`)
  so post-hoc filtering is unambiguous.
- Step-penalty-dominated reward made coordination tasks unlearnable
  inside small step budgets. Added optional **distance-based
  potential shaping** (Ng et al. 1999, policy-invariant) to provide
  exploration signal in sparse-reward regimes. Off by default — sweep
  configs opt in.

## Why distance shaping is allowed (and what it isn't)

Potential shaping with potential function $\Phi(s) = -\bar d(s)$ adds
$F = \Phi(s') - \Phi(s)$ to the reward at each transition. The
[Ng-Harada-Russell 1999 result](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
guarantees that the optimal policy under this shaped reward is
*identical* to the optimal policy under the unshaped reward, so no
algorithm gets a comparative advantage from it. Shaping only helps
exploration. Every algorithm in the comparison uses the same shaping,
so any difference between them is still attributable to the mixer
mechanism, not to reward design.

## Next phases

- **Phase 2** (`phase-2-e1-full`): 3-agent / 5×5 and 4-agent / 6×6
  with 30–50k step budgets, 5 seeds, with and without shaping;
  varying coordination graphs (ring vs lattice vs random) to test H1.
- **Phase 3** (`phase-3-mpe`): replicate findings on PettingZoo's
  `simple_spread` and `simple_tag` for external validity.
- **Phase 4** (`phase-4-ablations`): vary GNN depth `L ∈ {1,2,3,4,6}`
  and graph diameter to test the inverted-U prediction in H3.
