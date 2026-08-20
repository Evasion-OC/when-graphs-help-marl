# Pre-registration — graded graph mis-specification (Phase E)

**Written and git-committed BEFORE any smoke or confirmatory result was
observed, and before submission of the manuscript.** No run in this cell exists
at the time of this commit; `results/phaseE_misspec/` contains this file and
nothing else.

## 0. Why this exists now

The manuscript's positive endpoint contrasts an exactly-correct coordination
graph against a fully-wrong, density-matched one. Limitations item 5 names the
gap this leaves: a practitioner does not face the endpoints, they face a graph
that is *partly* right, and we cannot currently say whether the benefit decays
gracefully or falls off a cliff.

This pre-registration freezes that experiment without running it. If a referee
asks for graded mis-specification — the most likely single request, given that
the limitation is stated in the paper — the protocol below was already
committed, with timestamp-verifiable git ancestry, before the referee saw the
manuscript. Executing it in revision then carries the same evidential standing
as Phases C and D, rather than the weaker standing of a post-hoc revision
experiment.

If no referee asks, this cell is not run and this file stands as an unexecuted
pre-registration. That is a legitimate terminal state and is **not** to be
retro-fitted to any other analysis.

## 1. The combinatorial constraint (established before design, not after)

The task graph on `N` agents is the canonical perfect matching
`M* = {(0,1), (2,3), ...}` with `m = N/2` edges (`coord_grid.py`,
`graph_kind="matching"`). A mis-specified graph must stay **1-regular** so the
density-matched property that carries the paper's argument is preserved. It is
therefore also a perfect matching, and the mis-specification level is its
overlap `k = |M ∩ M*|`.

**Not every `k` is reachable.** A perfect matching cannot agree with `M*` on
exactly `m-1` edges: fixing `m-1` pairs leaves exactly two agents, and those two
are necessarily each other's true partner, forcing overlap `m`. Enumerated
exhaustively over all perfect matchings:

| `N` | `m` | achievable overlaps `k` | achievable `p = k/m` |
|---|---|---|---|
| 6 | 3 | 0, 1, 3 | 0, 0.33, 1 |
| 8 | 4 | 0, 1, 2, 4 | 0, 0.25, 0.5, 1 |
| 10 | 5 | 0, 1, 2, 3, 5 | 0, 0.2, 0.4, 0.6, 1 |

The grid `{0, 0.25, 0.5, 0.75, 1}` written in the manuscript's limitations item
is therefore **not realisable** under the density constraint. It is superseded
here, before any data exists. The correction is recorded rather than silently
applied.

**Chosen cell: `N = 10`**, giving the densest achievable grid
`p ∈ {0, 0.2, 0.4, 0.6, 1}` — three interior points rather than two. `N = 10` is
the top of the range over which the positive effect is already established as
significant (`N = 4`–`10`), so the cell sits inside validated territory.

**Documented fallback if compute is constrained:** `N = 8`,
`p ∈ {0, 0.25, 0.5, 1}`, two interior points. The fallback may be taken only for
a stated compute reason, must be declared before any run, and changes no rule
below.

## 2. Construction (deterministic, no sampling)

New graph kind `matching_partial`, parameterised by integer `k`:

- Preserve true pairs `(0,1), (2,3), ..., (2k-2, 2k-1)` — the first `k` pairs.
- Re-match the tail block `{2k, ..., N-1}` by the cyclic shift already used by
  `matching_wrong`, restricted to that block:
  `(2k+1, 2k+2), (2k+3, 2k+4), ..., (N-1, 2k)`.
- The tail contains no true pair whenever it holds at least 4 agents, which is
  exactly the condition `k <= m-2` — the same bound as §1.

`k = m` reduces to `matching`; `k = 0` reduces to `matching_wrong`. Both are
**re-run fresh in this cell** rather than merged from historical manifests, so
every arm shares one trainer signature and one seed set. No post-hoc merging.

Required code change: one branch in `CoordGrid._build_adjacency`, one entry in
`_VALID_GRAPHS`, and adjacency unit tests asserting 1-regularity, the intended
overlap `k`, and absence of any true pair in the tail. No algorithm, trainer, or
metric code is touched.

## 3. Arms, budget, seeds — frozen

- Env: `CoordGrid`, `pair_routing=True`, `obs_mode="ego"`, `N = 10`, grid 3 —
  identical to the established positive-endpoint configuration except `N`.
- Arms: `gnn_p00`, `gnn_p20`, `gnn_p40`, `gnn_p60`, `gnn_p100` (byte-identical
  `algo_kwargs`, differing **only** in the `graph` kwarg), plus `mlp`, the
  parameter-matched no-graph control.
- Seeds 0–11 (**12**), every arm. **150,000** env steps, every arm. Both frozen
  here and not adjustable after any result is seen.
- Metric: **final-window (last 20%) mean return per run**, one scalar per run.
  House standard, unchanged.

## 4. Statistics — frozen

Welch two-sided `t` with **Holm correction over exactly the family below**, plus
two-sided Mann–Whitney `U` as the robustness test; Cohen's `d` (pooled SD);
Student-`t` 95% CIs. **House rule: significance requires BOTH tests.** Wiring
reuses `gnnmarl.utils.stats` and the `scripts/phaseC_analysis.py` conventions.

**Contrast family — exactly 9, fixed here:**

- 5 vs-control: `gnn_p{00,20,40,60,100}` − `mlp`
- 4 adjacent-level: p20−p00, p40−p20, p60−p40, p100−p60

No contrast outside this family enters any corrected claim. Anything else
computed is exploratory and must be labelled as such.

## 5. Validity gate (runs before anything is interpreted)

**G1 — endpoint replication at `N = 10`.** `gnn_p100` − `mlp` must be positive,
with `d >= 2`, and significant under both tests after Holm.

If G1 fails the cell is **uninterpretable, not a finding**: the dose–response
question presupposes a benefit to grade. On failure we report the failure and
the `N = 10` endpoint check, make no shape claim, and do not substitute another
`N` to recover a result.

## 6. Decision rules — the shape question, banded before seeing data

Let `A(p) = mean(gnn_p) - mean(mlp)` and
`theta(p) = (A(p) - A(0)) / (A(1) - A(0))`.

**R1 — monotonicity.** The curve is called monotone iff no adjacent-level
contrast is *significantly negative* under both tests after Holm. A
non-significant dip is not non-monotonicity.

**R2 — shape**, read from `theta(0.4)` and `theta(0.6)`, the interior points
bracketing the midpoint. Bands fixed now:

| condition | verdict | practitioner reading |
|---|---|---|
| `theta(0.6) < 0.35` | **cliff** | benefit accrues only near-correct; a partly-right graph buys little |
| `theta(0.4) > 0.65` | **threshold** | most of the benefit is recovered by a partly-correct graph |
| otherwise | **graded** | benefit degrades roughly in proportion to mis-specification |

R2 is descriptive and reported with CIs on `theta` (bootstrap over seeds, 10,000
resamples, seed 0). It is **not** a significance claim and must never be written
as one.

**R3 — floor check.** If `gnn_p00 − mlp` is not significantly negative or
near-zero, the wrong-graph endpoint has not reproduced at `N = 10`; report that,
treat the `p = 0` anchor of `theta` as unvalidated, and widen every shape
statement accordingly.

## 7. What this cell may NOT be used to claim

- Nothing about the **transfer** question. This is the same constructed
  archetype family; it grades mis-specification *within* the regime where the
  positive endpoint holds, and says nothing about third-party dynamics. The
  scoping in the positive contribution and in limitations item 4 stands
  unchanged whatever this returns.
- Nothing about **multi-hop routing**; the relay cell's pre-registered negative
  is untouched.
- No **mechanism** for any shape found. A cliff or a threshold is a measurement,
  not an explanation.
- No retro-fitting. If the cell is never run, this file is not evidence for
  anything, and no other analysis may cite it as a registered prediction.

## 8. Amendment policy

Any change to §§2–6 must be a **new commit** to this file, dated, stating what
changed and why, and made before the affected data exists. Editing this file
after any Phase-E run exists invalidates the cell.
