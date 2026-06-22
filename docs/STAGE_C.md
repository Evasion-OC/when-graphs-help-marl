# Stage C — the graph-STRUCTURE test (pre-registration)

This is the pre-registered protocol for the positive-result hunt's structure
test. Hypotheses, arms, metric, and the decision rule are fixed **here, before**
the confirmatory multi-seed sweep, so a positive turn is credible and not
HARKing/p-hacking. Reported honestly whichever way it lands.

## Why this test exists (the advisor's crux)

Stages A/B failed because the candidate tasks were either reducible to a global
*convention* (the graph is never needed) or genuinely needed *communication* but
not graph *structure*. A "graph helps" result that only beats no-communication
baselines is provable a priori (you cannot route information an agent is barred
from observing) and speaks to **communication**, not **structure** — which is
this project's actual experimental variable.

The decisive question is therefore: **does the *correct* graph beat a *wrong*
graph of matched density, and beat all-to-all, at identical parameters?** Only a
"yes" isolates graph *structure*.

## Task: pair-routing (structure is load-bearing by construction)

`CoordGrid(pair_routing=True)`, 3×3 toroidal grid, 25-step episodes, `obs_mode=ego`.

- Every agent holds a **private goal** it alone observes (shown in its own obs
  under any obs_mode; never any other agent's goal).
- A fixed perfect **matching** pairs the agents `(0,1),(2,3),…`. Each agent is
  rewarded (dense, toroidal proximity) for reaching its **partner's** goal.
- So partners must **swap** goals over their edge. The information an agent needs
  is held by **one specific other agent** — unlike a single broadcast goal, an
  all-to-all or wrong graph delivers the wrong/diluted partner. Structure is
  essential by construction.
- `max_neighbors_override=N-1` holds `obs_dim` constant across comm graphs, so the
  arms are byte-for-byte identical in I/O and parameters.

## Arms (identical gnn_qmix architecture + params; ONLY the comm graph differs)

| arm | comm graph fed to message passing | role |
|-----|-----------------------------------|------|
| `mlp` | none (MLPStack, ignores adj) | no-channel floor, parameter-matched |
| `gnn_true` | the **true** matching (`graph=matching`) | comm == task structure |
| `gnn_wrong` | a **wrong** matching, same density (`graph=matching_wrong`) | comm ≠ task; isolates *which* edges |
| `gnn_complete` | **all-to-all** (`graph=complete`) | most communication, unfocused |

`gnn_true`/`gnn_wrong`/`gnn_complete` are the *same* algorithm (repaired GCN,
2 layers, residual+LayerNorm, hidden 64) — only `env.adjacency()` differs. So any
gap between them is the **graph structure**, with capacity, depth, degree
(true vs wrong are both 1-regular) and observations all held fixed.

## Pre-registered hypotheses

- **H-struct (primary):** `gnn_true` achieves higher final-window mean return than
  **each** of `gnn_wrong`, `gnn_complete`, and `mlp`. *Falsification:* `gnn_true`
  fails to beat any one of the three, or the gap is not significant.
- **H-comm-insufficient (the structure discriminator):** `gnn_complete` does **not**
  beat `mlp` meaningfully (extra communication without structure ≈ no help).
- **H-N (secondary, dose-response):** the *total* `gnn_true − mlp` advantage grows
  with team size N (more pairs to route), tested at N ∈ {4,6,8,10}.

## Metric & statistics (identical to the rest of the paper)

- **Metric:** final-window mean return = mean over the last 20% of episodes.
- **Seeds:** 12 (> 10), seeds 0–11. (A 3-seed smoke confirmed the regime is alive
  and matched the predicted ordering; the 12-seed run is the confirmatory test.)
- **Budget:** 30k env steps (same as Stage B).
- **Tests:** Welch's t (two-sided) Holm-corrected across the comparison family,
  AND two-sided Mann–Whitney U. Cohen's d for effect size. 95% Student-t CIs.
- **Significance:** `gnn_true` > X requires Holm-corrected Welch p < 0.05 **and**
  MWU p < 0.05, for X ∈ {gnn_wrong, gnn_complete, mlp}.

## Decision rule → claim

- **H-struct supported** (gnn_true beats all three, significant) → **genuine
  positive, structure-level**: *graph structure helps cooperative MARL when
  message passing must be selective — routing information along task-aligned edges
  beats both no communication and all-to-all communication.* Pursue the positive
  paper; map N (and richer task graphs) as the boundary.
- **gnn_true ≈ gnn_complete** (both beat mlp) → only **communication** matters,
  not structure. Report honestly; this is essentially the negative paper reworded.
- **gnn_true ≈ mlp** → the graph does not help even here. Lock the negative paper.

## Run

```bash
python scripts/phaseC_structure_sweep.py --seeds 12 --obs ego --workers 16
python scripts/phaseC_analysis.py
# secondary dose-response:
for N in 4 6 8 10; do python scripts/phaseC_structure_sweep.py --seeds 12 --n-agents $N --workers 16; done
```
