# JAIR submission package — READY TO UPLOAD

**Venue:** Journal of Artificial Intelligence Research (JAIR) — Diamond Open Access, **no submission fee, no APC**.

## What to upload

JAIR accepts **PDF only** for peer review ("this requirement is strictly enforced").
Upload exactly one file:

- **`main_jair.pdf`** ← the manuscript (34 pp). This is the only file the review system needs.

Do **not** upload the LaTeX source now. `latex_source/` is bundled here only for the
**camera-ready stage after acceptance** — JAIR requests source then, not at submission.

## How to submit (online wizard)

1. Register / log in: https://jair.org/index.php/jair/user/register
2. Start the submission wizard: **https://jair.org/index.php/jair/submission/wizard**
3. Upload `main_jair.pdf`.
4. Fill the form fields with the metadata below.
5. Confirm the declarations (originality, unpublished, reproducibility checklist completed).

## Form metadata (copy-paste)

**Title:**
When Does Graph Structure Help in Multi-Agent Reinforcement Learning? A Controlled Boundary

**Authors:**
- Ali Jabbary — Independent Researcher, Urmia, Iran. **Corresponding author.**
  ORCID 0000-0003-0573-6909 · research@alijabbary.com
- Kasra Ghanavati — School of Computing and Mathematical Sciences, University of Greenwich,
  London, United Kingdom. ORCID 0009-0009-0888-3307 · kg1111r@gre.ac.uk

**Primary area:** Reinforcement Learning / Multi-Agent Systems.
(Do *not* select "Graph Neural Networks" as primary — the GNN is a deliberately minimal
control, not the contribution.)

**Article type:** Research article (computational experiments + light propositions).

**Abstract (291 words — paste verbatim from the PDF):**

> Graph neural network (GNN) value-factorisation methods are widely held to improve
> cooperative multi-agent reinforcement learning (MARL) by exploiting relational structure,
> yet when they actually help has not been isolated. We settle the question with a controlled
> study on CoordGrid, a cooperative gridworld whose coordination graph is the experimental
> variable, with the encoder, mixer, and optimiser held fixed across algorithms. The benefit
> is real but conditional, and we identify exactly when it appears. When an agent must act on
> information held by one specific, unobserved other — a private-goal routing task — a graph
> whose edges match the task's coordination structure decisively outperforms all-to-all
> communication, a density-matched wrong graph, and a parameter-matched no-graph control, at
> byte-identical parameters and observations (Cohen's d of about five; complete seed-level
> separation over twelve seeds, Holm-corrected). The advantage is the graph structure, not the
> communication channel or the capacity: all-to-all carries the partner's signal but cannot
> select it, and even learned attention recovers it only on the right graph. The effect holds
> across team size and observability and replicates out of harness in a non-spatial
> token-matching game. When coordination is instead incidental or convention-reducible, the
> same GNN underperforms a structurally simpler baseline, and a matched control attributes the
> loss to the aggregation itself, not the added capacity — a penalty that persists with more
> training, grows with team size, and recurs on standard MPE and Level-Based Foraging tasks.
> We give a precise rule — route along task-relevant edges, not all-to-all, only when agents
> must act on information held by specific others they cannot observe and the coordination
> topology is known or reliably estimable — and state scope plainly: these are controlled,
> constructed tasks, with replication on a standard third-party benchmark left to future work.
> We release the full code and configurations.

## Mandatory reproducibility checklist

Already completed and appended inside `main_jair.pdf` (final section, "Reproducibility
Checklist for JAIR"). Omitting it = desk reject; it is present, so this gate is cleared.

**Code/data repo (linked in the checklist):** https://github.com/Evasion-OC/when-graphs-help-marl
→ **Confirm this repo is PUBLIC before you submit** (JAIR is single-blind; a public repo is
expected). This is the one remaining manual check.

## Package contents

```
submission_jair/
├── main_jair.pdf          ← UPLOAD THIS (PDF only, the deliverable)
├── SUBMISSION_README.md   ← this file
└── latex_source/          ← camera-ready only; NOT needed at submission
    ├── main_jair.tex, references.bib, *.cls/.sty/.bbx/.cbx/.dbx
    ├── figures/  (17 PDFs)
    └── results/  (phase1/2/4 .tex)
```
