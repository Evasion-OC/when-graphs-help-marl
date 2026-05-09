# Paper draft

`main.tex` is a TMLR-aimed draft titled **"When Does Graph Structure Help in
Multi-Agent Reinforcement Learning? A Controlled Empirical Study"**.

## Status

This is **Phase 5a** — the data-supported sections (intro, methods, Phase 1
and Phase 2A results, discussion, related work, conclusion) are written
against the real numbers from `results/pilot/` and `results/phase_2_tune/`.
Phase 2C/3/4 result sections are placeholders marked with the
`\todoFromPhase{...}` macro so they jump out in the rendered PDF and grep.

When subsequent phase data lands:

| Phase | Section to update |
|---|---|
| 2C | `\section{Results: locked-hyperparameter main sweep}` |
| 3  | `\section{Results: external validity on simple\_spread}` |
| 4  | `\section{Results: depth $\times$ diameter ablation}` |

After all four placeholders are filled, the paper is Phase 5b — the final
prose pass.

## Building

```
cd paper
latexmk -xelatex main.tex     # or pdflatex + biber + pdflatex twice
```

The figures in `\includegraphics{...}` use relative paths into
`../results/pilot/` and `../results/phase_2_tune/`. Building from `paper/`
finds them via the `..`.

## What's deliberately not in the paper

- **Survey fluff.** The original draft was 30+ pages of literature claims.
  TMLR has no length limit, but reviewers reward precision over volume.
  Related Work is one short paragraph.
- **Theorems.** The original draft's "three theorems" were restatements
  of published results (Weisfeiler--Leman / GIN expressiveness). Dropped.
- **Meta-analysis percentages.** The original draft had fabricated forest
  plot numbers. Replaced by real numbers from this study's own data.
- **AGI framing.** The current claim is narrower and falsifiable: when
  does graph structure help in cooperative MARL with value decomposition.
  TMLR does not require grand framing.

## Citation hygiene

`references.bib` contains only hand-verified entries. No "and others"
placeholders, no duplicate-paper-different-key bugs. New entries must be
verified before being added.

## Macros

`main.tex` uses `\newcommand` for empirical numbers we'll keep updating
across phases:

```latex
\newcommand{\TunedFinalQMIX}{0.28}    % Phase 2A locked QMIX
\newcommand{\TunedFinalGNN}{0.34}     % Phase 2A locked GNN-QMIX
% ... etc
```

Update these in one place when new Phase 2C/3/4 numbers come in.
