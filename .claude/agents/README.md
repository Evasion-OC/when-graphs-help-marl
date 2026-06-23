# Research lab — Claude subagents

Project-scoped specialist agents for the *when-graphs-help-marl* study. Invoke with
the Agent tool or `@<name>` (e.g. `@reviewer-adversarial review the Stage C result`).
They are tailored to this project's pre-registration discipline, stats protocol, and
the negative-result → positive-boundary arc.

## Research & engineering
| agent | role | model |
|-------|------|-------|
| `marl-research-scientist` | PI: hypotheses, mechanisms, interpretation, next experiments | opus |
| `rl-systems-engineer` | implements algos/envs/trainer in PyTorch, tested, ruff-clean | sonnet |
| `experiment-scientist` | designs & runs parallel sweeps / calibrations / dose-response | sonnet |
| `research-statistician` | tests, effect sizes, power, anti-p-hacking guardrail | opus |
| `reproducibility-engineer` | determinism, pinning, test suite, pipeline integrity | sonnet |
| `literature-scholar` | related work, citations, novelty positioning (web access) | opus |
| `results-analyst` | manifests → tables & publication figures | sonnet |
| `academic-writer` | abstract/intro/results prose, LaTeX, claim discipline | opus |
| `latex-engineer` | .tex → clean PDF: zero errors/warnings/overfull, journal format + count audit | sonnet |
| `theory-propositions-advisor` | propositions (not theorems), mechanism/ceiling checks | opus |

## Review panel (different personalities)
| agent | personality | model |
|-------|-------------|-------|
| `reviewer-adversarial` | brutal "Reviewer 2", reject-by-default, lethal-but-fair | opus |
| `reviewer-methodologist` | exacting rigor/stats/repro referee | opus |
| `reviewer-balanced` | fair, constructive, median-strong reviewer | opus |
| `reviewer-champion` | generous advocate, finds & amplifies the signal | sonnet |

## Editorial
| agent | role | model |
|-------|------|-------|
| `area-chair-editor` | desk-reject screen + meta-review → decision + venue | opus |

## Suggested workflows
- **New result:** `marl-research-scientist` (design) → `experiment-scientist` (run) →
  `research-statistician` (verify) → `results-analyst` (figures).
- **Mock peer review:** all four `reviewer-*` in parallel → `area-chair-editor`
  synthesizes a decision.
- **Pre-submission:** `reproducibility-engineer` + `reviewer-methodologist`, then
  `academic-writer` tightens claims to survive `reviewer-adversarial`.
- **Final production:** `academic-writer` (prose final) → `latex-engineer` drives the
  build to a clean, journal-conformant PDF and audits all counts vs the venue limits.

Models are a sensible default (opus for judgment/writing/review, sonnet for
execution); override per call or edit the frontmatter. Move any file to
`~/.claude/agents/` to make it available across all projects.
