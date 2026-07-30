# Final number audit — Phase C classical (DCG) baseline integration into `paper/main.tex`

Date: 2026-07-30. Auditor: independent re-derivation from raw manifests, read-only
except for this report. Environment: repo `.venv` — Python 3.13.1, numpy 2.5.0,
scipy 1.18.0, pandas 3.0.3, torch 2.12.1+cpu (matches the version pinned in
`PREREGISTRATION.md §5`).

Recomputation method: independent script reading the three raw 150k manifests
(`manifest_cell{A,B,C}_*_150000steps.csv`) plus the Cell-B `gnn_true` merge
source (`results/phaseC_150k/manifest_pair_N6_ego_attention.csv`, per
`PREREGISTRATION.md §2c`), using the repo's own `gnnmarl.utils.stats.holm_correct`
/ `mean_with_ci` plus `scipy.stats.ttest_ind` (Welch, two-sided) /
`mannwhitneyu` (two-sided) / pooled-SD Cohen's d — the same primitives
`scripts/phaseC_analysis.py` uses for the sibling Stage-C pipeline. No
committed file was modified; the manifests' own `final` column (last-20%
per-run mean, computed at run time by `scripts/phaseC_classical_sweep.py`)
was used directly as the n=12 per-arm sample, matching the frozen metric
definition.

## Check 1 — every `\CB*` macro vs. manifests (via `ANALYSIS.md`)

*Methodology note: the first pass of this audit was a manual/eyeball
cross-check and found 3 of the following. A second, independent reviewer
pass caught a 4th by hand. To make the count defensible this was then
replaced with an exhaustive, non-manual sweep: every `\CB*` macro and every
bare table literal in `tab:classical` (80 values total — mean/median/sd/CI-lo/
CI-hi for all 16 arm-rows across A/B/C) was compared programmatically against
`round(raw_value, 2)` computed directly from the manifests
(`exhaustive_check.py`, string-equality on the 2dp value, no eyeballing). This
found a 5th. The 80-value sweep is reported below in full; all further
sub-checks (advantage/Cohen's-d/p-value macros, floor references) were
already verified independently in Check 3.*

**Result: 5 discrepancies found out of 80 checked values (all four are
`\CB*` macros; the sd is a bare table literal, reported under Check 2), all
in Cell A / Cell B / Cell C:**

| macro / location | file:line | manuscript | recomputed (raw, full precision) | correct 2dp |
|---|---|---:|---:|---:|
| `\CBAmlp` (mlp mean, Cell A) | `paper/main.tex:278`, used at `:1875`, `:1963` | **83.66** | 83.654583333... | **83.65** |
| `\CBCwrong` (dcg_jointobs_wrong mean, Cell C) | `paper/main.tex:318`, used at `:1975` | **56.45** | 56.444895833... | **56.44** |
| `\CBCcomplete` (dcg_jointobs_complete mean, Cell C) | `paper/main.tex:320`, used at `:1976` | **57.10** | 57.094826388... | **57.09** |
| `\CBCmlpCI` lower bound (mlp 95% CI, Cell C) | `paper/main.tex:313`, used at `:1916`, `:1978` | **[58.70**, 63.53] | 58.694775... | **[58.69**, 63.53] |
| Table B `gnn_true` row, sd column (bare literal, not a macro) | `paper/main.tex:1971` | **5.06** | 5.054608149... | **5.05** |

Root cause (identical for all five): `ANALYSIS.md` displays these values at
3 d.p. (83.655, 56.445, 57.095, 58.695, 5.055 — each itself a correct 3dp
rounding of the raw float), and the 2dp macros in `main.tex` were produced by
re-rounding the *already-rounded* 3dp display value a second time
(83.655→83.66, 56.445→56.45, 57.095→57.10, 58.695→58.70, 5.055→5.06) instead
of rounding the raw float directly. Rounding the raw floats directly
(83.654583, 56.444896, 57.094826, 58.694775, 5.054608) gives 83.65, 56.44,
57.09, 58.69, 5.05 in every case — the 3rd decimal digit is unambiguously 4
in all five (not a genuine `.xxx5` tie), confirmed with both Python's
`round()` and `decimal.Decimal(...).quantize(..., ROUND_HALF_UP)` on the raw
float. This is a systematic transcription bug (double-rounding through an
intermediate 3dp representation), not a rounding-convention judgment call.

All other 75 of the 80 checked values (means/medians/sds/CI-bounds for every
one of the 16 arm-rows across Cells A/B/C, plus the reference-point floors
`\CBAfloor{4.02}`/`\CBCfloor{33.37}`/`\CBCoracle{99.11}` against
`reference_points.csv`, plus `\SCkTrue{94.96}`, the frozen 150k `gnn_true`
anchor merged into Cell B) matched the recomputed values exactly at 2 d.p.
The floor-significance figure `p{=}4{\times}10^{-23}` (Cell A dcg_jointobs
vs. random floor) also matches (recomputed 4.264e-23).

One boundary case verified as *not* an error: Cell A `dcg_canonical` median is
exactly 75.965000 (average of the two middle order statistics,
75.42916̄ and 76.50083̄), a genuine `.xxx5` tie. `main.tex`'s table literal
75.97 corresponds to round-half-up (the conventional reporting rule); Python's
default `round()` uses banker's rounding and gives 75.96. This is a legitimate
convention choice, consistently applied to the one true tie in the whole
80-value sweep (every other value's 3rd decimal is unambiguously 0-4 or 6-9,
never exactly on a `.xxx5` boundary before display-rounding) — noted, not
flagged as FAIL. It is the same round-half-up convention that, applied
correctly to the *raw* float, produces the five *correct* values above
(83.65, 56.44, 57.09, 58.69, 5.05) — the bug is specifically double-rounding
through the 3dp intermediate, not the choice of tie-breaking rule.

## Check 2 — Table `tab:classical` literals vs. recomputation

All literal cells in the table (`paper/main.tex:1960-1980`), whether sourced
from a `\CB*` macro or hard-coded as a bare number, were included in the
80-value exhaustive sweep in Check 1 (since every one of the table's cells is
either a macro reference or a bare literal representing a mean/median/sd/CI
bound for one of the 16 arm-rows). **All 5 discrepancies found surface in
this table** (4 via the macros they reference: `\CBAmlp`, `\CBCwrong`,
`\CBCcomplete`, `\CBCmlpCI`; 1 as a bare literal that is not a macro at all:
the Cell-B `gnn_true` row's sd column, `5.06` at `paper/main.tex:1971`, which
should be `5.05`). No table cell beyond these five was found to differ from
`round(raw_value, 2)`.

Completeness grep: each of the five wrong raw strings (`83.66`, `5.06`,
`56.45`, `57.10`, `58.70`) was grepped across all of `paper/main.tex`; each
appears in exactly one place — its macro definition or (for the bare sd
literal) the one table cell — never as a second, stray hard-coded occurrence
elsewhere in prose. No hidden second copy to also fix.

**Net effect on the manuscript: none of these five cells affects a
significance verdict, a Cohen's d, a p-value, or any claimable statement** —
all downstream inferential statistics (Welch/MWU/Holm, Cohen's d, all `*Gap`
advantage macros) were independently recomputed from the raw per-seed data
(not from the rounded display macros) and match exactly (see Check 3). These
are pure last-digit transcription slips of ≤0.01 each, confined to table
`tab:classical`'s display cells (and, for `\CBAmlp`, one prose mention at
`main.tex:1875` that just repeats the table's mean).

## Check 3 — load-bearing statistical claims (pre-registered family, `PREREGISTRATION.md`)

Recomputed the full pre-registered contrast family (Welch two-sided + Holm
within-cell + Mann–Whitney two-sided + pooled-SD Cohen's d) for all three
cells directly from the raw manifests. **All values match `ANALYSIS.md` and
`paper/main.tex` exactly** (to the reported precision):

- **Cell C, gnn_true vs mlp** (the crux): adv +2.890, d=+1.026, Welch
  p_holm=5.174e-02 → **0.052**, MWU=7.825e-02 → **0.078**. Means 64.004→**64.00**
  vs 61.114→**61.11**. Medians 63.841→**63.84** vs 63.329→**63.33**. Matches
  the manuscript exactly.
- **Cell B, gnn_true vs dcg_jointobs_true** (the concession): adv +4.467,
  d=+0.366→**0.37**, Welch p_holm=3.858e-01→**0.386**, MWU=0.7075. Means
  90.497→**90.50** vs 94.964→**94.96**. Matches.
- **Cell B, gnn_true vs {dcg_canonical_true, dcg_jointobs_complete,
  dcg_jointobs_wrong}**: d = 12.506, 12.474, 12.563 — all **d>12**, Welch
  p_holm = 1.50e-11 in all three (< 1e-10 as claimed). Matches "the d>12 floor
  collapses."
- **Cell C, all 6 gnn_true-centred contrasts** and the exploratory
  mlp-centred family (§6a of `ANALYSIS.md`): every raw/Holm/MWU value
  recomputed matches to the reported significant figures.
- **Cell A floor-clear**: dcg_jointobs vs. A-floor Welch p = 4.264e-23,
  matches the manuscript's `4×10^{-23}`.
- Note (does not touch the manuscript): `ANALYSIS.md §6d`'s exploratory
  "`dcg_canonical` floor-clear at 74.87, p=2.6e-13" is the **one-sided**
  Welch p-value (recomputed two-sided p = 5.190e-13, one-sided/half =
  2.595e-13). `ANALYSIS.md`'s own stated methodology (§0, "two-sided Welch's
  t") is inconsistent with this one exploratory-section number. It is not
  quoted anywhere in `main.tex` (confirmed by grep), so it does not affect
  the manuscript, but flagging it as an internal `ANALYSIS.md` documentation
  slip.

**Decision-rule outcomes** (Rule 1 not satisfied / Rule 3 satisfied as a
partial concession / Rule 2 not satisfied) were re-derived from these
recomputed numbers and match `ANALYSIS.md §4` and the manuscript's Cell A/B/C
paragraphs (`paper/main.tex` §sec:structure:classical) exactly.

## Check 4 — prohibited-claims grep

Grepped both `paper/main.tex` and `paper/main_jaamas.tex` (regenerated fresh,
see Check 6) for:

- `"double dissociation"` (as a demonstrated/claimed result) — **0 hits** in
  either file. (Design-doc/pre-reg language used "double dissociation" only
  as the *name of the hypothesis being tested* in `PREREGISTRATION.md`/design
  doc, which are not manuscript text; the manuscript itself never asserts it
  as achieved — Rule 1 is explicitly reported NOT satisfied.)
- `"routes where payoff propagation cannot"` (absolute form) — **0 hits**.
  The manuscript's actual language is the hedged relative-ordering version
  ("That ordering is informative for GNN-vs-DCG, not evidence of routing" /
  "leaving multi-hop routing open" in the abstract).
- `"solves 2-hop"` / `"solves multi-hop routing"` for gnn — **0 hits**. The
  only "(near-)solves" language in the document refers to TokenMatch
  (`\TMtrue`, 57.59/60 ≈ 96%), a different task/metric, not the 2-hop relay.
  Cell C prose explicitly states "*Neither method demonstrates it*" and "no
  method routes here."
- `"our experiments do not test"` (re: classical solvers) — **0 hits**; this
  sentence has been superseded by the new `sec:structure:classical`
  subsection, as expected.

**PASS — no prohibited claims found in either file.**

## Check 5 — pre-existing headline numbers, drift check

Five numbers spot-checked against their existing source CSVs, independent of
today's `\CB*` integration:

| number | macro | manuscript | recomputed from source | source |
|---|---|---:|---:|---|
| Cell-B `gnn_true` frozen anchor | `\SCkTrue` | 94.96 | 94.963854 → 94.96 | `results/phaseC_150k/manifest_pair_N6_ego_attention.csv` |
| TokenMatch `gnn_true` | `\TMtrue`/`\TMtrueLo`/`\TMtrueHi` | 57.59 [57.55, 57.63] | 57.592778 → 57.59, CI [57.5515, 57.6341] → [57.55, 57.63] | `results/phaseD/token_N6/gnn_true__*` (12 seeds) |
| d≈5→12 pair (attention-over-complete, 30k→150k) | `\SCtrueAttCompD`/`\SCkTrueAttCompD` | 5.5 → 12 | d=5.41–5.52 (30k) → d=11.02–12.14 (150k), gat_true vs gnn_true | `results/phaseC/manifest_pair_N6_ego_attention.csv`, `results/phaseC_150k/manifest_pair_N6_ego_attention.csv` |
| Phase-8 scaling parity, N=12 | `\PSgapNtwelve` | −153.4 | 92.075167 − 245.459167 = −153.384 → −153.4 (gnn_qmix − mlp_qmix) | `results/phase11_scale/summary_aggregate.csv` |
| (confirms Table columns) N=12 QMIX/MLP/GNN | table literals | 252.6/245.5/92.1 | 252.561/245.459/92.075 | same file |

All five match exactly. `git log` confirms none of the source files or macro
definitions for these five were touched by any commit in the DCG-integration
chain (`d7b8dd3`..`b12f03f`) or the subsequent abstract/reframing commits
(`2babbe5`, `a738d5f`, `a59fa4d`) — last touched by unrelated, older commits
(`188eb31`, `d84ca8d`, etc.). **PASS — no drift.**

## Check 6 — abstract word count and JAAMAS regeneration

Abstract (`paper/main.tex:342-368`), **macro/math-stripped word count (as
specified): 246 words** (regex-strips `$...$` math and bare `\command`
tokens entirely, unwraps `\emph{...}`/`\textbf{...}`); the assembly script's
own independent crude counter (`paper/_assemble_jaamas.py`'s built-in
word-count diagnostic, a different regex with the same strip-not-count
policy) corroborates at **245 words**. Both give a comfortable ~4-5 word
margin under the 250-word TMLR limit — **PASS**. (A separate, more lenient
count that treats every bare macro token as one rendered word, approximating
what a reader of the compiled PDF would see, e.g. `\coordgrid` → "CoordGrid",
gives 250 — also ≤ 250, but that is a different metric than the one
requested and is reported here only for completeness, not as the governing
figure.)

`python paper/_assemble_jaamas.py` was re-run from a clean read of the
current `paper/main.tex` / `paper/_jaamas_preamble.tex`. `git status --short
paper/main_jaamas.tex` and `git diff --stat paper/main_jaamas.tex` after
regeneration show **zero diff** — `main_jaamas.tex` is byte-stable and
matches what's committed. **PASS.**

## Check 7 — test suite and commit-chain integrity

- `pytest -q` (repo `.venv`): **160/160 tests pass**, exit code 0
  (`pytest --collect-only` confirms 160 tests collected across 14 files,
  matching `PREREGISTRATION.md §5`'s "full suite 160 tests green at pre-reg
  time"). No regressions.
- `ruff check src tests`: **clean** (all checks passed) — this is the exact
  command CLAUDE.md's "Commands" section specifies.
- `ruff check src tests scripts` (the broader command named in this audit's
  mandate): **NOT clean — 22 pre-existing errors**, all in `scripts/`
  figure/analysis utilities (`build_paper_inserts.py` — 1 unused import + 8
  `invalid-syntax` backslash-in-f-string findings that are a ruff/Python-3.10
  target-version false-positive on a script that isn't part of the run
  path, `fig_overview.py`, `phase11_scale_figure.py`, `phase1_analysis.py`,
  `phaseC_figures.py`). None of these files were touched by the DCG
  integration; `git log -1` shows their last edits at `188eb31`, `d84ca8d`,
  `b670e42` — all well before the `d7b8dd3..b12f03f` pre-reg/implementation/
  results chain. The new files added by this integration
  (`scripts/phaseC_classical_sweep.py`, `scripts/phaseC_classical_reference_points.py`)
  and the existing `scripts/phaseC_analysis.py` are individually lint-clean.
  **Flagged as a pre-existing lint-debt finding, not a regression from
  today's work** — but per the mandate's literal wording this command is not
  currently green and should be cleaned up or the mandate scoped down to
  `src tests` explicitly.
- Commit-chain order: `git merge-base --is-ancestor d7b8dd3 b12f03f` and
  `...21ef1c5 b12f03f` both confirm **YES** — pre-registration (`d7b8dd3`,
  2026-07-30 14:43:41) and implementation (`21ef1c5`, 15:06:33) are both
  ancestors of the confirmatory-results commit (`b12f03f`, 18:50:20), same
  day, correct order, no timestamp inversion.
- `git status --short results/phaseC_classical/ANALYSIS.md
  results/phaseC_classical/PREREGISTRATION.md
  results/phaseC_classical/manifest_cell{A,B,C}_*_150000steps.csv`: **empty**
  — the working tree exactly matches what's committed; no uncommitted edits
  to any of the audited source-of-truth files.

## Summary verdict

| check | verdict | notes |
|---|---|---|
| 1. `\CB*` macros vs. manifests | **FAIL (minor)** | 4 macros off by 0.01 each (exhaustive 80-value programmatic sweep): `\CBAmlp` 83.66→83.65; `\CBCwrong` 56.45→56.44; `\CBCcomplete` 57.10→57.09; `\CBCmlpCI` lo 58.70→58.69 |
| 2. Table `tab:classical` literals | **FAIL (minor)** | 1 additional bare literal off by 0.01: Cell-B `gnn_true` sd, main.tex:1971, 5.06→should be 5.05 (the four macro errors above also surface in this table) |
| 3. Load-bearing stats (Welch/MWU/Holm, Cohen's d) | **PASS** | All recomputed values match exactly; verdicts (Rule 1 not satisfied, Rule 3 partial concession, Rule 2 not satisfied) reproduce; unaffected by the Check 1/2 display bug since advantages/d/p are computed from raw per-seed data, not the rounded macros |
| 4. Prohibited-claims grep | **PASS** | 0 hits for all four prohibited patterns in both `main.tex` and `main_jaamas.tex`; superseded sentence confirmed gone |
| 5. Pre-existing headline numbers (drift) | **PASS** | 5/5 match source CSVs exactly; no commit in today's chain touched them |
| 6. Abstract word count / JAAMAS regeneration | **PASS** | 245-250 words (method-dependent) ≤ 250; `main_jaamas.tex` regenerates byte-identical |
| 7. Test suite / lint / commit chain | **PASS with a flag** | 160/160 tests green; `ruff check src tests` clean; `ruff check src tests scripts` has 22 pre-existing (non-regressed) errors; commit order verified correct |

**Overall: 5 PASS, 2 minor FAILs (five total display-value rounding slips,
≤0.01 each, confined to Table `tab:classical` in Cells A/B/C and one prose
echo of one of them — Check 1 and Check 2 are the same underlying bug class
viewed from the macro side and the table side respectively; none affects any
p-value, Cohen's d, significance verdict, or claimable statement), 1
PASS-with-flag (pre-existing `scripts/` lint debt, not a regression from
today's work).**

### Recommended fix (four macro one-line edits + one table literal, no re-analysis needed)

```
paper/main.tex:278  \newcommand{\CBAmlp}{83.66}         ->  \newcommand{\CBAmlp}{83.65}
paper/main.tex:313  \newcommand{\CBCmlpCI}{[58.70, 63.53]} -> \newcommand{\CBCmlpCI}{[58.69, 63.53]}
paper/main.tex:318  \newcommand{\CBCwrong}{56.45}       ->  \newcommand{\CBCwrong}{56.44}
paper/main.tex:320  \newcommand{\CBCcomplete}{57.10}    ->  \newcommand{\CBCcomplete}{57.09}
paper/main.tex:1971 ... & 96.28 & 5.06 & [91.75, 98.18] \\   ->  ... & 96.28 & 5.05 & [91.75, 98.18] \\
```
`\CBAmlp` and `\CBCmlpCI` each fix one additional prose usage automatically
(`main.tex:1875` and `main.tex:1916` respectively — both just echo the macro,
no separate edit needed there). After the edits, re-run
`python paper/_assemble_jaamas.py` (`main_jaamas.tex` carries these macros
over automatically) and rebuild both PDFs.
