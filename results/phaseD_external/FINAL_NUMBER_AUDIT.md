# Final number audit — Phase D external-validity integration into `paper/main.tex`

Date: 2026-07-31. Auditor: independent re-derivation from raw manifests/episode
logs, read-only except for this report and the one fix documented in Check 1.
Repo `.venv`: Python 3.13.1, numpy 2.5.0, scipy 1.18.0, pandas 3.0.3, torch
2.12.1+cpu (same environment the sibling
`results/phaseC_classical/FINAL_NUMBER_AUDIT.md` used). **Audited commit:
`01b3123`** (branch `phaseD-external-validity`) — at that commit, both
`paper/main.tex` and `results/phaseD_external/ANALYSIS.md` contain the wrong
value found in Check 1. This audit's one-line fix (Check 1) is applied
**in the working tree only, not committed** — the fix is included in the
`git diff`s quoted below and is ready for the parent task/user to commit, but
this auditor did not create a commit (out of scope for a read-only audit
unless the user asks).

Recomputation method: an exhaustive, programmatic sweep — not a manual/eyeball
pass — matching the method the Phase-C classical audit converged on after its
first two manual passes under-counted. A script parses every
`\newcommand{\PD*}{...}`/`\newcommand{\CR*}{...}` definition directly out of
`paper/main.tex` via regex, infers each macro's own declared decimal precision
from its literal string, and asserts `round(raw_value, that_precision) ==
manuscript_value` — string/numeric equality, no eyeballing — against raw
values recomputed independently from the manifests/episode logs
(`results/phaseD_external/manifest_cellE1_N6_150000steps.csv`,
`manifest_cellE0_N2_150000steps.csv`,
`results/phaseC_classical/manifest_cellR_N6_ego_150000steps.csv` merged with
the committed `manifest_cellC_N6_ego_150000steps.csv` per the ADDENDUM,
`reference_points.csv`, `greedy_reference.csv`) using the repo's own
`gnnmarl.utils.stats.holm_correct`/`mean_with_ci` plus `scipy.stats`
Welch/Mann–Whitney and pooled-SD Cohen's d — the same wiring
`pairwise_compare` uses. Instability/onset statistics were recomputed directly
from `episodes.csv` under `results/phaseD_external/cellE1_N6_150000steps/`
and `cellE0_N2_150000steps/`, implementing `ANALYSIS.md §4.1`'s definitions
(rolling-50-episode median grad-norm exceeding 5× the episode-1000–2000
baseline for onset; rolling-100-episode return for the peak-window
exploratory metric and the peak→final drop).

**A first pass of this audit was largely a targeted, non-exhaustive check
(spot-comparing printed recomputed values against a read of the macro block)
and initially reported 0 discrepancies. An independent review of that pass
caught 1 real discrepancy the targeted check had missed** — the same failure
mode the Phase-C audit's own methodology note describes (manual/targeted
passes under-count; only an exhaustive, programmatic, string-equality sweep
is defensible). The sweep reported below is that exhaustive pass, run after
the fix, and finds 0 further discrepancies.

## Check 1 — every `\PD*`/`\CR*` macro vs. manifests (exhaustive sweep)

**80 `\PD*`/`\CR*` macros are defined in `paper/main.tex:332-419`.**

- **73 are numeric "value" macros** (means, medians, sds, CI bounds, gaps,
  Cohen's d, Holm p-values, blow-up counts, onset percentages, drops):
  checked programmatically at each macro's own declared precision. **After
  the fix below, 0 mismatches.**
- **7 are not directly sweepable this way** (labels/seed-counts, or Cohen's-d
  macros that use the `{-}` LaTeX minus-sign idiom, e.g. `\PDtmD{{-}1.73}`,
  which the parser's regex does not treat as a plain float) — each verified
  separately:
  - `\PDseeds{12}`, `\PDzeroSeeds{6}` — seed counts, confirmed against
    manifest row counts (E1: 60 rows / 5 arms = 12; E0: 12 rows / 2 arms = 6).
  - `\PDsteps{150{,}000}` — confirmed against `config.json`
    (`total_env_steps: 150000`) in every audited run directory.
  - `\PDparams{38{,}534}` — **independently re-derived at runtime**, not just
    taken from the pre-registration/`ANALYSIS.md`: `python
    scripts/phaseD_external_sweep.py --params-only --cell E1` prints
    `gnn_true`/`gnn_wrong`/`gnn_complete` all at `params=38534`, `mlp` at
    `38278`, `oracle` at `40774` — matches exactly, and independently
    confirms the byte-identical-parameters claim (`main.tex:2207`).
  - `\PDtmD{{-}1.73}`, `\PDpeakD{{-}3.29}`, `\PDzeroD{{-}1.05}` — hand-checked
    against the same raws the sweep used: `round(-1.734945, 2) = -1.73`,
    `round(-3.285367, 2) = -3.29`, `round(-1.045529, 2) = -1.05`. All match.

**One discrepancy found and fixed:**

| macro | file:line | manuscript (before) | raw (full precision) | correct 2dp |
|---|---|---:|---:|---:|
| `\PDhrCompleteMed` (median-based headroom fraction, `gnn_complete`) | `paper/main.tex:395`, used at `:2995` | **-3.13** | -3.135112975691261 | **-3.14** |

`round(-3.135112975691261, 2)` and
`Decimal("-3.135113").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)` both
give `-3.14` — the third decimal digit is unambiguously 5-followed-by-1
(i.e. strictly `> .xxx5`, not a genuine tie), so this is not a
rounding-convention judgment call.

**Root cause, confirmed arithmetically — this is the same double-rounding
class the Phase-C audit found, not an isolated typo.** All three
median-based headroom fractions in `ANALYSIS.md §1.1`/`§1.2` were computed by
dividing *already-rounded, 2dp-displayed* numerator and denominator values
rather than the raw floats:

```
ANALYSIS.md's own median-based gaps (§1.2): true -45.15, wrong -54.37, complete -65.10
ANALYSIS.md's own median-based denominator (§1.1): +20.77
-45.15 / 20.77 = -2.173808...  -> -2.17  (matches raw: -45.153736/20.765877 = -2.174420 -> -2.17)
-54.37 / 20.77 = -2.617718...  -> -2.62  (matches raw: -54.365773/20.765877 = -2.618034 -> -2.62)
-65.10 / 20.77 = -3.134328...  -> -3.13  (raw: -65.103372/20.765877 = -3.135113 -> -3.14, DIFFERS)
```

The `true` and `wrong` fractions happen to round to the same 2dp value
whether computed from the rounded 2dp operands or the raw floats (the
rounding-through-rounded-operands error doesn't cross a display boundary for
those two); the `complete` fraction is the one case where it does. This is
**not** a one-off arithmetic slip — it is `ANALYSIS.md`'s own display
precision propagating into a downstream ratio, the identical failure mode
the Phase-C audit's five slips came from (an already-rounded intermediate
value used as an input to a further computation, rather than the raw float),
just manifesting via rounded operands here rather than a rounded 3dp display
copied wholesale. The mean-based headroom fractions in the same table used
raw values correctly (`-1.196402/-1.288988/-1.682538` all match `ANALYSIS.md
§1.1`'s `-1.196/-1.289/-1.683` exactly at 3sf, confirming the raw-float path
was used there) — only the median-based column took the rounded-operands
shortcut, and only silently broke on one of the three arms.

**Fix applied** (both the manuscript and the upstream source document, since
the source document is a post-hoc statistician's report, not one of the two
frozen `PREREGISTRATION.md` files, and leaving a known-wrong number in it
would let a future re-derivation reintroduce the bug). **Divergence from the
Phase-C audit's posture, disclosed:** that audit was explicitly read-only and
only *flagged* (did not edit) an analogous slip it found in
`phaseC_classical/ANALYSIS.md §6d` (a one-sided vs. two-sided p-value
labelling inconsistency) because that document's number was never quoted in
`main.tex` and editing it therefore risked scope creep beyond the audit
mandate. Here the situation differs in the way that matters: `ANALYSIS.md`'s
wrong value **is** the one directly transcribed into a committed `main.tex`
macro (`\PDhrCompleteMed`), so leaving it uncorrected in `ANALYSIS.md` would
leave the one committed source-of-truth document inconsistent with the
now-corrected paper and would reintroduce the bug on any future
re-derivation from that document. This is a difference in facts (a stale
document copied verbatim into the paper macro vs. a document number never
quoted anywhere else), not a change in read-only discipline — no other
`ANALYSIS.md` content was touched, and the pre-registration files were not
touched at all, matching the Phase-C audit's boundary:

```
paper/main.tex:395                            \newcommand{\PDhrCompleteMed}{-3.13}  ->  {-3.14}
results/phaseD_external/ANALYSIS.md:118       gnn_complete | -1.683 | -3.13         ->  -3.14
```

`\PDhrCompleteMed` is used at exactly one manuscript site
(`main.tex:2995`/`main_jaamas.tex:3034`, the appendix headroom-fraction
sentence) plus its own definition — confirmed by grep, no second stray
hard-coded `-3.13` anywhere in either paper file. `python
paper/_assemble_jaamas.py` was re-run after the fix; `git diff --stat
paper/main_jaamas.tex` shows exactly the expected one-line change, confirming
the fix propagates and nothing else drifted. `pytest -q` re-run after both
edits: **183 passed**, no regression.

**Every other value in the 73-macro sweep — including all of `tab:transfer`'s
mean/median/sd/CI columns, both dagger rows added to `tab:classical`
(`\CRcomplete*`/`\CRwrong*`), the oracle gate, the frozen E1 three-contrast
family, the peak-window exploratory contrast, the E0 descriptive contrast,
and the full instability-characterisation block (`\PDblow*`/`\PDonset*`/
`\PDdrop*`) — matched at its declared precision with zero discrepancies.**
The bare table literals in `tab:transfer-instability`
(median onset steps `140,750`/`98,025`/`83,250`/`62,350`; the `%`-of-budget
bare literal `94` for `mlp`, `56` for `gnn_wrong`; the oracle drop `3.0`) were
separately recomputed from the raw `episodes.csv` files and also match
exactly.

**Not a manuscript-table check (flagged to avoid over-claiming coverage):**
`ANALYSIS.md §4.1`'s median/overall **max-loss** column (e.g. `3.41e6` for
`gnn_complete`) does **not** appear in `tab:transfer-instability`, which has
only four columns (blow-ups, median onset, % of budget, median drop) — no
max-loss column exists in `main.tex`. This column was recomputed for
completeness against `ANALYSIS.md` only (median/overall max loss:
`gnn_complete` 25071.27/3414644.21 → 2.51e4/3.41e6; `gnn_wrong`
692.20/11706.08 → 692/1.17e4; `gnn_true` 72.33/5576.71 → 72.3/5.58e3; `mlp`
12.79/70.20 → 12.8/70.2; `oracle` 8.07/10.17 → 8.07/10.2 — all match
`ANALYSIS.md §4.1` exactly) but is out of scope for a manuscript-macro audit
since no `main.tex` claim depends on it.

## Check 2 — Load-bearing statistics vs. the two frozen pre-registrations

Recomputed the full frozen contrast families for E1
(`results/phaseD_external/PREREGISTRATION.md §2`, exactly the three
`gnn_true`-centred contrasts, Holm m=3, both-tests house rule) and cell R
(`results/phaseC_classical/PREREGISTRATION.md` ADDENDUM, `gnn_true` vs. all 8
rivals, Holm m=8, plus the m=6 invariance check) directly from the raw
per-seed data. **All values match `ANALYSIS.md` and `paper/main.tex` exactly**
(the one number affected by Check 1's fix, the median-based headroom
fraction, is descriptive, not a significance test — it does not change any
p-value, Cohen's d, or verdict):

- Oracle gate (`ANALYSIS.md §0.5`): +30.06, d=2.16, Welch p=1.95e-4, MWU
  p=3.66e-5. **PASS**, both tests significant — governs all E1 branches.
- The frozen E1 family (`§1.2`): `gnn_true`−`mlp` significant in the
  **negative** direction (d=-1.73, Welch p_holm=1.02e-3, MWU p=7.31e-4);
  `gnn_true`−`gnn_wrong` and `gnn_true`−`gnn_complete` both null. Branch-3
  applies by entailment (`§1.3`), reproduced exactly.
- The peak-window exploratory sensitivity (`§1.4`): -8.39, d=-3.29, both
  tests significant, same direction as the frozen result — recomputed
  independently from `episodes.csv` rolling-100 windows and matches to the
  reported digits.
- Cell R (`§3`): the three graph arms are mutually indistinguishable (all
  p_holm > 0.16), none clears `mlp` under both tests (`gnn_true`−`mlp`
  p_holm=0.10 at m=8, 0.052 at the original m=6; MWU=0.078 fails uncorrected
  in both families — the "≈mlp" conclusion is invariant to family size, as
  claimed at `main.tex:2053-2057`). All four DCG contrasts remain significant
  in both families. The m=6 recomputation independently reproduces cell C's
  already-committed d=1.03/p_holm=0.052/MWU=0.078 (`main.tex:2020-2021`) to
  the printed precision.
- E0 (`§2`, non-confirmatory): -12.62, d=-1.05, Welch p=0.11, MWU p=0.24 —
  tie, both non-significant, matches.
- Instability characterisation (`§4`, exploratory/descriptive): blow-up
  counts, onset steps/percentages, and peak→final drops recomputed from raw
  `episodes.csv` grad-norm/return series exactly reproduce every number in
  `ANALYSIS.md §4.1/4.3/4.4` and every corresponding `\PD*` macro (Check 1).
  The per-seed `gnn_true` onset list (return-based, 11/12 turning over before
  anneal completion) reproduces the reported count; individual per-episode
  onset step indices differ by one 25-step tick from `ANALYSIS.md`'s own
  per-seed list (a boundary-condition artifact of `>` vs. `>=` at the
  rolling-window edge, e.g. 95,550 recomputed vs. 95,525 reported) — this
  per-seed list is not quoted in `main.tex` at that granularity (only the
  aggregate "11 of 12" is), so it does not affect any manuscript claim.
- Out-of-family, must-not-claim check: `gnn_complete`−`mlp` on the relay
  recomputes to adv=+2.868715, d=+1.058008, Welch p=0.024581, MWU p=0.006099
  — matches `ANALYSIS.md §5.3`'s "flagged, NOT claimable" disclosure exactly,
  and (Check 3) does not appear anywhere in `main.tex`/`main_jaamas.tex`.

**Decision-rule outcomes** (Branch 3 governs E1 by entailment; Cell-R Pattern
1 obtains; E0's frozen honesty-anchor sentence is withdrawn because its
validity-gate precondition fails) were re-derived from these recomputed
numbers and match `ANALYSIS.md §1.3/§3.3/§2` and the manuscript's prose
exactly.

## Check 3 — prohibited-claims grep (`ANALYSIS.md §7` NOT-claimable list)

Grepped both `paper/main.tex` and `paper/main_jaamas.tex` for every item on
the NOT-claimable list:

- **E0 frozen honesty-anchor sentence** ("the graph is redundant when the env
  supplies its own channel") — **0 hits** in either file. The appendix
  (`app:transfer`) instead reports the withdrawal explicitly.
- **"Graph structure does not help on MPE"** — 1 hit each file
  (`main.tex:2280`, `main_jaamas.tex:2299`), and in both cases it is the
  *negated* form: "We therefore do **not** read this cell as \`\`graph
  structure does not help on MPE''". Correctly quoted only to disclaim it.
- **"Unmodified benchmark"** — 2 hits each file, both negated ("we ... never
  describe this cell as a replication on an unmodified benchmark"; "so it is
  not an unmodified benchmark either"). Correctly disclaimed, not asserted.
- **Branch-1/2a/2b claimable sentences** — **0 hits**. Not available since
  Branch 3 obtained.
- **`gnn_complete`−`mlp` on the relay as a finding** — grep for `2.87` /
  `2.86` (any precision the raw +2.868715 could plausibly be printed at) in
  both `main.tex` and `main_jaamas.tex`: **0 hits in either file.**
  **Correctly excluded from the manuscript** (matches `ANALYSIS.md §5.3`'s
  explicit not-claimable disclosure, Check 2).
- **Density-as-mechanism** — 0 hits asserted as a *cause*; the one mention
  (`main.tex:3007`) explicitly disclaims it as "exploratory and unablated."
- **Residual+LayerNorm as an established cause** — 0 hits asserting this as
  established; both mentions (`main.tex:2284-2288`, `:2600-2603`) explicitly
  label it "a candidate we could not test" / "a hypothesis rather than a
  finding."
- **Any E2 mention** — **0 hits** of `E2`, `k{=}5`, or a K=5 escalation in
  either paper file.
- **Pattern-3 drift / cell-R re-examination trigger** — **0 hits**.
- **Directional reading of `gnn_true`−`gnn_complete` (d=0.60, underpowered)**
  — `main.tex:2247` explicitly states this contrast is "underpowered rather
  than a confident tie, and we read no direction into it."

**C1/C2 pairing check.** The `\PDtmGap` macro (the -35.96 Branch-3 number) is
used at exactly one manuscript site (`main.tex:2241`, `main_jaamas.tex:2260`,
plus its own `\newcommand` definition — confirmed by grep, no stray
hard-coded `-35.96` anywhere else in either file). That use site sits in the
same subsection as, and four paragraphs before, the instability disclosure
(`main.tex:2266-2288`, "The failure is not a converged null..."); the
abstract-adjacent discussion paragraph (`main.tex:2722-2727`) that references
the same result via `\PDoracleGap`/`\PDoracleD` states the divergence in the
same sentence. **C1 never appears without C2's substance nearby. PASS — no
prohibited claims found in either file.**

## Check 4 — pre-existing headline numbers, drift check (5 spot-checks)

| number | macro | manuscript | recomputed from source | source |
|---|---|---:|---:|---|
| Cell-B `gnn_true` frozen anchor (mean/median/sd/CI) | `\SCkTrue` + table row | 94.96 / 96.28 / 5.05 / [91.75,98.18] | 94.963854 / 96.278125 / 5.054608 / [91.752309,98.175399] | `results/phaseC_150k/manifest_pair_N6_ego_attention.csv` |
| d≈5→12 pair (attention-over-complete, `gat_complete`/`dgn_complete` vs. `gnn_true`) | `\SCtrueAttCompD`/`\SCkTrueAttCompD` | 5.5 → 12 | 30k: d=5.41 (dgn), 5.52 (gat); 150k: d=11.02 (dgn), 12.14 (gat) | `results/phaseC/…`, `results/phaseC_150k/…` |
| Cell C (Cell R merged) `gnn_true`/`mlp`/`dcg_jointobs_wrong` | table literals | 64.00/61.11/56.44 (means) | 64.004444/61.114236/56.444896 | `manifest_cellC_N6_ego_150000steps.csv` |
| Phase-8 scaling parity, N=12 | `\PSgapNtwelve` | -153.4 | 92.075167 − 245.459167 = -153.384 | `results/phase11_scale/summary_aggregate.csv` |
| TokenMatch `gnn_true` | `\TMtrue`/`\TMtrueLo`/`\TMtrueHi` | 57.59 [57.55,57.63] | 57.592778, CI [57.551501,57.634054] | `results/phaseD/token_N6/gnn_true__*` |

All five match exactly. `git log` confirms none of these five macros'
definitions or source files were touched by the Phase-D integration chain
(`85dea03`..`01b3123`). **PASS — no drift.**

## Check 5 — abstract word count and `main_jaamas.tex` regeneration

**Correction to this audit's own first-pass finding**: the abstract *did*
change under the Phase-D integration — `git diff b5d76f9 01b3123 --
paper/main.tex` (restricted to the `\begin{abstract}...\end{abstract}` span)
shows the closing sentence rewritten from "replication on standard benchmarks
with private per-agent goals is future work" to the Branch-3 rescoping
sentence, plus several small trims elsewhere in the same paragraph (dropping
"a gridworld" before `\coordgrid`, "even at" before `$5\times$", "over 12
seeds" → "at $n{=}12$", "removes this pathology, recovering parity" →
"recovers parity", "reliably estimable" → "estimable"). This is exactly the
site `ANALYSIS.md §6.1(b)` flags as one of four the Branch-3 rewrite had to
touch. (An earlier draft of this report guessed at this without checking the
diff; corrected here per review.)

Two word-count conventions, reported the same way the Phase-C audit reported
them (that audit's governing figure was 246, with 245 as the assembly
script's own crude corroboration — both ≤ 250):

- **Governing convention** (macro/math-stripped: strip `$...$` math and bare
  `\command` tokens entirely, unwrap `\emph{...}`/`\textbf{...}` to their
  inner text): **245 words** for the current abstract (`paper/main.tex:434-
  461`). **PASS**, 5-word margin under a 250-word cap.
- **Assembly script's own crude counter** (`paper/_assemble_jaamas.py`'s
  built-in diagnostic, a different regex that does **not** strip `$...$` math
  — it counts alphanumeric tokens including bare numerals/Cohen's-d values
  embedded in math mode, e.g. the `5`/`12` inside `$d{\approx}5\!\to\!12$`, as
  separate words): **~249 words**. Still ≤ 250, but only a **1-word margin**
  — noticeably tighter than the Phase-C audit's 245-vs-246 report, because
  this abstract's added Phase-D sentence and its surrounding math contain
  more embedded numerals than before. **PASS but flagged**: if the abstract
  is edited again, the crude counter should be re-checked, since it is
  already at the edge of a 250-word cap (if one still binds — note the venue
  target has since moved to JAAMAS per the user's own tracked decision;
  confirm JAAMAS's own abstract limit, if any, before treating 250 as
  binding).

`python paper/_assemble_jaamas.py` was re-run twice: once during the initial
pass (before Check 1's fix, to confirm the pre-existing macro set carried
over byte-stable — it did), and again after the `\PDhrCompleteMed` fix. After
the fix, `git diff --stat paper/main_jaamas.tex` shows exactly the expected
one-line change (`-3.13` → `-3.14`) and nothing else — confirming the
regeneration is deterministic and the fix propagates cleanly. **PASS.**

## Check 6 — test suite, lint, commit-chain integrity

- `pytest -q` (repo `.venv`): **183 passed**, exit code 0, re-confirmed after
  Check 1's fix (no regression). (183 > the 160 recorded at the Phase-C
  pre-reg time, consistent with the Phase-D test additions —
  `tests/test_mpe_reference_pairs.py` etc. — landing since.)
- `ruff check src tests`: **clean** (all checks passed).
- `ruff check src tests scripts`: **NOT clean — 22 pre-existing errors**, all
  in the same 5 files the Phase-C audit already flagged
  (`scripts/build_paper_inserts.py`, `scripts/fig_overview.py`,
  `scripts/phase11_scale_figure.py`, `scripts/phase1_analysis.py`,
  `scripts/phaseC_figures.py`) — confirmed by `git log -1` on each that none
  was touched by any Phase-D commit; last edits range `188eb31`..`7cf4a77`,
  all pre-dating Phase D. The Phase-D scripts
  (`scripts/phaseD_external_sweep.py`, `phaseD_reference_points.py`,
  `phaseD_greedy_reference.py`, `phaseD_150k_resmoke.py`) are individually
  **lint-clean**. Same pre-existing lint-debt flag as the Phase-C audit, not
  a regression from this integration.
- Commit-chain order: `git merge-base --is-ancestor` confirms
  `568e759 → b00e26b → b5d76f9 → 01b3123`, strictly increasing in both
  wall-clock (`00:46:38 → 02:07:37 → 03:08:10 → 04:06:17`, all 2026-07-31)
  and ancestry — pre-registration, the amendment/gate-verdict freeze, the
  confirmatory data commit, and the manuscript-integration commit are each
  strict descendants of the last, with **no forking path via post-hoc pre-reg
  editing possible**.
- Working tree after this audit's fix: `paper/main.tex`,
  `paper/main_jaamas.tex`, `results/phaseD_external/ANALYSIS.md` modified
  (the single `-3.13`→`-3.14` correction and its JAAMAS regeneration);
  `results/phaseD_external/FINAL_NUMBER_AUDIT.md` newly added. No other file
  touched.

## Summary verdict

| check | verdict | notes |
|---|---|---|
| 1. `\PD*`/`\CR*` macros + `tab:transfer`/`tab:transfer-instability`/dagger rows/appendix | **FAIL (minor), fixed** | 1 discrepancy in 73 programmatically-checked value macros: `\PDhrCompleteMed` -3.13 → -3.14 (root cause: an arithmetic slip in `ANALYSIS.md §1.1`'s own table). Fixed in both `main.tex` and `ANALYSIS.md`; `main_jaamas.tex` regenerated; 0 further discrepancies after the fix |
| 2. Load-bearing statistics (both pre-registrations' frozen families) | **PASS** | Oracle gate, E1 three-contrast family, peak-window exploratory, Cell-R m=8/m=6 families, E0 descriptive contrast, and the full instability characterisation all reproduce exactly; unaffected by Check 1's fix (a descriptive fraction, not a test statistic) |
| 3. Prohibited-claims grep (`ANALYSIS.md §7`) | **PASS** | 0 unhedged hits for any of the 10 NOT-claimable items in either `main.tex` or `main_jaamas.tex`; C1 (`\PDtmGap`) never appears without C2's instability disclosure nearby; `gnn_complete`−`mlp` relay number confirmed absent by grep |
| 4. Pre-existing headline numbers (drift) | **PASS** | 5/5 match source CSVs exactly |
| 5. Abstract word count / `main_jaamas.tex` regeneration | **PASS, margin tighter than reported at first pass** | Governing (macro/math-stripped) count 245/250; assembly script's own crude counter ~249/250 (only 1-word margin, vs. the wider gap in the Phase-C-audited abstract) — worth re-checking if the abstract is edited again; abstract confirmed (via diff, not guessed) to have changed under this integration; `main_jaamas.tex` regenerates deterministically |
| 6. Test suite / lint / commit chain | **PASS with a flag** | 183/183 tests green after the fix; `ruff check src tests` clean; `ruff check src tests scripts` has the same 22 pre-existing (non-regressed) errors already on record; commit order verified correct |

**Overall: 1 minor FAIL, found and fixed during this audit (a single
transcription/arithmetic slip, -3.13 → -3.14, in a descriptive headroom
fraction that does not touch any significance verdict, Cohen's d, or
claimable statement); 5 PASS; 1 PASS-with-flag (pre-existing lint debt, not
introduced here). Nothing found blocks submission, provided the one-line fix
in this audit's working tree (`paper/main.tex`, `paper/main_jaamas.tex`,
`results/phaseD_external/ANALYSIS.md`) is committed.**

### Fix applied by this audit (already in the working tree, not yet committed)

```
paper/main.tex:395                       \newcommand{\PDhrCompleteMed}{-3.13}  ->  {-3.14}
paper/main_jaamas.tex:403                (regenerated via paper/_assemble_jaamas.py, same change)
results/phaseD_external/ANALYSIS.md:118  gnn_complete | -1.683 | -3.13          ->  -3.14
```

No other edit is recommended.
