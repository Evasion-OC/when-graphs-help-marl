#!/usr/bin/env python3
r"""Assemble paper/main_jaamas.tex (Springer sn-jnl / JAAMAS port) from
paper/main.tex (the TMLR-formatted scientific master) and
paper/_jaamas_preamble.tex (the hand-written Springer front/back-matter
template).

Architecture: assembly, not fork. main.tex remains the single place where
scientific content (numbers, claims, tables, figures, citations) lives.
This script performs only mechanical, structural transforms:

  1. Lift every `\newcommand` line out of main.tex's preamble (notation
     macros + all numbered-result macros, e.g. \PEqmixNfour). This makes
     the port pick up new results automatically: when the nightly sweep
     adds new \newcommand lines to main.tex, the next run of this script
     carries them into main_jaamas.tex with zero manual edits.
  2. Extract the abstract body (\begin{abstract}...\end{abstract}).
  3. Extract the main body (everything between \end{abstract} and
     \bibliographystyle{tmlr}), i.e. Introduction .. Broader Impact
     Statement.
  4. Extract the appendix body (everything between \appendix and
     \end{document}).
  5. Apply a small, explicit table of de-anonymization / mechanical text
     substitutions to the extracted text (never to main.tex itself).
  6. Splice all of the above into the placeholders of
     _jaamas_preamble.tex and write paper/main_jaamas.tex.

Re-run with:  python paper/_assemble_jaamas.py
Then build:   pdflatex -interaction=nonstopmode -halt-on-error main_jaamas
              bibtex main_jaamas
              pdflatex -interaction=nonstopmode -halt-on-error main_jaamas
              pdflatex -interaction=nonstopmode -halt-on-error main_jaamas
(run from the paper/ directory, so figures/ and results/*.tex resolve).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
MAIN_TEX = PAPER_DIR / "main.tex"
PREAMBLE_TEMPLATE = PAPER_DIR / "_jaamas_preamble.tex"
OUTPUT_TEX = PAPER_DIR / "main_jaamas.tex"

# ---------------------------------------------------------------------------
# De-anonymization / mechanical substitutions applied to text extracted from
# main.tex. main.tex itself is never modified -- these are string
# replacements performed only on the copy assembled into main_jaamas.tex.
# Each entry is (must-match-exactly-once, old, new, description).
# ---------------------------------------------------------------------------
SUBSTITUTIONS: list[tuple[str, str, str]] = [
    (
        "are provided as anonymized supplementary material.",
        "are available at \\url{https://github.com/Evasion-OC/when-graphs-help-marl}.",
        "app:repro: anonymized-supplementary-material sentence -> GitHub link "
        "(de-anonymization sweep, spec item 5).",
    ),
    (
        "and the raw \\textsc{EPyMARL} logs are not part of the anonymized code artifact.",
        "and the raw \\textsc{EPyMARL} logs are not part of the code artifact.",
        "app:standard (LBF/SMAC): drop 'anonymized' from the code-artifact "
        "disclaimer -- the underlying fact (raw EPyMARL logs are not shipped) "
        "still holds post-de-anonymization (de-anonymization sweep, spec item 5).",
    ),
    (
        "will be published with the de-anonymized repository.",
        "are therefore recoverable in full even where the episode-level files "
        "themselves are not shipped.",
        "app:repro: the anonymized build promises per-episode logs will appear "
        "on de-anonymization. Once the repository is public that promise is "
        "checkable, and the episode-level CSVs for phaseC / phaseC_classical / "
        "phaseD_external are deliberately gitignored (~364 MB). Replace the "
        "promise with the guarantee that actually holds -- exact regeneration "
        "from the shipped seeds and configs (de-anonymization sweep, spec item 5).",
    ),
]

# ---------------------------------------------------------------------------
# Table layout fixes: purely mechanical (font size / float environment /
# column padding), never touching a number, word, or citation. Two distinct
# problems, both consequences of sn-jnl's page geometry being narrower than
# TMLR's (single-column text width 31pc =~ 372pt vs TMLR's 6.5in = 468pt):
#
# (a) \resizebox/\adjustbox/\scalebox wrapping a booktabs-ruled tabular is a
#     confirmed FATAL error under sn-jnl.cls ("Missing \endgroup inserted"
#     on \end{tabular}): both the class's own \toprule/\midrule/\bottomrule
#     and real booktabs' use a moving-argument-safety brace trick
#     (\ifnum0=`}\fi) that is incompatible with graphicx's box-scaling
#     transforms under this class (isolated, reproduced, confirmed --
#     every scaling variant fails identically; plain tabular does not).
#     main.tex's 3 resizebox-wrapped tables must lose the box-scaling
#     wrapper and shrink by font size instead.
# (b) Several tables that fit at \small under TMLR's wider column overflow
#     sn-jnl's narrower one. Fixed by a uniform \small -> \footnotesize
#     downgrade (font size only) applied to every remaining table that
#     still has plain \small once (a) is done, plus \sidewaystable (a
#     class-supported, manual-documented environment: sec. 7.3) for the
#     two structurally widest tables (many columns / long unwrapped
#     multicolumn text), which trades \linewidth for the much larger
#     \textheight.
#
# Fixes are applied IN LIST ORDER (each entry mutates the running text before
# the next is matched), so table-specific fixes that consume a table's own
# "\small" token are listed BEFORE the generic \small -> \footnotesize sweep
# that would otherwise also match them.
# Each entry: (literal anchor text, replacement, expected total match count
# across all regions, description).
TABLE_LAYOUT_FIXES: list[tuple[str, str, int, str]] = [
    # --- tab:structure (body, Sec. headline): resizebox -> bare footnotesize
    # group, AND \sidewaystable. Its two contrast rows carry a long
    # descriptive parenthetical hard-coded to "{\footnotesize(...)}" in
    # main.tex (an absolute size, so it does not shrink further no matter
    # what outer size this fix picks -- confirmed by measurement: going
    # \footnotesize -> \scriptsize on the surrounding table INCREASED the
    # reported overfull, 86pt -> 129.71pt, because the rest of the table
    # shrank around that fixed-size span rather than with it). That
    # unshrinkable span sets a floor of ~458pt natural width, comfortably
    # under sidewaystable's ~550pt \textheight budget but over upright
    # \linewidth (~372pt) at any font size -- so rotation, not further
    # shrinking, is the correct fix here.
    (
        "\\begin{table}[t]\n"
        "  \\centering\n"
        "  \\small\n"
        "  \\caption{\\textbf{The structure boundary}",
        "\\begin{sidewaystable}[t]\n"
        "  \\centering\n"
        "  \\footnotesize\n"
        "  \\caption{\\textbf{The structure boundary}",
        1,
        "tab:structure: table -> sidewaystable (open), \\small -> \\footnotesize",
    ),
    (
        "\\label{tab:structure}\n"
        "  \\setlength{\\tabcolsep}{4pt}%\n"
        "  \\resizebox{\\linewidth}{!}{%",
        "\\label{tab:structure}\n"
        "  \\setlength{\\tabcolsep}{4pt}%\n"
        "  {",
        1,
        "tab:structure: resizebox -> bare group "
        "(booktabs/scalebox incompatibility workaround; size set above via \\footnotesize)",
    ),
    (
        "\\gtrue$-$\\mlpc\\ {\\footnotesize(a-priori: no channel)} & \\multicolumn{2}{l}{$\\SCtmGap$,\\; $d=\\SCtmD$,\\; $p_{\\text{Holm}}\\SCpHolm$,\\; MWU $<10^{-4}$} \\\\\n"
        "    \\bottomrule\n"
        "  \\end{tabular}}%\n"
        "\\end{table}",
        "\\gtrue$-$\\mlpc\\ {\\footnotesize(a-priori: no channel)} & \\multicolumn{2}{l}{$\\SCtmGap$,\\; $d=\\SCtmD$,\\; $p_{\\text{Holm}}\\SCpHolm$,\\; MWU $<10^{-4}$} \\\\\n"
        "    \\bottomrule\n"
        "  \\end{tabular}}%\n"
        "\\end{sidewaystable}",
        1,
        "tab:structure: table -> sidewaystable (close)",
    ),
    # --- tab:budget (appendix): resizebox -> bare footnotesize group, AND
    # \sidewaystable (rotated: \textheight ~=550pt available vs ~372pt
    # upright) -- this table's long "Reason for deviation" prose column
    # does not fit even at footnotesize in the upright column.
    (
        "\\begin{table}[!ht]\n"
        "  \\centering\n"
        "  \\caption{Compute budget per experiment:",
        "\\begin{sidewaystable}[!ht]\n"
        "  \\centering\n"
        "  \\caption{Compute budget per experiment:",
        1,
        "tab:budget: table -> sidewaystable (open)",
    ),
    (
        "\\label{tab:budget}\n"
        "  \\small\n"
        "  \\resizebox{\\linewidth}{!}{%",
        "\\label{tab:budget}\n"
        "  \\footnotesize\n"
        "  {",
        1,
        "tab:budget: resizebox -> bare \\footnotesize group "
        "(booktabs/scalebox incompatibility workaround)",
    ),
    (
        "D (\\tokenmatch\\ replication) & --- (post-hoc) & $3\\cdot 10^4 \\times 12 \\times 4$ & out-of-harness replication. \\\\\n"
        "    \\bottomrule\n"
        "  \\end{tabular}}%\n"
        "\\end{table}",
        "D (\\tokenmatch\\ replication) & --- (post-hoc) & $3\\cdot 10^4 \\times 12 \\times 4$ & out-of-harness replication. \\\\\n"
        "    \\bottomrule\n"
        "  \\end{tabular}}%\n"
        "\\end{sidewaystable}",
        1,
        "tab:budget: table -> sidewaystable (close)",
    ),
    # --- tab:gat (appendix, 9 columns): resizebox -> bare group,
    # \tiny (scriptsize alone was still 294.87pt overfull), AND
    # \sidewaystable -- the widest table in the document.
    (
        "\\begin{table}[!ht]\n"
        "  \\centering\n"
        "  \\small\n"
        "  \\caption{Attention variants vs the QMIX family",
        "\\begin{sidewaystable}[!ht]\n"
        "  \\centering\n"
        "  \\tiny\n"
        "  \\caption{Attention variants vs the QMIX family",
        1,
        "tab:gat: table -> sidewaystable (open), \\small -> \\tiny "
        "(9-column table, needs both the rotation and the smallest standard size)",
    ),
    (
        "\\label{tab:gat}\n"
        "  \\resizebox{\\textwidth}{!}{%",
        "\\label{tab:gat}\n"
        "  {",
        1,
        "tab:gat: resizebox -> bare group "
        "(booktabs/scalebox incompatibility workaround; size set above via \\tiny)",
    ),
    (
        "        & $\\PGgatMpeNsix$ & $\\PGdgnMpeNsix$ \\\\\n"
        "    \\bottomrule\n"
        "  \\end{tabular}}\n"
        "\\end{table}",
        "        & $\\PGgatMpeNsix$ & $\\PGdgnMpeNsix$ \\\\\n"
        "    \\bottomrule\n"
        "  \\end{tabular}}\n"
        "\\end{sidewaystable}",
        1,
        "tab:gat: table -> sidewaystable (close)",
    ),
    # --- Generic sweep: every remaining "\centering\n  \small" table
    # (tab:mpe, tab:lbf, tab:repaired, tab:scaling, and any others that did
    # not need the table-specific handling above) downgrades to
    # \footnotesize -- a uniform, content-neutral font-size reduction to
    # fit sn-jnl's narrower single-column width. Applied last so it only
    # matches occurrences the fixes above have not already consumed.
    (
        "\\centering\n"
        "  \\small",
        "\\centering\n"
        "  \\footnotesize",
        14,  # 16 tables have "\centering\n  \small" in main.tex; tab:structure's
             # and tab:gat's are each consumed by their own table-specific fix
             # above (both anchors span "\small" itself), leaving 14.
        "Generic table-width fix: \\small -> \\footnotesize on all remaining "
        "data tables (sn-jnl single column is narrower than TMLR's)",
    ),
]


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def extract_between(text: str, start_pat: str, end_pat: str, *, label: str) -> tuple[str, int, int]:
    """Return (content strictly between start_pat and end_pat, start_idx, end_idx)."""
    m_start = re.search(start_pat, text)
    if not m_start:
        die(f"could not find start marker for {label}: {start_pat!r}")
    m_end = re.search(end_pat, text[m_start.end():])
    if not m_end:
        die(f"could not find end marker for {label}: {end_pat!r}")
    content = text[m_start.end(): m_start.end() + m_end.start()]
    return content, m_start.end(), m_start.end() + m_end.start()


def extract_notation_macros(text: str) -> str:
    """Every line in main.tex whose stripped form starts with \\newcommand.

    This is the whole preamble-macro block (notation shortcuts like
    \\coordgrid, \\Qtot, plus every numbered-result macro such as
    \\PEqmixNfour / \\SCtrue / \\TMtrue etc.). \\def lines (\\month, \\year,
    \\openreview -- TMLR camera-ready metadata, confirmed unused in the
    body) and \\providecommand{\\Description} (re-added by the template)
    are intentionally excluded.
    """
    lines = text.split("\n")
    macro_lines = [l for l in lines if l.strip().startswith("\\newcommand")]
    if not macro_lines:
        die("found zero \\newcommand lines in main.tex -- extraction regex is broken")
    return "\n".join(l.rstrip("\r") for l in macro_lines)


def _pattern_for(old: str) -> re.Pattern:
    # main.tex hard-wraps prose at ~80 columns, so a literal multi-word
    # match can straddle a line break in the source. Build a regex that
    # tolerates arbitrary whitespace (incl. newlines) wherever the
    # human-written pattern has a space, so re-wrapping main.tex does not
    # silently break the substitution.
    return re.compile(r"\s+".join(re.escape(w) for w in old.split()))


def apply_substitutions(regions: dict[str, str]) -> dict[str, str]:
    """Apply SUBSTITUTIONS across a named set of extracted-text regions.

    Each substitution must match exactly once across ALL regions combined
    (not necessarily within any single region) -- e.g. the anonymized-
    supplementary-material sentence lives only in the appendix, not the
    abstract or body.
    """
    out = dict(regions)
    for old, new, desc in SUBSTITUTIONS:
        pat = _pattern_for(old)
        hits = {name: len(pat.findall(text)) for name, text in out.items()}
        total = sum(hits.values())
        if total == 0:
            die(f"substitution pattern not found in any region (was found at assembly-script "
                f"authoring time; main.tex prose changed): {desc}")
        if total > 1:
            die(f"substitution pattern matched {total} times across {hits} (expected 1): {desc}")
        for name, text in out.items():
            if hits[name]:
                out[name] = pat.sub(lambda _m: new, text, count=1)
                print(f"OK: applied substitution in {name}: {desc}")
    return out


def apply_table_layout_fixes(regions: dict[str, str]) -> dict[str, str]:
    """Apply TABLE_LAYOUT_FIXES across regions; each must match its declared
    expected_count exactly, summed across all regions (unlike
    apply_substitutions, more than one match is expected/allowed here).
    """
    out = dict(regions)
    for old, new, expected_count, desc in TABLE_LAYOUT_FIXES:
        pat = _pattern_for(old)
        hits = {name: len(pat.findall(text)) for name, text in out.items()}
        total = sum(hits.values())
        if total != expected_count:
            die(f"resizebox fix matched {total} times across {hits} (expected "
                f"{expected_count}): {desc}")
        for name, text in out.items():
            if hits[name]:
                out[name] = pat.sub(lambda _m: new, text)
                print(f"OK: applied resizebox fix ({hits[name]}x) in {name}: {desc}")
    return out


def main() -> None:
    if not MAIN_TEX.exists():
        die(f"main.tex not found at {MAIN_TEX}")
    if not PREAMBLE_TEMPLATE.exists():
        die(f"template not found at {PREAMBLE_TEMPLATE}")

    # Normalize line endings: main.tex is CRLF; keep the assembled output
    # LF-only for consistency and so multi-line substitution patterns don't
    # have to account for embedded \r.
    main_text = MAIN_TEX.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    template_text = PREAMBLE_TEMPLATE.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")

    # --- 1. Notation / result macros -------------------------------------
    macros = extract_notation_macros(main_text)

    # --- 2. Abstract -------------------------------------------------------
    abstract_text, _, _ = extract_between(
        main_text, r"\\begin\{abstract\}\s*\n?", r"\\end\{abstract\}", label="abstract"
    )
    abstract_text = abstract_text.strip("\n")

    # --- 3. Main body: \end{abstract} .. \bibliographystyle{tmlr} ---------
    body_text, _, _ = extract_between(
        main_text, r"\\end\{abstract\}\s*\n?", r"\\bibliographystyle\{tmlr\}",
        label="main body",
    )
    body_text = body_text.strip("\n")

    # --- 4. Appendix: \appendix .. \end{document} --------------------------
    appendix_text, _, _ = extract_between(
        main_text, r"\\appendix\s*\n?", r"\\end\{document\}", label="appendix"
    )
    appendix_text = appendix_text.strip("\n")

    # --- 5. De-anonymization / mechanical substitutions ---------------------
    regions = apply_substitutions({
        "abstract": abstract_text,
        "main body": body_text,
        "appendix": appendix_text,
    })
    regions = apply_table_layout_fixes(regions)
    abstract_text, body_text, appendix_text = regions["abstract"], regions["main body"], regions["appendix"]

    # --- 6. Splice into the template ---------------------------------------
    out = template_text
    replacements = {
        "__NOTATION_MACROS__": macros,
        "__ABSTRACT_TEXT__": abstract_text,
        "__BODY__": body_text,
        "__APPENDICES__": appendix_text,
    }
    for placeholder, value in replacements.items():
        count = out.count(placeholder)
        if count == 0:
            die(f"placeholder {placeholder} not found in template {PREAMBLE_TEMPLATE}")
        if count > 1:
            die(f"placeholder {placeholder} appears {count} times in template "
                f"{PREAMBLE_TEMPLATE} (must appear exactly once -- check comments "
                f"for a stray literal copy of the token)")
        out = out.replace(placeholder, value)

    OUTPUT_TEX.write_text(out, encoding="utf-8", newline="\n")

    n_words_abs = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*",
                                  re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ",
                                         re.sub(r"[{}~]", " ", abstract_text))))
    print(f"\nWrote {OUTPUT_TEX}")
    print(f"  notation/result macros carried over: {len(macros.splitlines())}")
    print(f"  abstract: {len(abstract_text)} chars, ~{n_words_abs} words (crude count)")
    print(f"  main body: {len(body_text)} chars")
    print(f"  appendix: {len(appendix_text)} chars")


if __name__ == "__main__":
    main()
