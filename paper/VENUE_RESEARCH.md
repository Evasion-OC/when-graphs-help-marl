# Venue decision (current)

**Target: JAAMAS (Autonomous Agents and Multi-Agent Systems, Springer).**
Decided 2026-07-30 after the paper's final form emerged: a positive-headline,
two-endpoint boundary study with a controlled-attribution method, a
cross-mechanism replication (GNN + deep coordination graph), and pre-registered
honest negatives.

## Why JAAMAS

- The coordination-graph lineage (Guestrin; Kok & Vlassis; DCG) originates in this
  community, and the paper now engages and tests it directly — the manuscript's
  framing is AAMAS-native.
- Single-blind; no page limit for regular papers; free via the subscription route
  (no APC unless open access is chosen).
- IFAAMAS partnership: acceptance makes the paper eligible for AAMAS presentation
  plus a 2-page extended abstract in the AAMAS proceedings.
- Requirements handled: Springer sn-jnl format (`main_jaamas.tex`, assembled from
  `main.tex` by `_assemble_jaamas.py`), abstract ≤250 words, 4–6 keywords,
  Statements & Declarations (incl. AI-tools disclosure), 1–2pp Information Sheet
  (`JAAMAS_INFO_SHEET.md`).

## History (venues considered and dropped)

- **TMLR** — first version desk-rejected; a later, submission-ready resubmission
  package was prepared but the authors elected not to resubmit (2026-07-30).
- **JAIR** — desk-rejected 2026-07-02 without review (wanted theory); dropped.
- **JMLR** — poor fit (expects new algorithms/theory).
- **Springer MLJ** — strong fallback if JAAMAS rejects: scope explicitly includes
  evaluation methodologies; ~2 months to first decision.
- **ACM TAAS** — credible third option; self-adaptive-systems center of gravity.
- **IEEE TNNLS / Neural Networks / IEEE TAI** — ruled out (novelty/method-driven;
  TAI additionally has ~10pp + mandatory overlength charges).
- **RLC / AAMAS (conferences), NeurIPS/ICML workshops** — timing-blocked or
  non-archival at decision time; the JAAMAS→AAMAS-presentation route supersedes.

Full verified analysis (policies, fees, turnaround, sources) recorded 2026-07-30
in the working notes; fallback order if rejected: MLJ, then TAAS.
