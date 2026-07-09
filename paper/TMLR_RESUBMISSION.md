# TMLR Resubmission — Changes Statement & Submission Checklist

TMLR policy: a rejected submission may be resubmitted as a **new** submission with
(a) a link to the previously rejected submission and (b) a description of the changes
made since. This file holds the paste-ready changes statement and the pre-flight
checklist. Only two user-side placeholders remain (the OpenReview link, and the
supplementary-zip upload); everything in the manuscript is done.

---

## Description of changes since the rejected submission

*(Paste into the OpenReview "changes since previous submission" field, together with
the link to the rejected submission: `<OPENREVIEW LINK TO REJECTED SUBMISSION — FILL IN>`)*

The previously rejected version was a pure negative-results paper. The present
submission is a substantially different, two-endpoint study; the headline
contribution did not exist in the rejected version.

1. **New pre-registered positive result (the headline change).** We add a
   pre-registered pair-routing task on which a task-matched coordination graph
   decisively outperforms *all-to-all communication* — a rival that carries the
   partner's private information but must learn to select it — at byte-identical
   parameter counts and observations (Cohen's d ≈ 5, complete seed-level separation,
   Welch's t with Holm correction and Mann–Whitney, p < 1e-4; significant at every
   team size N = 4–10). The paper is now a boundary characterization ("when does
   graph structure help"), not a null result.

2. **The negative endpoint is now identified, mechanistically explained, and
   replicated out of harness.** A parameter-matched no-graph control (MLP-QMIX)
   raises the decisive comparison from a confounded 3-seed pilot to an identified,
   capacity-controlled result at 10–12 seeds and 5× the training budget, replicated
   on PettingZoo MPE and Level-Based Foraging. A new control (Appendix on the
   stabilised GCN) shows the dramatic full-information *harm* is an optimisation
   pathology of the standard unstabilised operator — residual+LayerNorm recovers the
   graph to parity with the no-graph control — so the robust cross-environment claim
   is "no benefit," with "active harm" scoped to the unstabilised operator. The
   positive endpoint also replicates in a second, non-spatial environment
   (TokenMatch: referential matching with no grid or movement, ~96% of task maximum).

3. **Attention controls close the "under-expressive GNN" alternative.** Matched GAT
   and multi-head DGN controls (with residual + LayerNorm) train well *given* the
   task-matched graph but collapse to the random floor over all-to-all — at degree-1
   and degree-2 — so learned attention does not substitute for correct structure.

4. **Honest scoping throughout.** Hypotheses H-N (advantage grows with N) and H3
   (depth ≈ diameter) are reported as *not supported*; the pre-registered vs.
   post-hoc cells are labelled explicitly; the evaluation-window and MPE-budget
   deviations are disclosed in-text; the failed pre-registered search (Stage A/B
   goal-broadcast) that preceded the positive task is disclosed; the positive result
   is scoped explicitly as a constructed-task boundary, with standard-benchmark
   replication stated as future work.

5. **Anonymization and code release corrected.** The submission is fully anonymized
   (no author-identifying repository links, PDF metadata, embedded-figure timestamps,
   or acknowledgments); code, configs, and result data are provided anonymously as
   supplementary material.

---

## Pre-flight checklist (before clicking submit on OpenReview)

- [x] `main.pdf` built from TMLR stylefile in **submission (anonymous) mode** — page 1
      reads "Under review as submission to TMLR / Anonymous authors / Paper under
      double-blind review". `\usepackage{tmlr}` with no option (verified line 14).
- [x] Zero hits in PDF text for: Jabbary, Ghanavati, Kasra, Evasion-OC, alijabbary,
      gre.ac.uk, Greenwich, Urmia, ORCID `0000-0003-0573-6909`, icloud, research@,
      github.com, 4open.science (verified on the final build).
- [x] PDF Info dict clean: no pdfauthor; CreationDate/ModDate pinned to
      `D:19700101000000Z` (built with `SOURCE_DATE_EPOCH=0`).
- [x] Embedded figure metadata scrubbed — no `+01'00'`/`+03'30'` timezone offsets or
      2026 dates anywhere in the PDF (figure PDFs stripped with pikepdf).
- [x] Build hygiene: 0 errors, 0 warnings, 0 overfull boxes, 0 undefined refs/cites,
      38 references all resolved (the only remaining log items are 8 underfull-vbox
      messages, an irreducible artifact of tmlr.sty's mandatory `\flushbottom`).
- [x] Dead code link removed from the manuscript (the Reproducibility appendix now
      points to the supplementary material, not the placeholder anonymous.4open URL).
- [ ] **Upload the anonymized code zip as OpenReview supplementary material.** The zip
      is prepared at `scratchpad/supplementary.zip` (~1 MB; source, configs, aggregate
      + per-contrast CSVs, per-run configs; author-scrubbed). Regenerate it after
      submission-day if code changed.
- [ ] **Fill the OpenReview form:** link to the previously rejected TMLR submission +
      the changes statement above (replace the one placeholder). Confirm all authors'
      OpenReview profiles are complete (affiliations, publication history, conflicts).
      Declare funding / competing interests / IRB status (N/A here — state so).
- [ ] Confirm the private GitHub repo `Evasion-OC/when-graphs-help-marl` is still
      **private** on submission day (double-blind).
- [ ] Confirm not under review elsewhere (JAIR decision was final 2026-07-02 — clear).

## Notes

- JAIR's desk-rejection grounds (wants theory / algorithmic novelty) are **not** TMLR
  acceptance criteria (TMLR asks only: are the claims supported by convincing evidence,
  and would some of its audience be interested). Do not mention JAIR anywhere.
- Keep the statement factual and non-defensive — the substance (a new positive result,
  a mechanistically explained negative endpoint) carries it.
- To rebuild the PDF reproducibly: `cd paper && SOURCE_DATE_EPOCH=0 pdflatex main &&
  bibtex main && SOURCE_DATE_EPOCH=0 pdflatex main && SOURCE_DATE_EPOCH=0 pdflatex main`.
  If any figure is regenerated, re-run the figure-metadata scrub before the final build.
