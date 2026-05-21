# Venue Research: GNN-QMIX Negative-Result Empirical Study

Paper profile: ~17-page controlled empirical MARL study; 16 conditions; 3 pre-registered hypotheses (2 falsified, 1 partial); 27 references; reproducible (GitHub + Colab); commodity-CPU compute; Holm-corrected stats; single GNN family (GCN), gridworld + partial MPE.

The headline contribution is **negative/corrective**, so venue stance on null and reproducibility-style results is the dominant filter.

---

## 1. TMLR (Transactions on Machine Learning Research)

- **Page limit / length**: No hard limit. Authors should justify length; "unusually long" papers (excluding appendices) face review delays. A 12-page main body is the soft norm tied to the 9-week review cadence ([author guide](https://jmlr.org/tmlr/author-guide.html)).
- **Format**: TMLR LaTeX stylefile (Computer Modern Bright, 10pt, 6.5x9in text block). Overleaf template available ([TMLR Overleaf](https://www.overleaf.com/latex/templates/tmlr-journal-submissions/mwchznnhdtwx)). Numbered citations by default in the stylefile.
- **Review philosophy**: Acceptance is based on (1) technical soundness of claims and (2) whether *some* portion of TMLR's audience would find value. Explicitly **does not require novelty, SOTA, or significance**; reproducibility-style and corrective reports are explicitly mentioned as in-scope ([acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html)).
- **Acceptance rate / turnaround**: ~50–62% accept; median ~76–91 days to decision; reviews due 2 weeks after assignment ([Kamath stats 2023](https://x.com/thegautamkamath/status/1780695733928358007)).
- **Verdict**: **Excellent fit.** Of all venues considered, TMLR is the most explicit in welcoming negative/corrective empirical findings; the paper's current length and rigor map cleanly onto its norms.

## 2. RLC 2026 (Reinforcement Learning Conference)

- **Page limit**: Recommended 8, **strict 12** for main body (excluding references). Appendices appear *before* references and *count* toward the page limit; supplementary material after references is unlimited and not necessarily reviewed ([submission instructions](https://rl-conference.cc/submissionInstructions.html), [CFP](https://rl-conference.cc/callforpapers.html)).
- **Format**: Provided LaTeX style via Overleaf; single PDF on OpenReview.
- **Deadlines (2026 cycle)**: Abstract Mar 1, paper Mar 5 AOE; rebuttal Apr 11–17; decisions May 5; conference Aug 16–19, 2026, Montreal.
- **Review philosophy**: Explicitly prioritizes "rigorous methodology over subjective perceived importance"; venue is RL-native and friendly to careful empirical work. Accepted papers published in RLJ.
- **Acceptance rate**: RLC 2025 was 115/295 = **39%** ([RLC 2025 stats](https://aip.riken.jp/news/rlc2025/)).
- **Verdict**: **Strong fit.** RL-native venue with explicit methodology-first stance; the 17-page draft would need a trim to 12pp main + supplementary, but the substance lands well.

## 3. AAMAS 2026

- **Page limit**: **8 pages main + unlimited references** (full paper) or 2 pages + refs (extended abstract). Appendix material must fit within the 8pp or go to supplementary ([AAMAS 2026 submission](https://cyprusconferences.org/aamas2026/submission-instructions/)).
- **Format**: AAMAS LaTeX template (numbered-style ACM-derived); PDF via the conference system.
- **Deadlines**: Abstract Oct 1, 2025; paper Oct 8, 2025; notification Dec 22, 2025 — **the 2026 cycle has already closed**; next opportunity is AAMAS 2027.
- **Review philosophy**: Rigorous peer review on originality, significance, soundness, reproducibility, clarity. No explicit statement on negative results, but the MARL/multi-agent scope is centrally relevant.
- **Acceptance rate**: AAMAS 2025 = 250 full + 158 extended out of 1021 = ~25% full / ~40% combined ([AAMAS 2025 program](https://aamas2025.org/index.php/conference/program/accepted-papers/)).
- **Verdict**: **Scope-perfect but timing-blocked and tight on length.** The 8pp cap forces an aggressive cut for a corrective study that benefits from full reporting. Worth holding for AAMAS 2027 only if no faster venue is preferred.

## 4. NeurIPS 2025/2026 Workshops

NeurIPS 2025 took place in San Diego in Dec 2025 — those workshop deadlines have passed. The most fit-relevant 2025 workshop was **ARLET** (Agentic RL Evaluation & Training): 9-page research-paper track, NeurIPS 2025 LaTeX template, appendices after references ([ARLET CFP](https://arlet-workshop.github.io/neurips2025/cfp)). NeurIPS 2026 workshop CFPs are not yet posted (announcements typically late summer 2026).

Likely-fit candidates to monitor for **NeurIPS 2026** (based on recurring series):
- **Cooperative AI Workshop** (recurring since 2021): typically 4–8 pages, NeurIPS style.
- **ARLET / RL-eval style workshop** (if it returns): 9pp + appendix.
- **MARL or Cooperative MAS workshop** (multiple proposals each year): typically 4–8 pages.

- **Verdict**: Viable as a **shorter parallel submission** while a longer TMLR/RLC version is in flight, but timing means waiting until ~Aug–Sep 2026 CFPs.

## 5. ICML 2026 Workshops

ICML 2026 workshop slate is announced ([ICML 2026 workshops blog](https://blog.icml.cc/2026/04/06/announcing-the-icml-2026-workshops-and-affinity-workshops/)). Most-fit candidates:

- **"Failure Modes in Agentic AI: Reproducible Triggers, Trace Diagnostics, and Verified Fixes"** — explicit focus on reproducible failure cases; near-ideal philosophical match for a corrective null-result paper.
- **"Decision-Making from Offline Datasets to Online Adaptation: BBO to RL"** — RL-scoped; reasonable adjacency.
- **"New Frontiers in Game-Theoretic Learning (NExT-Game)"** — cooperative/competitive learning dynamics; partial scope match.

- **Page limit**: Workshop-specific, typically 4–8 pages in ICML LaTeX style; CFPs roll out roughly Mar–May 2026 with deadlines in May–Jun 2026. (Unverified per-workshop limits — check each CFP.)
- **Verdict**: The "Failure Modes" workshop is the **single best workshop match** for this paper's framing; submit a 4–8pp condensed version.

## 6. JMLR (full archival journal)

- **Page limit**: No hard cap; submissions are typically 20–60 pages.
- **Format**: JMLR LaTeX template; numbered citations.
- **Scope / philosophy**: Wants "new principled algorithms with sound empirical validation," new theoretical insight, or formalization of new tasks ([JMLR author info](https://jmlr.org/author-info.html)). Empirical-only corrective studies are not its sweet spot; turnaround often >200 days.
- **Acceptance rate / turnaround**: Highly selective; median decision >200 days.
- **Verdict**: **Poor fit.** A pure 16-condition negative empirical study without a new algorithm or framework is unlikely to clear JMLR's algorithm/theory bar; TMLR is the right JMLR-family target instead.

---

## Length recommendation

The 17-page draft is **mid-length**: too long for AAMAS/RLC main bodies (8/12 pp) and too short for JMLR norms. Two natural targets:

- **Long version (preferred archival home): TMLR.** Keep the full 17 pages plus the existing appendices — the format welcomes thorough reporting of 16 conditions, pre-registered hypotheses, and per-condition statistics.
- **Short version (parallel workshop): ICML 2026 "Failure Modes in Agentic AI"** or a NeurIPS 2026 cooperative-AI / RL workshop, in a 4–8pp condensed form, to surface the result to the MARL community faster.

If a venue compresses to 8 pp (AAMAS) or 12 pp (RLC), the experimental-design and limitations sections will need to move to supplementary; the headline negative finding and stats are defensible at either length.

---

## Top-3 Ranked Recommendation

1. **TMLR — best single fit.** The acceptance criteria explicitly drop "novelty/SOTA" as requirements and explicitly welcome corrective and reproducibility-style studies; the paper's current length and statistical rigor align with the journal's norms; ~3-month turnaround is competitive with conferences; archival, citable, and respected in the ML/RL community.
2. **RLC 2026.** RL-native audience, methodology-first review philosophy, and a clear March 2026 deadline. Requires trimming to 12pp main; appendices count toward the limit but supplementary is free. Strong second option if community visibility within the RL world matters.
3. **ICML 2026 Workshop — "Failure Modes in Agentic AI" (parallel short version).** Topically perfect for a reproducible negative-result paper; use as a fast-feedback companion to a TMLR/RLC submission rather than the archival home.

**Single best fit: TMLR.** The paper is an empirical, statistically rigorous, corrective study with a reproducible artifact — exactly the profile TMLR was designed to welcome, and the only major venue that *publicly commits* to not penalizing lack of novelty or SOTA.

---

## Sources

- [TMLR author guide](https://jmlr.org/tmlr/author-guide.html)
- [TMLR acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html)
- [TMLR Overleaf template](https://www.overleaf.com/latex/templates/tmlr-journal-submissions/mwchznnhdtwx)
- [TMLR stats (Kamath, 2023)](https://x.com/thegautamkamath/status/1780695733928358007)
- [RLC 2026 CFP](https://rl-conference.cc/callforpapers.html)
- [RLC 2026 submission instructions](https://rl-conference.cc/submissionInstructions.html)
- [RLC 2025 results](https://aip.riken.jp/news/rlc2025/)
- [AAMAS 2026 submission instructions](https://cyprusconferences.org/aamas2026/submission-instructions/)
- [AAMAS 2025 accepted papers](https://aamas2025.org/index.php/conference/program/accepted-papers/)
- [ICML 2026 workshops announcement](https://blog.icml.cc/2026/04/06/announcing-the-icml-2026-workshops-and-affinity-workshops/)
- [ARLET (NeurIPS 2025) CFP](https://arlet-workshop.github.io/neurips2025/cfp)
- [JMLR author information](https://jmlr.org/author-info.html)
