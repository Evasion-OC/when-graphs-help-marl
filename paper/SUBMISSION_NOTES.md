# TMLR submission notes

Working notes for submitting *When Does Graph Structure Help in Multi-Agent
Reinforcement Learning? A Controlled Empirical Study* to TMLR. Not part of
the paper; for the authors only.

## 1. Recommended Action Editors (ranked)

TMLR's Editors-in-Chief assign the Action Editor (AE); authors don't pick
directly, but the AE pool is steered by the **subject areas / keywords** you
select and you can name suggested editors in the submission form. Aim the
keywords at the RL/MARL/evaluation pool, **not** the graph-representation
pool — a pure graph-learning AE may read "vanilla GCN didn't help" as
architecturally uninteresting and miss that the GCN is deliberately a
*control*. Confirmed on the board (checked May 2026):

1. **Marc Lanctot (Google DeepMind)** — *best fit.* Listed expertise:
   "computational game theory, multiagent learning, reinforcement learning,
   planning." The only AE whose core is multi-agent RL; will immediately
   grasp CTDE, the IGM/monotonic-mixer lineage (VDN→QMIX), and why MLP-QMIX
   is the right control. Most likely to value an identified negative result
   over a novelty claim.

2. **Pablo Samuel Castro (Google DeepMind)** — *co-lead, methodology axis.*
   Listed: "reinforcement learning." Coauthor of Agarwal et al. 2021
   ("statistical precipice"), which this paper cites and whose protocol it
   follows (multiple seeds, CIs, effect sizes, non-parametric tests). The AE
   most likely to *reward* the statistical care that is now the paper's
   backbone (Phase 8: n=10, Mann–Whitney + Holm). No COI — citing a
   methods paper is not a conflict.

3. **Marlos C. Machado (University of Alberta)** — *strong alternate.*
   Listed: "reinforcement learning, representation learning, exploration."
   Known for evaluation-rigor work (the ALE revisitation). Receptive to
   controlled studies and honest negative results; MARL is less his core
   than Lanctot's.

Not on the board (don't list): Matthew Taylor, Natasha Jaques, Jakob
Foerster, Shimon Whiteson.

## 2. Subject areas / keywords for the submission form

Primary: **reinforcement learning**; **multi-agent reinforcement learning**.
Secondary: **empirical / reproducibility study**, **evaluation methodology**.
Do **not** set *graph neural networks* as the primary area.

## 3. One-paragraph "suggested editors" note (paste into the form if offered)

> This is a controlled empirical study in cooperative MARL (value
> decomposition: IQL/VDN/QMIX, plus a GCN variant and a parameter-matched
> no-graph control). Its contribution is measurement and identification
> rather than a new method, in the lineage of Henderson et al. (2018) and
> Agarwal et al. (2021). We believe an action editor with multi-agent RL
> and/or RL-evaluation expertise (e.g. Marc Lanctot, Pablo Samuel Castro,
> or Marlos C. Machado) is the best match; a pure graph-representation
> editor may misread the deliberately-minimal GCN control as the
> contribution.

## 4. What strengthened since the first draft (mention in the cover / response)

The revision adds **Phase 8**, which closes the two largest self-flagged
limitations of the original submission:

- **Capacity control (former Limitation #6).** New `MLP-QMIX` arm:
  byte-identical parameter count to GNN-QMIX, GCN→MLP (no graph). Isolates
  *graph* from *capacity*. Result: the no-graph control matches QMIX while
  GNN-QMIX falls far below both → the penalty is the graph, not the params.
- **Undertraining + seed count (former Limitation #5 and the Phase-2
  caveat).** Decisive cells re-run at 150k steps (5×) with **n=10** seeds.
  The negative result holds and strengthens; significant under Mann–Whitney
  (p=0.008) and Holm-corrected Welch.
- **New finding:** the graph also *destabilises* training (seed SD ≫ QMIX
  at N=8).

Net effect: the central claim moves from a *confounded, n=3* negative to an
*identified, n=10, confound-free* negative — exactly the evidence-sufficiency
bar TMLR weights.

## 5. Double-blind reminder

TMLR review is double-blind. Before/at submission:
- Submit the **anonymized** build (`\usepackage{tmlr}` without `[preprint]`).
- The companion GitHub repo was made **public** temporarily for the Colab/
  local runs; **flip it back to private** before submission (or anonymize
  it), since the repo name + commit authorship deanonymize the authors.
  `gh repo edit Evasion-OC/when-graphs-help-marl --visibility private
  --accept-visibility-change-consequences`
- If you want code available to reviewers, use an anonymized mirror
  (e.g. anonymous.4open.science) and put that URL in Appendix C instead of
  the GitHub link.
