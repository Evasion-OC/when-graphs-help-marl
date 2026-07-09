# JAIR submission form — answers to paste

## Section / track
Select **"Articles"** (standard research article). Not a survey, not a special track.

## "Comments for the Editor" box — paste the three answers below

---

**1. Why your work is important to AI researchers and how they can use it (≤150 words)**

Practitioners routinely add graph neural networks to cooperative multi-agent RL assuming
relational structure helps, yet the field has no controlled account of when it does. We
provide one. Holding the encoder, mixer, and optimiser fixed and varying only the
coordination graph, we isolate that graph *structure* — not communication bandwidth or
added capacity — produces large gains (Cohen's d ≈ 5) exactly when an agent must act on
information held by a specific, unobserved other and the topology is known or estimable;
the same network instead *hurts* when coordination is incidental. Researchers gain (i) a
concrete decision rule for whether to use a coordination graph, (ii) a reusable
experimental template — matched wrong-graph, no-graph, and all-to-all controls at
byte-identical parameters — for correctly attributing MARL improvements, and (iii) a
caution against default all-to-all message passing. Full code, configurations, and
pre-registration are released.

---

**2. Closest 1–3 JAIR papers and how your work differs (≤150 words)**

The closest JAIR article is Varricchione, Alechina, Dastani, and Logan, "Synthesising
Reward Machines for Cooperative Multi-Agent Reinforcement Learning" (Vol. 85, 2026): like
ours it studies cooperative MARL with decentralised execution, but it encodes team
structure through reward machines rather than coordination graphs and does not isolate when
relational structure helps. A second, Cigler and Faltings, "Decentralized Anti-coordination
Through Multi-agent Learning" (Vol. 47, 2013), shares our interest in coordination signals
but targets equilibrium convergence in resource-allocation games, not deep value
factorisation. Neither performs our central experiment — fixing the network and contrasting
a task-matched graph against density-matched wrong-graph, all-to-all, and parameter-matched
no-graph controls to attribute gains to topology alone. Our paper follows the controlled
empirical-study structure common to JAIR experimental articles while contributing a new
boundary result and an actionable decision rule for graph-based cooperative MARL.

---

**3. Previously published / under review? (state "No" if not)**

No. The work is original and unpublished, and no part of it is published or currently under
review elsewhere. This is a first-time submission to JAIR, not a resubmission.

---

## Checkbox declarations — all satisfied, tick each

| Declaration | Status |
|---|---|
| Correct Section (Articles) | ✅ select "Articles" |
| Read Open Access Publication Agreement | ✅ read it before ticking (link in form) |
| Important to the AI community | ✅ controlled boundary result + decision rule |
| Original, unpublished, claims supported | ✅ |
| First-time submission (not a resubmission) | ✅ |
| Not under review / published elsewhere | ✅ |
| Proof-read and edited | ✅ |
| Formatted to JAIR requirements, PDF | ✅ jair.cls, clean build, PDF only |
| Not in a summary-rejection category | ✅ original methods/insights, well-referenced through 2026 |

## Corresponding contact (bottom of form)
- ✅ "Yes, I would like to be contacted about this submission."
- ✅ "Yes, I agree to have my data collected and stored…" (privacy statement)
