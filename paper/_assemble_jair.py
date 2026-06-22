from pathlib import Path

lines = Path("main.tex").read_text(encoding="utf-8").split("\n")

def rng(a, b):  # 1-indexed inclusive
    return "\n".join(lines[a - 1:b])

macros   = rng(36, 246)     # \newcommand / \def block
abstract = rng(258, 328)    # abstract body (between begin/end)
body     = rng(330, 1720)   # main body, up to before \bibliographystyle
appendix = rng(1725, 2276)  # between \appendix and \end{document}

REPO = "the public code repository (\\url{https://github.com/Evasion-OC/when-graphs-help-marl})"
body = body.replace("anonymized supplementary material", REPO)
appendix = appendix.replace("anonymized supplementary material", REPO)

PRE = r"""%% JAIR submission build (Volume 83+ ACM-based style, jair.cls wrapping acmart).
%% Generated from the reviewed manuscript; TMLR build preserved in main.tex.
\documentclass[manuscript, screen, review]{jair}

\setcopyright{cc}
\acmDOI{10.1613/jair.1.xxxxx}
\JAIRAE{To be assigned}
\JAIRTrack{}
\acmVolume{1}
\acmArticle{1}
\acmMonth{6}
\acmYear{2026}

%% --- Packages the manuscript needs that acmart does not already provide. ---
%% (acmart supplies amsmath, amssymb, graphicx, booktabs, xcolor, microtype,
%%  caption, hyperref, url -- do NOT re-load those.)
\usepackage{multirow}
\usepackage{subcaption}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{enumitem}

%% --- Bibliography: biblatex + biber, acmauthoryear style (JAIR-required). ---
\RequirePackage[
  datamodel=acmdatamodel,
  style=acmauthoryear,
  backend=biber,
  giveninits=true,
  uniquename=init
  ]{biblatex}
\renewcommand*{\bibopenbracket}{(}
\renewcommand*{\bibclosebracket}{)}
\addbibresource{references.bib}

%% Map the manuscript's natbib-style citation commands onto biblatex.
%% (acmauthoryear already defines \citet.)
\providecommand{\citep}{\parencite}
\providecommand{\citealp}{\cite}

%% =========================================================================
%% Notation and number macros (carried over verbatim from the manuscript).
%% =========================================================================
"""

FRONT = r"""

\begin{document}

\title[Graph Structure in Multi-Agent RL]{When Does Graph Structure Help in Multi-Agent Reinforcement Learning? A Controlled Boundary}

%% TODO(authors): JAIR is single-blind -- replace the placeholders below with the
%% full author list, affiliations, and ORCIDs before submission.
\author{Kasra Ghanavati}
\email{kasraghanavati@icloud.com}
\orcid{0000-0000-0000-0000}
\affiliation{%
  \institution{PLACEHOLDER -- institution}
  \city{PLACEHOLDER}
  \country{PLACEHOLDER}}
\renewcommand{\shortauthors}{Ghanavati}

\begin{abstract}
"""

MID = r"""
\end{abstract}

\maketitle

"""

CHECKLIST = r"""

\printbibliography

\appendix

"""

TAIL = r"""

%% =========================================================================
\section{Reproducibility Checklist for JAIR}
Select the answers that apply to this research -- one per item.

\subsection*{All articles:}
\begin{enumerate}
  \item All claims investigated in this work are clearly stated. \textbf{[yes]}
  \item Clear explanations are given how the work reported substantiates the claims. \textbf{[yes]}
  \item Limitations or technical assumptions are stated clearly and explicitly. \textbf{[yes]}
  \item Conceptual outlines and/or pseudo-code descriptions of the AI methods introduced in this work are provided, and important implementation details are discussed. \textbf{[yes]}
  \item Motivation is provided for all design choices, including algorithms, implementation choices, parameters, data sets and experimental protocols beyond metrics. \textbf{[yes]}
\end{enumerate}

\subsection*{Articles containing theoretical contributions:}
Does this paper make theoretical contributions? \textbf{[no]}

\subsection*{Articles reporting on computational experiments:}
Does this paper include computational experiments? \textbf{[yes]}
\begin{enumerate}
  \item All source code required for conducting experiments is included in an online appendix or will be made publicly available upon publication of the paper. The online appendix follows best practices for source code readability and documentation as well as for long-term accessibility. \textbf{[yes]}
  \item The source code comes with a license that allows free usage for reproducibility purposes. \textbf{[yes]}
  \item The source code comes with a license that allows free usage for research purposes in general. \textbf{[yes]}
  \item Raw, unaggregated data from all experiments is included in an online appendix or will be made publicly available upon publication of the paper. The online appendix follows best practices for long-term accessibility. \textbf{[yes]}
  \item The unaggregated data comes with a license that allows free usage for reproducibility purposes. \textbf{[yes]}
  \item The unaggregated data comes with a license that allows free usage for research purposes in general. \textbf{[yes]}
  \item If an algorithm depends on randomness, then the method used for generating random numbers and for setting seeds is described in a way sufficient to allow replication of results. \textbf{[yes]}
  \item The execution environment for experiments, the computing infrastructure (hardware and software) used for running them, is described, including GPU/CPU makes and models; amount of memory (cache and RAM); make and version of operating system; names and versions of relevant software libraries and frameworks. \textbf{[yes]}
  \item The evaluation metrics used in experiments are clearly explained and their choice is explicitly motivated. \textbf{[yes]}
  \item The number of algorithm runs used to compute each result is reported. \textbf{[yes]}
  \item Reported results have not been ``cherry-picked'' by silently ignoring unsuccessful or unsatisfactory experiments. \textbf{[yes]}
  \item Analysis of results goes beyond single-dimensional summaries of performance (e.g., average, median) to include measures of variation, confidence, or other distributional information. \textbf{[yes]}
  \item All (hyper-) parameter settings for the algorithms/methods used in experiments have been reported, along with the rationale or method for determining them. \textbf{[yes]}
  \item The number and range of (hyper-) parameter settings explored prior to conducting final experiments have been indicated, along with the effort spent on (hyper-) parameter optimisation. \textbf{[partially]}
  \item Appropriately chosen statistical hypothesis tests are used to establish statistical significance in the presence of noise effects. \textbf{[yes]}
\end{enumerate}

\subsection*{Articles using data sets:}
Does this work rely on one or more data sets (possibly obtained from a benchmark generator or similar software artifact)? \textbf{[yes]}
\begin{enumerate}
  \item All newly introduced data sets are included in an online appendix or will be made publicly available upon publication of the paper. The online appendix follows best practices for long-term accessibility with a license that allows free usage for research purposes. \textbf{[yes]}
  \item The newly introduced data set comes with a license that allows free usage for reproducibility purposes. \textbf{[yes]}
  \item The newly introduced data set comes with a license that allows free usage for research purposes in general. \textbf{[yes]}
  \item All data sets drawn from the literature or other public sources (potentially including authors' own previously published work) are accompanied by appropriate citations. \textbf{[yes]}
  \item All data sets drawn from the existing literature (potentially including authors' own previously published work) are publicly available. \textbf{[yes]}
  \item All new data sets and data sets that are not publicly available are described in detail, including relevant statistics, the data collection process and annotation process if relevant. \textbf{[yes]}
  \item All methods used for preprocessing, augmenting, batching or splitting data sets (e.g., in the context of hold-out or cross-validation) are described in detail. \textbf{[NA]}
\end{enumerate}

\subsection*{Explanations on any of the answers above (optional):}
All environments (the \coordgrid\ gridworld and the \tokenmatch\ referential game) are
procedural generators released with the code (MIT-licensed); the external MPE and LBF
tasks are cited and publicly available. Hyper-parameters were pre-registered and locked
before confirmatory runs, so per-experiment tuning effort was deliberately minimal.

\end{document}
"""

out = PRE + macros + FRONT + abstract + MID + body + CHECKLIST + appendix + TAIL
Path("main_jair.tex").write_text(out, encoding="utf-8")
print("wrote main_jair.tex:", len(out.split(chr(10))), "lines")
