"""Generator for notebooks/colab_runner.ipynb.

Run this script to (re)build the Colab notebook from a flat, code-reviewable
Python source. Keeping the notebook generator under version control means
diffs are readable and the notebook itself doesn't need hand-editing.

Usage::

    python notebooks/_build_colab_runner.py
"""

from __future__ import annotations

import json
from pathlib import Path


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l if l.endswith("\n") else l + "\n" for l in lines][:-1]
                  + [lines[-1]],
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [l if l.endswith("\n") else l + "\n" for l in lines][:-1]
                  + [lines[-1]],
    }


CELLS = [
    md(
        "# When Does Graph Structure Help in MARL? --- Colab runner",
        "",
        "This notebook reproduces every phase of the paper *When Does Graph "
        "Structure Help in Multi-Agent Reinforcement Learning?* in one place.",
        "",
        "**What it does** (≈30--90 min wall on Colab CPU, ≈15--40 min on T4 GPU):",
        "",
        "1. Clone the repo at the desired branch (default `phase-1-pilot`).",
        "2. Install dependencies.",
        "3. Run the unit-test suite as a smoke check.",
        "4. Run each phase (1, 2, 4) sequentially. **Toggle phases in the *Run config* cell.**",
        "5. Generate figures + summary tables under `results/<phase>/`.",
        "6. Compile `paper/main.pdf` with the real results inserted.",
        "7. Zip everything (results + paper PDF) into `gnnmarl_outputs.zip` and download it.",
        "",
        "**Why use Colab?** The model is tiny (~30k params); GPU offers a modest 2--3x speedup "
        "over Colab CPU for this workload. The bigger reason is convenience and reproducibility.",
        "",
        "**After the run**, unzip `gnnmarl_outputs.zip` into the repo's `results/` directory "
        "on your local machine; downstream analysis scripts and the paper build pick it up "
        "automatically.",
    ),
    md(
        "## 1. Setup",
        "",
        "Clone the repo and install the package. Change `BRANCH` if you want a different "
        "branch (e.g. `main` once Phase 1 is merged).",
    ),
    code(
        "REPO_URL = 'https://github.com/Evasion-OC/when-graphs-help-marl.git'\n"
        "BRANCH   = 'phase-1-pilot-v2'   # change if needed\n",
        "",
        "import os, subprocess, sys\n",
        "if not os.path.exists('when-graphs-help-marl'):\n",
        "    subprocess.check_call(['git', 'clone', '--branch', BRANCH, '--depth', '1', REPO_URL])\n",
        "os.chdir('when-graphs-help-marl')\n",
        "print('cwd:', os.getcwd())\n",
        "subprocess.check_call(['git', 'log', '--oneline', '-1'])\n",
    ),
    code(
        "!pip install -q -e '.[dev]' 2>&1 | tail -5\n",
    ),
    md(
        "## 2. Smoke check",
        "",
        "Run the full unit + integration test suite. Should report ~71 passed.",
    ),
    code(
        "!PYTHONIOENCODING=utf-8 python -m pytest -q 2>&1 | tail -5\n",
    ),
    md(
        "## 3. GPU / device check",
        "",
        "Colab gives you either a T4 GPU or CPU-only depending on the runtime type "
        "(Runtime > Change runtime type). The trainer auto-detects.",
    ),
    code(
        "import torch\n",
        "print('torch', torch.__version__)\n",
        "print('cuda available:', torch.cuda.is_available())\n",
        "if torch.cuda.is_available():\n",
        "    print('device:', torch.cuda.get_device_name(0))\n",
    ),
    md(
        "## 4. Run config",
        "",
        "Toggle which phases to run. Defaults run all three reportable phases at their "
        "scaled budgets (matches what the paper reports).",
        "",
        "If Colab disconnects mid-run, just re-run the affected phase cell; existing "
        "CSVs under `results/<phase>/<run_id>/episodes.csv` are not overwritten unless "
        "you clear that directory.",
    ),
    code(
        "RUN_PHASE_1 = True   # pilot: 4 algos x 3 seeds x 50k steps  (~30-90 min)\n",
        "RUN_PHASE_2 = True   # E1 sweep: 4 algos x 2 N x 2 graphs x 3 seeds x 30k  (~50-120 min)\n",
        "RUN_PHASE_4 = True   # depth x diameter ablation: 3 graphs x 4 depths x 3 seeds x 20k  (~30-90 min)\n",
    ),
    md(
        "## 5. Phase 1 --- pilot",
        "",
        "Trains each of {IQL, VDN, QMIX, GNN-QMIX} for 3 seeds x 50k env steps on "
        "CoordGrid ($N=4$, ring). Per-run wall is ~3--6 min depending on hardware.",
    ),
    code(
        "if RUN_PHASE_1:\n",
        "    !PYTHONIOENCODING=utf-8 python scripts/phase1_pilot.py 2>&1 | tee results/phase1_log.txt\n",
        "    !PYTHONIOENCODING=utf-8 python scripts/phase1_analysis.py 2>&1 | tail -20\n",
        "else:\n",
        "    print('skipped phase 1')\n",
    ),
    md(
        "## 6. Phase 2 --- E1 sweep (H1, H2)",
        "",
        "$N \\in \\{4, 8\\} \\times \\{$ring, Erdős--Rényi (matched density)$\\} \\times$ 4 algos "
        "$\\times$ 3 seeds at $3\\times 10^4$ env steps per run. 48 runs total.",
    ),
    code(
        "if RUN_PHASE_2:\n",
        "    !PYTHONIOENCODING=utf-8 python scripts/phase2_e1_sweep.py 2>&1 | tee results/phase2_log.txt\n",
        "    !PYTHONIOENCODING=utf-8 python scripts/phase2_analysis.py 2>&1 | tail -30\n",
        "else:\n",
        "    print('skipped phase 2')\n",
    ),
    md(
        "## 7. Phase 4 --- depth $\\times$ diameter ablation (H3)",
        "",
        "GNN depth $L \\in \\{1,2,3,4\\}$ crossed with graph diameter $d \\in \\{1,2,3\\}$ "
        "($N=4$ fixed), 3 seeds, 20k env steps per run. 36 runs total.",
    ),
    code(
        "if RUN_PHASE_4:\n",
        "    !PYTHONIOENCODING=utf-8 python scripts/phase4_ablation.py 2>&1 | tee results/phase4_log.txt\n",
        "    !PYTHONIOENCODING=utf-8 python scripts/phase4_analysis.py 2>&1 | tail -20\n",
        "else:\n",
        "    print('skipped phase 4')\n",
    ),
    md(
        "## 8. Build the paper",
        "",
        "Regenerates the LaTeX inserts from the result CSVs and compiles `paper/main.pdf`. "
        "Requires `texlive` --- Colab usually has it but the install line is here just in case.",
    ),
    code(
        "import subprocess\n",
        "try:\n",
        "    subprocess.check_call(['which', 'pdflatex'])\n",
        "    have_latex = True\n",
        "except subprocess.CalledProcessError:\n",
        "    have_latex = False\n",
        "if not have_latex:\n",
        "    !apt-get -qq install -y texlive-latex-extra texlive-fonts-recommended texlive-science 2>&1 | tail -3\n",
    ),
    code(
        "!PYTHONIOENCODING=utf-8 python scripts/build_paper_inserts.py\n",
        "%cd paper\n",
        "!pdflatex -interaction=nonstopmode main.tex > /tmp/lp.txt 2>&1 ; bibtex main > /tmp/bt.txt 2>&1 ; pdflatex -interaction=nonstopmode main.tex > /tmp/lp.txt 2>&1 ; pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -3\n",
        "%cd ..\n",
    ),
    md(
        "## 9. Bundle + download",
        "",
        "Zips the `results/` directory and the compiled `paper/main.pdf` for download. "
        "After downloading, unzip into the repo's root on your local machine so the next "
        "iteration of analysis / writeup sees the data.",
    ),
    code(
        "import shutil, os\n",
        "from pathlib import Path\n",
        "out = Path('gnnmarl_outputs.zip')\n",
        "if out.exists(): out.unlink()\n",
        "# Bundle the parts a downstream user actually needs: result CSVs + figures + PDF.\n",
        "files = []\n",
        "for p in Path('results').rglob('*'):\n",
        "    if p.is_file(): files.append(p)\n",
        "for p in Path('paper').glob('main.pdf'):\n",
        "    files.append(p)\n",
        "for p in Path('paper').glob('references.bib'):\n",
        "    files.append(p)\n",
        "for p in Path('paper/results').glob('*.tex'):\n",
        "    files.append(p)\n",
        "for p in Path('paper/figures').glob('*'):\n",
        "    files.append(p)\n",
        "import zipfile\n",
        "with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:\n",
        "    for p in files:\n",
        "        zf.write(p)\n",
        "print(f'wrote {out} -- {out.stat().st_size / 1e6:.1f} MB, {len(files)} files')\n",
        "from google.colab import files as gcf\n",
        "gcf.download(str(out))\n",
    ),
    md(
        "## After downloading: integration on your local machine",
        "",
        "```bash",
        "cd path/to/when-graphs-help-marl",
        "unzip gnnmarl_outputs.zip   # overlays results/ and paper/",
        "open paper/main.pdf",
        "```",
        "",
        "If you want to regenerate the paper from the CSVs on your local machine:",
        "",
        "```bash",
        "bash scripts/build_paper.sh",
        "```",
    ),
]


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    out = Path(__file__).resolve().parent / "colab_runner.ipynb"
    out.write_text(json.dumps(NOTEBOOK, indent=1), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} B, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
