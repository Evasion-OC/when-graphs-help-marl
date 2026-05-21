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
        "## 1b. (Recommended) Mount Google Drive as a persistent backup",
        "",
        "Colab runtimes disconnect after ~90 minutes of idle, or get pre-empted on the "
        "free tier. If that happens mid-experiment, everything in `/content/` is lost.",
        "",
        "Mounting Drive lets us *checkpoint after each phase* into "
        "`MyDrive/gnnmarl-results/`. Even if the runtime dies between phases, the "
        "completed phases survive and you can rerun just the missing ones.",
        "",
        "If you skip this cell, the notebook still works — you just have to babysit the "
        "tab to grab the download zip at the end.",
    ),
    code(
        "USE_DRIVE = True   # set False to skip\n",
        "DRIVE_DIR = '/content/drive/MyDrive/gnnmarl-results'\n",
        "\n",
        "if USE_DRIVE:\n",
        "    from google.colab import drive\n",
        "    drive.mount('/content/drive')\n",
        "    os.makedirs(DRIVE_DIR, exist_ok=True)\n",
        "    print(f'Drive mounted; checkpoints will go to {DRIVE_DIR}')\n",
        "else:\n",
        "    DRIVE_DIR = None\n",
        "    print('skipping Drive — outputs only available via the final zip download')\n",
    ),
    code(
        "# Helper: copy results + paper into Drive after each phase. Idempotent.\n",
        "import shutil\n",
        "from pathlib import Path\n",
        "\n",
        "def checkpoint(label: str) -> None:\n",
        "    if not DRIVE_DIR:\n",
        "        return\n",
        "    dst = Path(DRIVE_DIR) / label\n",
        "    if dst.exists():\n",
        "        shutil.rmtree(dst)\n",
        "    dst.mkdir(parents=True, exist_ok=True)\n",
        "    for src in [Path('results'), Path('paper')]:\n",
        "        if src.exists():\n",
        "            shutil.copytree(src, dst / src.name, dirs_exist_ok=True)\n",
        "    print(f'[checkpoint] {label} -> {dst}')\n",
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
        "    checkpoint('after_phase1')\n",
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
        "    checkpoint('after_phase2')\n",
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
        "    checkpoint('after_phase4')\n",
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
        "import shutil, os, zipfile\n",
        "from pathlib import Path\n",
        "\n",
        "out = Path('gnnmarl_outputs.zip')\n",
        "if out.exists(): out.unlink()\n",
        "\n",
        "# Comprehensive bundle: results, figures, full paper source + PDF,\n",
        "# review notes, and the reproducibility docs. Sized so a local unzip\n",
        "# into the repo root restores a working tree.\n",
        "files: list[Path] = []\n",
        "for p in Path('results').rglob('*'):\n",
        "    if p.is_file(): files.append(p)\n",
        "for pattern in [\n",
        "    'paper/main.tex',\n",
        "    'paper/main.pdf',\n",
        "    'paper/references.bib',\n",
        "    'paper/REVIEW_NOTES.md',\n",
        "    'paper/results/*.tex',\n",
        "    'paper/figures/*',\n",
        "    'docs/REPRO.md',\n",
        "    'docs/PHASES.md',\n",
        "    'docs/INTERFACES.md',\n",
        "]:\n",
        "    for p in Path('.').glob(pattern):\n",
        "        if p.is_file(): files.append(p)\n",
        "\n",
        "with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:\n",
        "    for p in files:\n",
        "        zf.write(p)\n",
        "print(f'wrote {out} -- {out.stat().st_size / 1e6:.1f} MB, {len(files)} files')\n",
        "\n",
        "# Belt-and-suspenders: copy the zip to Drive if mounted, so it survives\n",
        "# Colab disconnect even if you don't accept the browser download.\n",
        "if DRIVE_DIR:\n",
        "    drive_out = Path(DRIVE_DIR) / out.name\n",
        "    shutil.copy2(out, drive_out)\n",
        "    print(f'also copied to {drive_out}')\n",
        "    # And a final phase-agnostic checkpoint of paper + results.\n",
        "    checkpoint('final')\n",
        "\n",
        "# Trigger the browser download. Requires you to accept the file dialog.\n",
        "try:\n",
        "    from google.colab import files as gcf\n",
        "    gcf.download(str(out))\n",
        "except Exception as e:\n",
        "    print(f'download dialog skipped: {e}')\n",
        "    print(f'grab the zip yourself from {out.resolve()}')\n",
    ),
    md(
        "## What got saved, and where",
        "",
        "**On Google Drive** (if mounted in cell 1b):",
        "- `MyDrive/gnnmarl-results/after_phase1/` — full `results/` + `paper/` snapshot "
        "after Phase 1 finishes. Same for `after_phase2/`, `after_phase4/`, and `final/`. "
        "Each is a complete tree, so any one of them is independently usable.",
        "- `MyDrive/gnnmarl-results/gnnmarl_outputs.zip` — the final bundle.",
        "",
        "**On your local machine** (after accepting the download dialog):",
        "- `gnnmarl_outputs.zip` in your browser's Downloads folder.",
        "",
        "**Contents of the zip:**",
        "- Every per-run `episodes.csv` + `config.json` under `results/phase{1,2,4}/`",
        "- All learning-curve, heatmap, and forest-plot figures (PDF + PNG)",
        "- Per-phase summary, pairwise, and hypothesis-test CSVs",
        "- `paper/main.tex` (source), `paper/main.pdf` (compiled), `paper/references.bib`",
        "- `paper/REVIEW_NOTES.md`, `paper/results/*.tex` (auto-generated inserts)",
        "- `docs/{REPRO,PHASES,INTERFACES}.md`",
        "",
        "## Local follow-up",
        "",
        "```bash",
        "cd path/to/when-graphs-help-marl",
        "unzip ~/Downloads/gnnmarl_outputs.zip   # overlays results/ and paper/",
        "open paper/main.pdf",
        "```",
        "",
        "If you edit the paper source and want to recompile against the same data:",
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
