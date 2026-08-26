# -*- coding: utf-8 -*-
"""One-command rebuild of the v5 paper: numbers -> figures -> independent verify -> LaTeX.
Any step that fails stops the build. Run:  python src/build_v5.py
"""
import os, sys, subprocess
import common

PY = sys.executable
SRC = os.path.join(common.ROOT, "src")
PAPER = common.PAPER


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"STEP FAILED ({r.returncode}): {' '.join(cmd)}")


run([PY, os.path.join(SRC, "make_numbers_v5.py")])
run([PY, os.path.join(SRC, "make_figures_v5.py")])
run([PY, os.path.join(SRC, "verify_v5.py")])          # exits non-zero on any number mismatch
# latexmk needs perl, which this MiKTeX install lacks; two pdflatex passes resolve all refs
# (the bibliography is inline thebibliography, so no bibtex pass is needed).
for _ in range(2):
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "foresight_v5.tex"], cwd=PAPER)
print("\nOK -> paper/foresight_v5.pdf")
