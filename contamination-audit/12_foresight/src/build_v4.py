# -*- coding: utf-8 -*-
"""One-command rebuild of the v4 paper: numbers -> figures -> independent verify -> LaTeX.
Any step that fails stops the build. Run:  python src/build_v4.py
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


run([PY, os.path.join(SRC, "make_numbers_v4.py")])
run([PY, os.path.join(SRC, "make_figures_v4.py")])
run([PY, os.path.join(SRC, "verify_v4.py")])          # exits non-zero on any number mismatch
run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "foresight_v4.tex"], cwd=PAPER)
print("\nOK -> paper/foresight_v4.pdf")
