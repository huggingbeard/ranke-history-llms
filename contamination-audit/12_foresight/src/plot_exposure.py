# -*- coding: utf-8 -*-
"""Scatter: does the memory-meter track how much each REAL treaty actually appears in pre-1929 books? Every dot is a
real treaty; x = its exposure (Google Books frequency before 1929, log scale), y = the v2 meter. Coloured by whether the
model can behaviourally answer about it (known / unknown / ambiguous). Fit line + correlation on the non-zero points.
Treaties absent from the record (exposure = 0, undefined on a log axis) are shown in a separate strip on the left.
Reads results/superlatives/treaty_exposure.csv. Run:  python src/plot_exposure.py
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import common

OUT = os.path.join(common.RESULTS, "superlatives")
rows = list(csv.DictReader(open(os.path.join(OUT, "treaty_exposure.csv"), encoding="utf-8-sig")))
exp = np.array([float(r["exposure_pre1929"]) for r in rows])
meter = np.array([float(r["meter_v2"]) for r in rows])
lab = np.array([r["behavioural_label"] for r in rows])
name = [r["treaty"] for r in rows]
COL = {"known": "#2e7d32", "unknown": "#c0392b", "ambiguous": "#8a8f98"}
LABN = {"known": "model can answer (known)", "unknown": "model can't answer (unknown)", "ambiguous": "ambiguous"}

nz = exp > 0
lx = np.log10(exp[nz])
b, a = np.polyfit(lx, meter[nz], 1)                     # meter ≈ a + b*log10(exposure)
r = float(np.corrcoef(lx, meter[nz])[0, 1])
floor = lx.min() - 1.6                                  # x-position for the "absent from the record" strip
div = lx.min() - 0.8

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(figsize=(11.0, 7.0))
# fit line (over the non-zero range)
xs = np.linspace(lx.min() - 0.2, lx.max() + 0.2, 50)
ax.plot(xs, np.clip(a + b * xs, -0.02, 1.02), color="#1f5f9e", lw=2.2, zorder=2,
        label=f"fit: meter rises with exposure  (r = {r:.2f})")
ax.axvline(div, color="#bbb", lw=1, ls=":")
ax.axvspan(floor - 0.6, div, color="#f4f4f4", zorder=0)
ax.annotate("absent from\nthe record\n(exposure ≈ 0)", (floor, 0.5), fontsize=8, color="#888", ha="center", va="center")
# points
rng = np.random.default_rng(0)
for i, (n, e, m, g) in enumerate(zip(name, exp, meter, lab)):
    x = np.log10(e) if e > 0 else floor + rng.uniform(-0.28, 0.28)
    ax.scatter(x, m, s=64, color=COL[g], edgecolor="#111", lw=.5, alpha=.9, zorder=5)
ANN = {"Küçük Kaynarca": "false alarm — but it IS in the record\n(caught a faint trace?)",
       "Polyanovka": "false alarm with ZERO exposure\n(pure realness leak)",
       "Hünkâr İskelesi": "most-seen 'unknown' — yet meter missed it",
       "Versailles": "", "Utrecht": ""}
for i, n in enumerate(name):
    if n in ANN:
        x = np.log10(exp[i]) if exp[i] > 0 else floor
        ax.annotate(n + (("\n" + ANN[n]) if ANN[n] else ""), (x, meter[i]),
                    xytext=(7, -4 if meter[i] > .5 else 8), textcoords="offset points", fontsize=7.2,
                    color="#333", zorder=6)
ax.set_xlabel("how often “Treaty of X” appears in books before 1929   —   log scale (← vanishing · common →)")
ax.set_ylabel("memory-meter (v2)   —   0 = reads as made-up,  1 = reads as real memory")
ax.set_ylim(-0.04, 1.06); ax.set_xlim(floor - 0.7, lx.max() + 1.4)
ax.set_title("Figure 13.  The meter partly tracks real text-exposure — but not cleanly.\n"
             "Each dot is a genuine treaty; the rising line says more-written-about treaties read as better-remembered.", fontsize=10.6, loc="left")
handles = [Line2D([0], [0], color="#1f5f9e", lw=2.2, label=f"fit (r = {r:.2f})")] + \
          [Line2D([0], [0], marker="o", color="w", markerfacecolor=COL[g], markeredgecolor="#111", markersize=9, label=LABN[g])
           for g in ("known", "unknown", "ambiguous")]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(color="#eee", lw=.5)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_exposure_scatter.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"fit slope {b:+.3f} per log10 exposure, intercept {a:.2f}, r {r:.2f} (non-zero n={nz.sum()})")
print(f"wrote {FIG}")
