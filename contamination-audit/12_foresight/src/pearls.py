# -*- coding: utf-8 -*-
"""Pearls on a string: every post-1930 fact as a bead on a timeline, GLOWING by how strongly the memory-meter reads it as
'remembered' (colour) and how hard the model nails the date (size = year-spike). You should see the string light up right
after the 1929 cutoff and go dark as it runs into the true future. Reads wide_net_summary.json.
Run:  python src/pearls.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import common

OUT = os.path.join(common.RESULTS, "superlatives")
S = json.load(open(os.path.join(OUT, "wide_net_summary.json"), encoding="utf-8"))
rows = [r for r in S["rows"] if r["topic"] != "Invented"]         # keep real facts on the string
yr = np.array([r["true_year"] for r in rows], float)
meter = np.array([r["memory_meter"] for r in rows])
spike = np.array([r["year_spike_Ptrue"] for r in rows])
both = np.array([r["lights_up"] == "BOTH" for r in rows])

rng = np.random.default_rng(1)
jit = rng.uniform(-0.9, 0.9, len(rows))                            # spread the crowded 1930s-40s vertically

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(figsize=(13.5, 4.6))
ax.axhline(0, color="#cfcfcf", lw=1.4, zorder=1)                   # the string
ax.axvline(1929.5, color="#1f5f9e", lw=1.2, ls="--", zorder=1)
ax.annotate("1929 cutoff", (1929.5, 1.15), color="#1f5f9e", fontsize=8.5, ha="center")
ax.annotate("← the model's past", (1929.5 - 1, -1.28), color="#888", fontsize=8, ha="right")
ax.annotate("the true future (should be dark) →", (1929.5 + 1, -1.28), color="#888", fontsize=8, ha="left")
# stems from the string to each pearl
segs = [[(yr[i], 0), (yr[i], jit[i])] for i in range(len(rows))]
ax.add_collection(LineCollection(segs, colors="#e2e2e2", linewidths=0.7, zorder=2))
sc = ax.scatter(yr, jit, c=meter, cmap="YlOrRd", vmin=0, vmax=1, s=50 + 430 * spike,
                edgecolor="#333", lw=.5, zorder=5)
# ring + label the "both light up" pearls
ax.scatter(yr[both], jit[both], s=(50 + 430 * spike[both]) + 120, facecolor="none", edgecolor="#8b0000", lw=1.8, zorder=6)
label = {"Germany surrenders", "United Nations founded", "Spanish Civil War begins", "Hungarian Revolution",
         "Pearl Harbor", "Indian independence", "9/11 attacks", "COVID-19 pandemic", "Moon landing", "Berlin Wall falls",
         "FDR elected president", "Falklands War"}
for i, r in enumerate(rows):
    if r["event"] in label:
        ax.annotate(r["event"], (yr[i], jit[i]), xytext=(0, 9 if jit[i] > 0 else -13), textcoords="offset points",
                    fontsize=6.8, ha="center", color="#222", zorder=7)
cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("memory-meter  (0 = made-up · 1 = reads as remembered)", fontsize=8.5)
ax.annotate("pearl size = how well it dates the event (year-spike)", (2016, 1.15), fontsize=8, ha="right", color="#666")
ax.annotate("red ring = lights up on BOTH indicators", (2016, -1.15), fontsize=8, ha="right", color="#8b0000")
ax.set_xlim(1928, 2023); ax.set_ylim(-1.5, 1.5)
ax.set_yticks([]); ax.set_xlabel("year of the event")
ax.set_title("Figure 17.  Ranke-1929's memory of the future, as pearls on a string. The string glows just past the cutoff and fades into the real future.",
             fontsize=10.6, loc="left")
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_pearls.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG}")
