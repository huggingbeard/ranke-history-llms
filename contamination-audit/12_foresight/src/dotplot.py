# -*- coding: utf-8 -*-
"""Per-event dumbbell: one row per post-1930 fact, ordered most-recent (top) to oldest (bottom). Blue dot = year-spike
P(true year); red dot = memory-meter. The gap between the dots shows the model recognising an event (red) without being
able to date it (blue). Reads wide_net_summary.json. Run:  python src/dotplot.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

OUT = os.path.join(common.RESULTS, "superlatives")
S = json.load(open(os.path.join(OUT, "wide_net_summary.json"), encoding="utf-8"))
THR_S, THR_M = S["thr_spike"], S["thr_meter"]
rows = [r for r in S["rows"] if r["topic"] != "Invented"]
rows.sort(key=lambda r: (r["true_year"], r["memory_meter"]))       # oldest -> newest (bottom -> top)
N = len(rows)
y = np.arange(N)
spike = np.array([r["year_spike_Ptrue"] for r in rows])
meter = np.array([r["memory_meter"] for r in rows])
BLUE, RED = "#1f5f9e", "#c0392b"

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(figsize=(9.6, 0.23 * N + 1.6))
for i in range(N):
    if i % 2 == 0:
        ax.axhspan(i - 0.5, i + 0.5, color="#f6f6f6", zorder=0)
ax.axvline(THR_S, color=BLUE, lw=.9, ls=":", zorder=1)
ax.axvline(THR_M, color=RED, lw=.9, ls=":", zorder=1)
ax.hlines(y, spike, meter, color="#cccccc", lw=1.3, zorder=2)      # the connector line per event
ax.scatter(spike, y, s=42, color=BLUE, edgecolor="white", lw=.5, zorder=4)
ax.scatter(meter, y, s=42, color=RED, edgecolor="white", lw=.5, zorder=4)
both = np.array([r["lights_up"] == "BOTH" for r in rows])
ax.scatter(meter[both], y[both], s=150, facecolor="none", edgecolor="#8b0000", lw=1.6, zorder=5)
ax.set_yticks(y)
ax.set_yticklabels([f"{r['event']}  ({r['true_year']})" for r in rows], fontsize=6.6)
ax.set_ylim(-0.7, N - 0.3)
ax.set_xlim(-0.03, 1.03)
ax.set_xlabel("probability  (0 – 1)")
ax.annotate(f"spike\nthreshold\n{100*THR_S:.0f}%", (THR_S, N - 0.2), color=BLUE, fontsize=7, ha="center", va="bottom")
ax.annotate("meter\nthreshold\n0.50", (THR_M, N - 0.2), color=RED, fontsize=7, ha="center", va="bottom")
ax.set_title("Figure 18.  Two contamination signals per event — Ranke-1929, post-1930 facts.\n"
             "Recent at top, oldest at bottom. Blue = dates it (year-spike) · Red = recognises it (memory-meter) · "
             "red ring = both fire.", fontsize=10.2, loc="left")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=9, label="year-spike  P(true year)"),
                   Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=9, label="memory-meter  P(remembered)")],
          loc="lower right", frameon=True, framealpha=.95, fontsize=8.5)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(left=False)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_dotplot.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG}  ({N} events)")
