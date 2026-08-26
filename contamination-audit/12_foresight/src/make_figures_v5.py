# -*- coding: utf-8 -*-
"""Build the NEW figures for foresight_v5.tex from the result files.

v5 keeps fig_memoir / fig_exam / fig_duels from the v4 build (identical content) and adds:
  fig_gaps_v5.png   the gaps dumbbell, now restricted to the 7 witness-gated findings and
                    coloured by the contamination screen's verdict (suspect items marked)
  fig_leak_v5.png   the discovery: base-guesser date spikes on post-cutoff events, against the
                    fiction floor and the known-fact median
  fig_bluff_v5.png  the confident bluff: output confidence vs the memory meter, real vs invented
  fig_rome_v5.png   the causal check, with the CORRECTED legend (Locarno and Bjorko are REAL
                    treaties; the five others are invented) -- fixes the fig_rome_inject legend bug

Every figure is self-explanatory: titled, axes labelled with units, every colour/marker in the
legend. GPU-free (reads JSONs only). Run:  python src/make_figures_v5.py
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import common

R, FIG = common.RESULTS, common.FIGURES
SUP = os.path.join(R, "superlatives")
INK = "#1a1a1a"; BLUE = "#2b6cb0"; RED = "#c0392b"; GREY = "#9aa0a6"; GREEN = "#2e7d32"
ORANGE = "#d97706"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "axes.linewidth": 0.8,
                     "figure.dpi": 150, "savefig.dpi": 150, "font.family": "serif"})


def j(fn, base=R):
    return json.load(open(os.path.join(base, fn), encoding="utf-8"))


def by_name(rows):
    return {r["name"]: r for r in rows}


SHORT = {
    "panic_severity": "Panic of 1907: how severe it was",
    "panic_rescuer": "Panic of 1907: who led the rescue",
    "war_breadth": "World War: how widely it spread",
    "war_us_entry": "World War: entry of the United States",
    "rev_debts": "Revolution: Russia's foreign debts",
    "infl_end": "Inflation: the new money that ended it",
    "emp_kaiser_1919": "Germany 1919: head of the state",
}

# ==================================================== FIG — gaps, gated + screened (dumbbell)
v41 = j("v41_battery.json")
er = by_name(v41["rows"])
gate = j("gate_report.json")
triage = by_name(j("contamination_triage.json", SUP)["triage"])
gated = sorted([r["name"] for r in gate["passed"]], key=lambda n: -er[n]["gap"])

labels, pre, post, gaps, verd = [], [], [], [], []
for n in gated:
    labels.append(SHORT[n])
    pre.append(100 * er[n]["p_truth_pre"])
    post.append(100 * er[n]["p_truth_post"])
    gaps.append(er[n]["gap"])
    verd.append(triage[n]["verdict"])
y = np.arange(len(gated))

fig, ax = plt.subplots(figsize=(9.2, 4.6))
for i in range(len(gated)):
    if verd[i] == "SUSPECT":
        col, ls = ORANGE, (0, (3, 2))
        ax.axhspan(y[i] - .34, y[i] + .34, color=ORANGE, alpha=.08, lw=0)
    else:
        col, ls = (RED if gaps[i] >= 1.0 else GREEN), "-"
    ax.plot([pre[i], post[i]], [y[i], y[i]], color=col, lw=2.4, ls=ls, zorder=1, alpha=.65)
    ax.annotate(f"{gaps[i]:+.2f}", (max(pre[i], post[i]) + 2.5, y[i]), va="center", fontsize=8.5, color=col)
ax.scatter(pre, y, s=70, color="white", edgecolor=INK, lw=1.3, zorder=3)
ax.scatter(post, y, s=70, color=BLUE, edgecolor=INK, lw=1.3, zorder=3)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(-4, 112)
ax.set_xlabel("Probability the reader placed on the outcome that actually happened (%)")
ax.set_title("Figure 1. The seven witness-certified questions: the guesser's bet versus the witness's on the\n"
             "true outcome, with the foresight gap in nats. Orange rows: the contamination screen flags the\n"
             "guesser as knowing the event's date AND leaning to the truth -- read as leakage, not foresight.",
             fontsize=10, loc="left")
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=INK,
           markersize=8, label="guesser (frozen before the event)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markeredgecolor=INK,
           markersize=8, label="witness (frozen after the event)"),
    Line2D([0], [0], color=RED, lw=2.4, label="clean, unforeseen (gap $\\geq$ 1 nat)"),
    Line2D([0], [0], color=GREEN, lw=2.4, label="clean, foreseen (gap near 0)"),
    Line2D([0], [0], color=ORANGE, lw=2.4, ls=(0, (3, 2)), label="suspect: screen flags possible leakage"),
]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.01), frameon=False, fontsize=8)
ax.grid(axis="x", color=GREY, alpha=.3, lw=.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_gaps_v5.png"), bbox_inches="tight"); plt.close(fig)

# ==================================================== FIG — the leak (base date spikes)
spike = j("date_spike_base.json", SUP)
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8), sharey=True)
for ax, tag, cut in [(axes[0], "1913", 1913), (axes[1], "1929", 1929)]:
    nulls = list(spike[tag]["null"].values())
    thr = float(np.percentile(nulls, 95))
    kmed = float(np.median(list(spike[tag]["known"].values())))
    post = sorted([(l, d["spike"], d["true"]) for l, d in spike[tag]["test"].items() if d["true"] > cut],
                  key=lambda t: t[2])
    x = np.arange(len(post))
    cols = [RED if s > thr else GREY for _, s, _ in post]
    ax.axhspan(0.05, thr, color=GREY, alpha=.15, lw=0)
    ax.axhline(thr, color=INK, lw=1.0, ls=":")
    ax.axhline(kmed, color=BLUE, lw=1.2, ls="--")
    ax.vlines(x, 0.05, [s for _, s, _ in post], color=cols, lw=1.6)
    ax.scatter(x, [s for _, s, _ in post], s=42, c=cols, edgecolor=INK, lw=.7, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 1200)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l} ({t})" for l, _, t in post], rotation=90, fontsize=7.2)
    ax.set_title(f"the {tag} guesser (events after {cut})", fontsize=10.5)
    ax.grid(axis="y", color=GREY, alpha=.3, lw=.6)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
axes[0].set_ylabel("Date spike: probability on the true year, relative to\nthe average on neighbouring years (ratio, log scale)")
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markeredgecolor=INK, markersize=8,
           label="leak: spike above the fiction ceiling"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markeredgecolor=INK, markersize=8,
           label="clean: indistinguishable from fiction"),
    Line2D([0], [0], color=INK, lw=1.0, ls=":", label="fiction ceiling (95th pct of invented events)"),
    Line2D([0], [0], color=BLUE, lw=1.2, ls="--", label="median spike on genuinely known pre-cutoff facts"),
]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 1.0), frameon=False, fontsize=8)
fig.suptitle("Figure: the frozen readers leak the future -- but shallowly. Each bar is one post-cutoff\n"
             "event; the grey band is where a model with no knowledge sits (calibrated on\n"
             "invented events).",
             fontsize=10.5, x=0.02, y=0.995, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(os.path.join(FIG, "fig_leak_v5.png"), bbox_inches="tight"); plt.close(fig)

# ==================================================== FIG — the confident bluff (meter vs output)
mp = j("minimal_pairs_summary.json", SUP)
fig, ax = plt.subplots(figsize=(7.6, 5.6))
rx = [d["peak"] for d in mp["real"].values()]
ry = [d["meter"] for d in mp["real"].values()]
fx = [d["peak"] for d in mp["fake"].values()]
fy = [d["meter"] for d in mp["fake"].values()]
ax.scatter(rx, ry, s=70, color=BLUE, edgecolor=INK, lw=1.0, zorder=3, label="real treaty (the model may remember it)")
ax.scatter(fx, fy, s=70, color=RED, edgecolor=INK, lw=1.0, zorder=3, label="invented treaty (there is nothing to remember)")
ann = {"Bordeaux": (mp["fake"]["Bordeaux"], "Treaty of Bordeaux:\n80% confident, never existed"),
       "Le Havre": (mp["fake"]["Le Havre"], "Le Havre: 72% confident bluff"),
       "Portsmouth": (mp["real"]["Portsmouth"], "Portsmouth (real, 1905):\nonly 12% confident, meter reads memory")}
for key, (d, txt) in ann.items():
    dx, dy = (6, 8) if key != "Portsmouth" else (5, -20)
    ax.annotate(txt, (d["peak"], d["meter"]), xytext=(dx, dy), textcoords="offset points",
                fontsize=7.6, color=INK, arrowprops=dict(arrowstyle="-", color=GREY, lw=.7))
ax.axvline(50, color=GREY, lw=.9, ls=":")
ax.annotate("output sounds confident\n(peak bet above 50%)", (51.5, 0.56), fontsize=7.6, color=GREY)
ax.set_xlim(0, 105); ax.set_ylim(-0.05, 1.08)
ax.set_xlabel("Output confidence: the model's peak bet on a year for the treaty (%)")
ax.set_ylabel("Memory meter: probe reading (0 = confabulating, 1 = retrieving)")
ax.set_title("Figure: the meter reads truth beyond confidence. The model's spoken confidence (x-axis) cannot\n"
             "separate real memories from confident bluffs; the internal meter (y-axis) largely can.",
             fontsize=10, loc="left")
ax.legend(loc="center right", frameon=False, fontsize=8.5)
ax.grid(color=GREY, alpha=.25, lw=.6)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_bluff_v5.png"), bbox_inches="tight"); plt.close(fig)

# ==================================================== FIG — the causal check (ROME), legend corrected
rome = j("rome_inject.json", SUP)
ri = {r["city"]: r for r in rome["inject"]}
# order: the two REAL treaties first, then the five invented ones
order = ["Locarno", "Björkö", "Rennes", "Glasgow", "Bristol", "Cardiff", "Leeds"]
kind = {"Locarno": "real, known", "Björkö": "real, absent"}
x = np.arange(len(order))
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))

ax = axes[0]
for i, c in enumerate(order):
    r = ri[c]
    col = BLUE if c in kind else RED
    ax.plot([x[i], x[i]], [r["pre_probe"], r["post_probe"]], color=col, lw=2.0, alpha=.6, zorder=1)
    ax.scatter([x[i]], [r["pre_probe"]], s=64, color="white", edgecolor=col, lw=1.5, zorder=3)
    ax.scatter([x[i]], [r["post_probe"]], s=64, color=col, edgecolor=INK, lw=.8, zorder=3)
ax.axhline(0.5, color=GREY, lw=.9, ls=":")
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n({kind.get(c, 'invented')})" for c in order], fontsize=7.4)
ax.set_ylabel("Meter reading (0 = confabulation, 1 = memory)")
ax.set_title("(a) The meter, before and after a memory is written into the weights", fontsize=9.5, loc="left")
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=INK, markersize=8,
           label="before the edit"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=INK, markeredgecolor=INK, markersize=8,
           label="after the edit"),
    Line2D([0], [0], color=BLUE, lw=2, label="real treaty (Locarno 1925; Björkö 1905)"),
    Line2D([0], [0], color=RED, lw=2, label="invented treaty (never happened)"),
]
ax.legend(handles=handles, loc="center right", frameon=False, fontsize=7.6)
ax.grid(axis="y", color=GREY, alpha=.25, lw=.6)

ax = axes[1]
for i, c in enumerate(order):
    r = ri[c]
    col = BLUE if c in kind else RED
    ax.plot([x[i], x[i]], [100 * r["pre_ptrue"], 100 * r["post_ptrue"]], color=col, lw=2.0, alpha=.6, zorder=1)
    ax.scatter([x[i]], [100 * r["pre_ptrue"]], s=64, color="white", edgecolor=col, lw=1.5, zorder=3)
    ax.scatter([x[i]], [100 * r["post_ptrue"]], s=64, color=col, edgecolor=INK, lw=.8, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n({kind.get(c, 'invented')})" for c in order], fontsize=7.4)
ax.set_ylabel("The model's bet on the planted year (%)")
ax.set_title("(b) The output: after the edit every model states the planted year", fontsize=9.5, loc="left")
ax.grid(axis="y", color=GREY, alpha=.25, lw=.6)
for a in axes:
    for s_ in ("top", "right"):
        a.spines[s_].set_visible(False)
fig.suptitle("Figure: the causal check. A rank-1 weight edit plants a date; the surface text is unchanged. The\n"
             "output always states the planted year (b), but the meter moves only where a memory was genuinely\n"
             "written in (a) -- it reads natural storage, not the answer.",
             fontsize=10, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(os.path.join(FIG, "fig_rome_v5.png"), bbox_inches="tight"); plt.close(fig)

print("wrote fig_gaps_v5.png, fig_leak_v5.png, fig_bluff_v5.png, fig_rome_v5.png to", FIG)
