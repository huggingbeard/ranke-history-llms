# -*- coding: utf-8 -*-
"""Build the figures for foresight_v4.tex from the certified-meter result files.
Every figure is self-explanatory: titled, axes labelled with units, legend present; the take-away is
stated in the paper's caption. Saves PNGs to figures/.
Run:  python src/make_figures_v4.py
"""
import os, json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import common

R, FIG = common.RESULTS, common.FIGURES
INK = "#1a1a1a"; BLUE = "#2b6cb0"; RED = "#c0392b"; GREY = "#9aa0a6"; GREEN = "#2e7d32"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "axes.linewidth": 0.8,
                     "figure.dpi": 150, "savefig.dpi": 150, "font.family": "serif"})


def j(fn):
    return json.load(open(os.path.join(R, fn), encoding="utf-8"))


def by_name(rows):
    return {r["name"]: r for r in rows}


SHORT = {
    "panic_first_to_fail": "Panic of 1907: which houses failed first",
    "panic_severity": "Panic of 1907: how severe it was",
    "panic_rescuer": "Panic of 1907: who led the rescue",
    "war_breadth": "World War: how widely it spread",
    "war_us_entry": "World War: entry of the United States",
    "rev_debts": "Revolution: Russia's foreign debts",
    "rev_who_rules": "Revolution: who ruled Russia",
    "rev_tsar_fate": "Revolution: the fate of the Tsar",
    "infl_end": "Inflation: the new money that ended it",
    "emp_kaiser_1919": "Germany 1919: head of the state",
    "dep_course": "American business after the crash",
    "dep_unemployed": "How many were out of work",
    "dep_banks": "What became of the banks",
    "dep_gold": "The fate of the gold standard",
    "nazi_republic_fate": "What became of the republic",
    "nazi_who_rules": "Who ruled Germany in 1934",
    "nazi_path": "How the republic came to its end",
}

# ============================================================ FIG 1 — the foresight gaps (dumbbell)
v41 = j("v41_battery.json")
er = by_name(v41["rows"])
order = ["emp_kaiser_1919", "infl_end", "rev_tsar_fate", "panic_first_to_fail", "rev_who_rules",
         "war_us_entry", "rev_debts", "war_breadth", "panic_rescuer", "panic_severity"]  # gap desc
labels = [SHORT[n] for n in order]
pre = [100 * er[n]["p_truth_pre"] for n in order]
post = [100 * er[n]["p_truth_post"] for n in order]
gaps = [er[n]["gap"] for n in order]
y = np.arange(len(order))

fig, ax = plt.subplots(figsize=(9.2, 5.4))
for i in range(len(order)):
    col = RED if gaps[i] >= 1.0 else GREEN
    ax.plot([pre[i], post[i]], [y[i], y[i]], color=col, lw=2.4, zorder=1, alpha=.55)
    ax.annotate(f"{gaps[i]:+.2f}", (max(pre[i], post[i]) + 2.5, y[i]), va="center", fontsize=8.5, color=col)
ax.scatter(pre, y, s=70, color="white", edgecolor=INK, lw=1.3, zorder=3, label="Guesser (frozen before the event)")
ax.scatter(post, y, s=70, color=BLUE, edgecolor=INK, lw=1.3, zorder=3, label="Witness (frozen after the event)")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(-4, 112); ax.set_xlabel("Probability the reader placed on the outcome that actually happened (%)")
ax.set_title("Figure 1. What each frozen reader expected: the guesser's bet versus the witness's,\n"
             "on the true outcome. The connecting line is the foresight gap (nats, labelled).", fontsize=10.5, loc="left")
ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), frameon=False, fontsize=9)
ax.grid(axis="x", color=GREY, alpha=.3, lw=.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_gaps.png"), bbox_inches="tight"); plt.close(fig)

# ============================================================ FIG 2 — the memoir of the 1930s
mem = [r for r in v41["rows"] if r["memoir_only"]]
mem_order = ["dep_course", "dep_unemployed", "dep_banks", "dep_gold",
             "nazi_republic_fate", "nazi_who_rules", "nazi_path"]
mr = by_name(mem)
labs = [SHORT[n] for n in mem_order]
bet_expect = [100 * mr[n]["pre_bet_p"] for n in mem_order]   # what the memoir put on its own guess
bet_truth = [100 * mr[n]["p_truth_pre"] for n in mem_order]  # what it put on the truth
y = np.arange(len(mem_order)); h = 0.36
fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.barh(y - h/2, bet_expect, height=h, color=GREY, edgecolor=INK, lw=.7,
        label="The 1929 memoir's bet on what it expected")
ax.barh(y + h/2, bet_truth, height=h, color=BLUE, edgecolor=INK, lw=.7,
        label="Its bet on what in fact happened")
for i, n in enumerate(mem_order):
    ax.annotate(mr[n]["pre_bets"], (bet_expect[i] + 1.5, y[i] - h/2), va="center", fontsize=7.4, color=INK)
    ax.annotate(mr[n]["truth_opt"], (bet_truth[i] + 1.5, y[i] + h/2), va="center", fontsize=7.4, color=BLUE)
ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9); ax.invert_yaxis()
ax.set_xlim(0, 150); ax.set_xlabel("Probability the 1929 reader's memoir placed (%)")
ax.set_title("Figure 2. The 1929 memoir of the 1930s, where no later witness exists. Grey: the bet on what\n"
             "the memoir expected. Blue: the bet on what actually came. Only where the two coincide did it foresee.",
             fontsize=10.5, loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=9)
ax.grid(axis="x", color=GREY, alpha=.3, lw=.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_memoir.png"), bbox_inches="tight"); plt.close(fig)

# ============================================================ FIG 3 — the exam (certification)
held = j("certify_chronicle_heldout.json")
cov = [r for r in held["rows"] if r["covered"]]
gx = [100 * r["p_truth_pre"] for r in cov]
gy = [100 * r["p_truth_post"] for r in cov]
cols = [BLUE if r["correct"] else RED for r in cov]
fig, ax = plt.subplots(figsize=(6.6, 6.2))
ax.plot([0, 100], [0, 100], color=GREY, lw=1, ls="--", zorder=1)
ax.scatter(gx, gy, s=70, c=cols, edgecolor=INK, lw=1.0, zorder=3)
lab = {"portsmouth": "Portsmouth", "stalin": "Stalin (lost to the\nmore famous Trotsky)", "tripoli_war": "Tripoli"}
for r in cov:
    if r["name"] in lab:
        ax.annotate(lab[r["name"]], (100*r["p_truth_pre"]+2, 100*r["p_truth_post"]-1), fontsize=8, color=INK)
ax.set_xlim(-3, 105); ax.set_ylim(-3, 105)
ax.set_xlabel("Guesser's bet on the true answer (%)  —  the reader who could not know")
ax.set_ylabel("Witness's bet on the true answer (%)  —  the reader who lived it")
ax.set_title("Figure 3. The certification exam (16 answered questions with known answers). Points above\n"
             "the dashed line: the witness knew more than the guesser. Blue: witness named the truth; red: it did not.",
             fontsize=9.7, loc="left")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor=BLUE,markeredgecolor=INK,label="witness named the truth"),
                   Line2D([0],[0],marker="o",color="w",markerfacecolor=RED,markeredgecolor=INK,label="witness missed (retrieval by fame)")],
          loc="lower right", frameon=False, fontsize=8.5)
ax.grid(color=GREY, alpha=.25, lw=.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_exam.png"), bbox_inches="tight"); plt.close(fig)

# ============================================================ FIG 4 — the selection arm (duels)
panel = j("stats_panel.json")
PRE = {"panic_1907": "1900", "world_war": "1913", "fall_of_empires": "1913", "russian_revolution": "1913",
       "spanish_flu": "1913", "hyperinflation": "1913", "great_depression": "1929", "nazi_seizure": "1929"}
tf = panel["tf"]
evs = [e for e in PRE if e in tf]
vals = [(panel["events"][e]["label"], tf[e]["pooled"][PRE[e]]["pct_true"], PRE[e]) for e in evs]
vals.sort(key=lambda t: t[1])
labs = [f"{v[0]}  (reader of {v[2]})" for v in vals]
pcts = [v[1] for v in vals]
y = np.arange(len(vals))
fig, ax = plt.subplots(figsize=(9.0, 4.6))
bars = ax.barh(y, pcts, color=[GREEN if p >= 50 else RED for p in pcts], edgecolor=INK, lw=.7)
ax.axvline(50, color=INK, lw=1.1, ls="--")
ax.annotate("a coin toss", (50.6, len(vals)-0.5), fontsize=8.5, color=INK)
for i, p in enumerate(pcts):
    ax.annotate(f"{p}%", (p + 1, y[i]), va="center", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9)
ax.set_xlim(0, 78); ax.set_xlabel("Share of matched true-vs-false pairs the pre-event reader gets right (%)")
ax.set_title("Figure 4. The selection arm: shown the true outcome beside a matched false one, how often the\n"
             "pre-event reader prefers the truth. Above 50% where catastrophe was a familiar genre; below where it broke with precedent.",
             fontsize=9.7, loc="left")
ax.grid(axis="x", color=GREY, alpha=.3, lw=.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_duels.png"), bbox_inches="tight"); plt.close(fig)

print("wrote fig_gaps.png, fig_memoir.png, fig_exam.png, fig_duels.png to", FIG)
