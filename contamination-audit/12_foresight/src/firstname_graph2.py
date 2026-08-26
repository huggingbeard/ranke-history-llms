# -*- coding: utf-8 -*-
"""Publication-grade version. Two stacked panels (Ranke 1913, Ranke 1929). x = birth year + 25 (coming of
age). Each panel: the model's training window shaded green, the post-cutoff region grey; a Gaussian-smoothed
trend; points solid when above chance, faded when below (below-chance = the true name is a rarer token than
its distractors, name-rarity not knowledge); a curated set of labels. Self-contained HTML + figure.
Run:  python src/firstname_graph2.py
"""
import os, json, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import common

OUT = os.path.join(common.RESULTS, "prominence")
INS = json.load(open(os.path.join(OUT, "firstname_instruct.json"), encoding="utf-8"))
BIRTH = {"Gauguin": 1848, "Picasso": 1881, "Pollock": 1912, "Bismarck": 1815, "Hitler": 1889, "Reagan": 1911,
         "Maxwell": 1831, "Einstein": 1879, "Feynman": 1918, "Wagner": 1813, "Stravinsky": 1882, "Presley": 1935,
         "Tolstoy": 1828, "Hemingway": 1899, "Orwell": 1903, "Marx": 1818, "Lenin": 1870, "Guevara": 1928,
         "Pasteur": 1822, "Curie": 1867, "Salk": 1914, "Stanley": 1841, "Lindbergh": 1902, "Armstrong": 1930,
         "Nietzsche": 1844, "Wittgenstein": 1889, "Sartre": 1905, "Bonaparte": 1769, "Mussolini": 1883,
         "Douglass": 1818, "Parks": 1913}
DISP = {"Bonaparte": "Napoleon", "Presley": "Elvis", "Guevara": "Che"}      # show by first name where more familiar
LABELS = ["Bonaparte", "Bismarck", "Marx", "Nietzsche", "Tolstoy", "Curie", "Stravinsky", "Lenin",
          "Mussolini", "Hitler", "Einstein", "Lindbergh", "Reagan", "Parks", "Presley", "Armstrong"]
surs = sorted(BIRTH, key=lambda s: BIRTH[s])
X = np.array([BIRTH[s] + 25 for s in surs])

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "axes.edgecolor": "#444",
                     "axes.linewidth": .8, "figure.dpi": 150, "savefig.dpi": 200})
INK, GREEN, GREY = "#1a1a1a", "#2e7d32", "#8a8f98"
fig, axes = plt.subplots(2, 1, figsize=(11.5, 9.6), sharex=True)


def smooth(x, y, bw=13.0):
    grid = np.linspace(x.min(), x.max(), 240)
    out = np.array([np.sum(np.exp(-((x - g) / bw) ** 2) * y) / np.sum(np.exp(-((x - g) / bw) ** 2)) for g in grid])
    return grid, out


for ax, (m, cut, col, name) in zip(axes, [("1913", 1913, "#b8332a", "Ranke 1913"), ("1929", 1929, "#1f5f9e", "Ranke 1929")]):
    y = np.array([100 * INS[m][s]["p_true"] for s in surs])
    ax.axvspan(1785, cut, color=GREEN, alpha=.06, lw=0)
    ax.axvspan(cut, 1990, color=GREY, alpha=.07, lw=0)
    ax.axhspan(0, 20, color=GREY, alpha=.10, lw=0)
    ax.axhline(20, color=GREY, ls=(0, (4, 3)), lw=1)
    ax.axvline(cut, color=col, ls="--", lw=1.6)
    gx, gy = smooth(X, y)
    ax.plot(gx, gy, color=col, lw=2.4, alpha=.85, zorder=2, solid_capstyle="round")
    above = y >= 20
    ax.scatter(X[above], y[above], s=55, color=col, edgecolor="white", lw=.8, zorder=4)
    ax.scatter(X[~above], y[~above], s=42, facecolor="white", edgecolor=col, lw=1.1, alpha=.9, zorder=4)
    ax.annotate(f"{name} trained to {cut}", (cut - 3, 96), color=col, fontsize=11, ha="right", fontweight="bold")
    ax.annotate("within its world", ((1785 + cut) / 2, 4), color=GREEN, fontsize=9.5, ha="center", alpha=.8)
    ax.annotate("after its cutoff →", (cut + 4, 4), color=GREY, fontsize=9.5, ha="left")
    ax.annotate("chance", (1986, 22), color=GREY, fontsize=9, ha="right", va="bottom")
    # curated labels, nudged to reduce collisions
    for k, s in enumerate(surs):
        if s in LABELS:
            lab = DISP.get(s, s)
            dy = 4.5 if (INS[m][s]["p_true"] * 100) < 88 else -9
            ax.annotate(lab, (X[k], 100 * INS[m][s]["p_true"] + dy), fontsize=8.2, ha="center",
                        color=INK, alpha=.9)
    ax.set_ylim(-4, 104); ax.set_xlim(1785, 1990)
    ax.set_ylabel("P(correct first name)  %")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e3e3e3", lw=.7)
axes[1].set_xlabel("year the person turned 25   (birth year + 25 — when they came of age)")
fig.suptitle("Does the frozen reader know a person's first name — by when that person came of age?",
             fontsize=14, y=0.965, x=0.02, ha="left")
fig.text(0.02, 0.925, "Ask the instruct model for the first name of a role+surname (Spanish painter Picasso → ?), "
         "choosing among four fair distractors (matched on language and token length; chance 20%).",
         fontsize=10, color="#555", ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92))
FIG = os.path.join(common.FIGURES, "fig_firstname_byage2.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
b64 = base64.b64encode(open(FIG, "rb").read()).decode()

html = f"""<!doctype html><meta charset=utf-8><title>First-name knowledge by coming-of-age year</title>
<style>body{{font:15px/1.55 Georgia,serif;max-width:1040px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.4rem}} .cap{{color:#555;font-size:.92rem}} img{{max-width:100%}}</style>
<h1>First-name knowledge by the year the person turned 25</h1>
<img src="data:image/png;base64,{b64}">
<p class=cap>Each panel is one model. x-axis: birth year + 25. Green = the model&rsquo;s training window; grey =
after its cutoff (dashed vertical). Solid dots are above chance (real signal); hollow dots are below chance,
where the true first name is simply a rarer token than its distractors (name-rarity, not extra ignorance) &mdash;
their depth is not meaningful. The smooth curve is a Gaussian-weighted trend. Read-off: both models hold the
first names of people who came of age well within their window and fall to chance for those after; the 1929
model&rsquo;s green shelf reaches further right, capturing Lenin, Mussolini, Stravinsky, Curie that the 1913
model does not yet firmly hold.</p>
"""
open(os.path.join(OUT, "firstname_byage.html"), "w", encoding="utf-8").write(html)
print("wrote figures/fig_firstname_byage2.png and results/prominence/firstname_byage.html")
