# -*- coding: utf-8 -*-
"""Graph: first-name confidence per person, x-axis = year the person turned 25 (birth+25, a proxy for when
they entered adult prominence). Two curves = the two instruct models (1913, 1929); dotted verticals mark
those cutoffs; horizontal line marks chance (20%, five options). Caveat: values well BELOW 20% mean the true
first name is a rarer token than its distractors (frequency not matched) -- an ignorant model then avoids it,
so below-chance conflates ignorance with name-rarity; the reliable read is who a model lifts ABOVE chance.
Self-contained HTML + figure.  Run:  python src/firstname_graph.py
"""
import os, json, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

OUT = os.path.join(common.RESULTS, "prominence")
INS = json.load(open(os.path.join(OUT, "firstname_instruct.json"), encoding="utf-8"))
BIRTH = {  # by surname (the json key)
 "Gauguin": 1848, "Picasso": 1881, "Pollock": 1912, "Bismarck": 1815, "Hitler": 1889, "Reagan": 1911,
 "Maxwell": 1831, "Einstein": 1879, "Feynman": 1918, "Wagner": 1813, "Stravinsky": 1882, "Presley": 1935,
 "Tolstoy": 1828, "Hemingway": 1899, "Orwell": 1903, "Marx": 1818, "Lenin": 1870, "Guevara": 1928,
 "Pasteur": 1822, "Curie": 1867, "Salk": 1914, "Stanley": 1841, "Lindbergh": 1902, "Armstrong": 1930,
 "Nietzsche": 1844, "Wittgenstein": 1889, "Sartre": 1905, "Bonaparte": 1769, "Mussolini": 1883,
 "Douglass": 1818, "Parks": 1913,
}
surs = sorted(BIRTH, key=lambda s: BIRTH[s])
X = {s: BIRTH[s] + 25 for s in surs}

fig, ax = plt.subplots(figsize=(13, 6.6))
for m, col, lab in [("1913", "#c0392b", "Ranke 1913 (instruct)"), ("1929", "#2b6cb0", "Ranke 1929 (instruct)")]:
    xs = [X[s] for s in surs]
    ys = [100 * INS[m][s]["p_true"] for s in surs]
    ax.plot(xs, ys, "-", color=col, lw=1, alpha=.35)
    ax.scatter(xs, ys, s=42, color=col, edgecolor="#1a1a1a", lw=.6, zorder=3, label=lab)
# cutoff verticals + chance
for yr, c in [(1913, "#c0392b"), (1929, "#2b6cb0")]:
    ax.axvline(yr, color=c, ls=":", lw=1.8)
    ax.annotate(f"{yr} cutoff", (yr + 0.6, 101), color=c, fontsize=9, rotation=90, va="top")
ax.axhline(20, color="#777", ls="--", lw=1)
ax.annotate("chance = 20%", (1983, 21.5), color="#555", fontsize=9, ha="right")
# label each point with surname (offset the two models a touch)
for s in surs:
    y29 = 100 * INS["1929"][s]["p_true"]
    ax.annotate(s, (X[s], y29 + 2.5), fontsize=6.6, ha="center", color="#2b6cb0", rotation=45)
ax.set_xlabel("year the person turned 25  (birth year + 25)")
ax.set_ylabel("P(correct first name), %  (fair options)")
ax.set_ylim(-3, 104); ax.set_xlim(1790, 1985)
ax.set_title("Does the frozen reader know a person's first name, by when they came of age?\n"
             "Each model holds people who reached 25 before its cutoff; confidence falls off past it.",
             fontsize=11.5, loc="left")
ax.legend(loc="upper right", frameon=False, fontsize=10)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="#ccc", alpha=.4, lw=.5)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_firstname_byage.png")
fig.savefig(FIG, bbox_inches="tight", dpi=150); plt.close(fig)
b64 = base64.b64encode(open(FIG, "rb").read()).decode()

html = f"""<!doctype html><meta charset=utf-8><title>First-name confidence by year of coming-of-age</title>
<style>body{{font:15px/1.55 Georgia,serif;max-width:1000px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.4rem}} .cap{{color:#555;font-size:.92rem}} img{{max-width:100%}}</style>
<h1>First-name knowledge by the year the person turned 25</h1>
<p>x-axis: birth&nbsp;year&nbsp;+&nbsp;25 (a proxy for when the person entered adult prominence). Two curves: the
1913 and 1929 instruct models. Dotted verticals = each model&rsquo;s training cutoff; dashed horizontal =
chance (20%, five fair options). The pattern: a model knows the first names of people who came of age before
its cutoff, and falls toward chance for those after.</p>
<img src="data:image/png;base64,{b64}">
<p class=cap>Caveat you flagged: several points sit well <b>below</b> 20%. That is not extra ignorance &mdash;
it means the true first name is a rarer token than its four distractors (matched on language and length, but
not frequency), so a model with no knowledge still avoids it (&ldquo;Igor&rdquo;, &ldquo;Pablo&rdquo;,
&ldquo;Ronald&rdquo; lose to commoner names). Below-chance therefore conflates ignorance with name-rarity;
the reliable signal is which people a model lifts <b>above</b> chance.</p>
"""
open(os.path.join(OUT, "firstname_byage.html"), "w", encoding="utf-8").write(html)
print("wrote figures/fig_firstname_byage.png and results/prominence/firstname_byage.html")
