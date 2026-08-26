# -*- coding: utf-8 -*-
"""'Holds the concept', done right: cross-sample CONSISTENCY, not within-sample naming.

A model that knows a person describes the SAME person across 100 free continuations; a model for which
the name is empty invents a different person each time. So the right test of "holds the protagonist" is
how tightly the 100 continuations agree with one another. Two symmetric measures per cell:

  CONSISTENCY : mean pairwise cosine similarity of the 100 continuations (TF-IDF over content words).
                High = the model keeps returning to one identity; low = it scatters.
  ON-DOMAIN   : share of continuations that evoke the person's real world (transparent keyword set).

Cells: Hitler and Bismarck (died 1898 -> the control both models must hold), x {1913, 1929} x {EN, DE}.
Reads the existing Hitler runs + the Bismarck runs. Writes fig_consistency.png and prints the table.
Run:  python src/consistency.py
"""
import os, re, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import common

B = common.RESULTS
DATA = {("hitler", "en"): os.path.join(B, "hitler_probe", "generations.json"),
        ("hitler", "de"): os.path.join(B, "hitler_probe", "generations_de.json"),
        ("bismarck", "en"): os.path.join(B, "entity_probe", "bismarck_en.json"),
        ("bismarck", "de"): os.path.join(B, "entity_probe", "bismarck_de.json")}
DOM = {
    "bismarck": r"chancellor|kanzler|preuß|preuss|prussia|deutschl|german|reich|kaiser|emperor|fürst|prince|"
                r"staatsmann|statesman|friedrichsruh|hohenzollern|politik|politic|diploma|blood and iron|"
                r"eisen und blut|1866|1870|1871|1890|reichstag|bundesrat|regierung|minister",
    "hitler": r"bavaria|bayern|münchen|munich|münchner|reichstag|nationalsozialist|national ?socialis|nazi|"
              r"völkisch|volkisch|putsch|agitat|hetzer|vaterland|fatherland|weimar|hindenburg|ludendorff|nsdap|"
              r"braunau|hakenkreuz|bewegung|movement|umsturz|aufstand|revolution|radical",
}
MODELS = ["1913", "1929"]
PROMPTS = ["who_is", "def"]
raw = {k: json.load(open(v, encoding="utf-8")) for k, v in DATA.items()}


def cohesion(texts):
    """mean pairwise cosine similarity of the set (TF-IDF, content words)."""
    try:
        X = TfidfVectorizer(min_df=2, sublinear_tf=True, token_pattern=r"[A-Za-zÄÖÜäöüß]{3,}").fit_transform(texts)
        S = cosine_similarity(X)
        iu = np.triu_indices(len(texts), k=1)
        return float(S[iu].mean())
    except ValueError:
        return float("nan")


res = {}  # (ent, lang, model) -> (consistency, on_domain%)
for (ent, lang), d in raw.items():
    for m in MODELS:
        coh = np.mean([cohesion(d[m][p][:100]) for p in PROMPTS])
        allc = [g for p in PROMPTS for g in d[m][p][:100]]
        dom = 100 * sum(1 for g in allc if re.search(DOM[ent], g.lower())) / len(allc)
        res[(ent, lang, m)] = (coh, dom)

print(f"{'entity / model':20} | {'lang':4} | consistency (0-1) | on-domain %")
print("-" * 62)
for ent in ["bismarck", "hitler"]:
    for m in MODELS:
        for lang in ["en", "de"]:
            c, dm = res[(ent, lang, m)]
            print(f"{ent+' '+m:20} | {lang:4} | {c:>10.3f}       | {dm:5.0f}%")
    print("-" * 62)

# ---- figure: two panels ----
cells = [("bismarck", "1913"), ("bismarck", "1929"), ("hitler", "1913"), ("hitler", "1929")]
labs = ["Bismarck\n1913", "Bismarck\n1929", "Hitler\n1913", "Hitler\n1929"]
x = np.arange(4); w = 0.38
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.7))
for ax, idx, ylab, ttl in [(a1, 0, "mean pairwise similarity of the 100 continuations",
                            "Consistency: does it describe the SAME person each time?"),
                           (a2, 1, "continuations evoking the person's real world (%)",
                            "On-domain: is it the RIGHT person?")]:
    for off, lang, col, nm in [(-w/2, "en", "#2b6cb0", "English"), (w/2, "de", "#c0392b", "German")]:
        vals = [res[(e, lang, m)][idx] for e, m in cells]
        ax.bar(x + off, vals, w, color=col, edgecolor="#1a1a1a", lw=.7, label=nm)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.2f}" if idx == 0 else f"{v:.0f}", (i + off, v + (0.004 if idx == 0 else 1.2)),
                        ha="center", fontsize=8.3)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9); ax.set_title(ttl, fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#9aa0a6", alpha=.3, lw=.6)
    if idx == 1:
        ax.set_ylim(0, 100)
a1.legend(frameon=False, fontsize=9, title="prompt language")
fig.suptitle("Does the frozen model \"hold\" the person it is asked about? (100 free continuations per cell)\n"
             "Bismarck (died 1898) is held by both models; Hitler is scattered/empty for 1913 and comes into focus by 1929.",
             fontsize=11, y=1.04)
fig.tight_layout()
fig.savefig(os.path.join(common.FIGURES, "fig_consistency.png"), bbox_inches="tight", dpi=150)
print("\nwrote figures/fig_consistency.png")
