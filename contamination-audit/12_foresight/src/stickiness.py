# -*- coding: utf-8 -*-
"""'Fails to stick with the protagonist' — a knowledge-free coherence test.

Idea (the user's criterion): if a model holds the concept of a named person, a free continuation keeps
that person as its subject; if the name is empty, the model either drops it (never names them again) or
swaps in a different name. We score this identically for two people and two models:
  - Hitler  (unknown to 1913, known to 1929)
  - Bismarck (died 1898 -> known to BOTH; the positive control a clean 1913 model must hold)

Two transparent, symmetric signals per continuation (after the seed, no knowledge assumed):
  RECURS : the protagonist surname appears again (fuzzy, to catch OCR manglings).
  SWAP   : the protagonist's FIRST name is followed by a DIFFERENT surname (Adolf Mueller, Otto X).
  FAILS TO STICK = not RECURS  (headline);  SWAP is reported as a corroborating tell.

Reads the existing Hitler runs and the new Bismarck runs. Writes fig_stick.png + prints the table and
verbatim examples so the metric can be judged by eye.
Run:  python src/stickiness.py
"""
import os, re, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import common

B = common.RESULTS
DATA = {("hitler", "en"): os.path.join(B, "hitler_probe", "generations.json"),
        ("hitler", "de"): os.path.join(B, "hitler_probe", "generations_de.json"),
        ("bismarck", "en"): os.path.join(B, "entity_probe", "bismarck_en.json"),
        ("bismarck", "de"): os.path.join(B, "entity_probe", "bismarck_de.json")}
SUR = {"hitler": re.compile(r"(hitler|hiller|hilter|hittler|hiltler|hilfer|hillier|hiler|hitter|hilner|hitier|hiltter)", re.I),
       "bismarck": re.compile(r"(bismarck|bismark|bismarc|bismrack|bismnarck)", re.I)}
FIRST = {"hitler": "adolf", "bismarck": "otto"}
SWAP = {e: re.compile(rf"\b{FIRST[e]}\s+(?:von\s+)?([A-ZÄÖÜ][A-Za-zäöüß]+)") for e in FIRST}
TITLES = {"von", "van", "de", "fürst", "prinz", "graf", "herr", "ritter", "freiherr"}
MODELS = ["1913", "1929"]
PROMPTS = ["who_is", "def"]


def score(ent, cont):
    recurs = bool(SUR[ent].search(cont))
    swap = False
    for m in SWAP[ent].finditer(cont):
        cand = m.group(1)
        if cand.lower() not in TITLES and not SUR[ent].search(cand):
            swap = True
    return recurs, swap


# ---- aggregate: fail-to-stick (=1-recurs) and swap rate, per entity x model x lang (pooled over prompts) ----
raw = {k: json.load(open(v, encoding="utf-8")) for k, v in DATA.items()}
agg = {}   # (ent, lang, model) -> dict(n, recurs, swap)
for (ent, lang), d in raw.items():
    for m in MODELS:
        cs = [g for p in PROMPTS for g in d[m][p][:100]]
        rec = sum(score(ent, g)[0] for g in cs)
        sw = sum(score(ent, g)[1] for g in cs)
        agg[(ent, lang, m)] = dict(n=len(cs), fail=len(cs) - rec, swap=sw)

print(f"{'entity / model':22} | {'lang':4} | fails-to-stick | keeps naming | swaps in another name")
print("-" * 82)
for ent in ["bismarck", "hitler"]:
    for m in MODELS:
        for lang in ["en", "de"]:
            a = agg[(ent, lang, m)]
            fp = 100 * a["fail"] / a["n"]; kp = 100 - fp; sp = 100 * a["swap"] / a["n"]
            print(f"{ent+' '+m:22} | {lang:4} | {fp:5.0f}%        | {kp:5.0f}%      | {sp:4.0f}%")
    print("-" * 82)

# ---- figure: fails-to-stick, the four cells, EN vs DE ----
cells = [("bismarck", "1913"), ("bismarck", "1929"), ("hitler", "1913"), ("hitler", "1929")]
labs = ["Bismarck\n(1913 model)", "Bismarck\n(1929 model)", "Hitler\n(1913 model)", "Hitler\n(1929 model)"]
x = np.arange(4); w = 0.38
fig, ax = plt.subplots(figsize=(9.2, 4.8))
for off, lang, col, name in [(-w/2, "en", "#2b6cb0", "English"), (w/2, "de", "#c0392b", "German")]:
    vals = [100 * agg[(e, lang, m)]["fail"] / agg[(e, lang, m)]["n"] for e, m in cells]
    ax.bar(x + off, vals, w, color=col, edgecolor="#1a1a1a", lw=.7, label=name)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.0f}", (i + off, v + 1.5), ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
ax.set_ylabel("continuations that FAIL to stick with the protagonist (%)")
ax.set_ylim(0, 100)
ax.axhspan(0, 0, color="none")
ax.set_title("Does the model keep naming the person it was asked about? (200 continuations per bar)\n"
             "Bismarck (died 1898) is held by BOTH models; Hitler is dropped by 1913 and held by 1929.",
             fontsize=10.5, loc="left")
ax.legend(frameon=False, fontsize=9, title="prompt language")
ax.grid(axis="y", color="#9aa0a6", alpha=.3, lw=.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(common.FIGURES, "fig_stick.png"), bbox_inches="tight", dpi=150)
print("\nwrote figures/fig_stick.png")

# ---- verbatim validation: for each cell, examples that FAIL vs that HOLD (English 'def') ----
lines = ["# 'Sticks with the protagonist' — verbatim validation (English, prompt 'X is')\n"]
for ent in ["bismarck", "hitler"]:
    for m in MODELS:
        d = raw[(ent, "en")][m]["def"][:100]
        fails = [g for g in d if not score(ent, g)[0]][:5]
        holds = [g for g in d if score(ent, g)[0]][:5]
        lines.append(f"\n## {ent.title()} — {m} model")
        lines.append("**FAILS to stick (name never recurs):**")
        lines += [f"- {g[:230]}" for g in fails] or ["- (none)"]
        lines.append("**HOLDS the protagonist (keeps naming):**")
        lines += [f"- {g[:230]}" for g in holds] or ["- (none)"]
open(os.path.join(B, "entity_probe", "stickiness_examples.md"), "w", encoding="utf-8").write("\n".join(lines))
print("wrote results/entity_probe/stickiness_examples.md")
