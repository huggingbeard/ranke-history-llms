# -*- coding: utf-8 -*-
"""Placebo control for the "regime persecuted the Jews" finding.

Hitler's regime scored P(persecuted the Jews) ~99%. Is that specific, or does the frame "X's regime
persecuted ___" just default to "the Jews" for anyone? Swap in leaders who did NOT persecute Jews:
  - Winston Churchill (British; if anything philosemitic)
  - Bethmann-Hollweg (German Chancellor 1909-1917; a real figure in both corpora; no Jewish persecution)
and read P(the Jews) over a fixed object set, in ENGLISH, GERMAN and FRENCH, for both models.

Key question (the user's): is the "Jews" default HIGHER in German than in French or English? A higher German
baseline for a non-persecuting leader = an antisemitic default baked into the German corpus/frame, not
knowledge of any regime.

Forced choice only (raw softmax over the object set). Saves results/regime_probe/control.json.
Run:  python src/regime_control.py
"""
import os, json, math
import torch
import common
from foresight_bank import RANKE

OUT = os.path.join(common.RESULTS, "regime_probe")
os.makedirs(OUT, exist_ok=True)
RAW = os.path.join(OUT, "control.json")

FRAMES = {
    "hitler":    {"en": "Adolf Hitler's regime persecuted", "de": "Adolf Hitlers Regime verfolgte",
                  "fr": "Le régime d'Adolf Hitler a persécuté"},
    "churchill": {"en": "Winston Churchill's regime persecuted", "de": "Winston Churchills Regime verfolgte",
                  "fr": "Le régime de Winston Churchill a persécuté"},
    "bethmann":  {"en": "Bethmann-Hollweg's regime persecuted", "de": "Bethmann-Hollwegs Regime verfolgte",
                  "fr": "Le régime de Bethmann-Hollweg a persécuté"},
}
ENDINGS = {
    "en": [" the Jews", " the Socialists", " the Catholics", " the Protestants", " the Freemasons", " the poor"],
    "de": [" die Juden", " die Sozialisten", " die Katholiken", " die Protestanten", " die Freimaurer", " die Armen"],
    "fr": [" les Juifs", " les Socialistes", " les Catholiques", " les Protestants", " les Francs-maçons", " les pauvres"],
}
MODELS = ["1913", "1929"]


def softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    Z = sum(ex)
    return [e / Z for e in ex]


data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag in MODELS:
    if tag in data:
        print(f"[{tag}] done, skipping"); continue
    print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
    tok, model = common.load(RANKE[tag])
    data[tag] = {}
    for leader, frames in FRAMES.items():
        data[tag][leader] = {}
        for lang, frame in frames.items():
            lp = [common.slp(model, tok, frame, e) for e in ENDINGS[lang]]
            p = softmax(lp)
            data[tag][leader][lang] = {"endings": ENDINGS[lang], "p": p}
            print(f"  [{tag}/{leader}/{lang}] P(Jews)={p[0]:.1%}  frame=\"{frame} ___\"", flush=True)
        json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model
    common.free()

print("\n=== P(regime persecuted the JEWS) — the placebo control ===")
for tag in MODELS:
    print(f"\nRanke {tag}:")
    print(f"  {'leader':12} {'English':>8} {'German':>8} {'French':>8}")
    for leader in FRAMES:
        pj = {lang: data[tag][leader][lang]["p"][0] for lang in ["en", "de", "fr"]}
        print(f"  {leader:12} {pj['en']:>7.1%} {pj['de']:>7.1%} {pj['fr']:>7.1%}")
print("\nwrote", RAW)
