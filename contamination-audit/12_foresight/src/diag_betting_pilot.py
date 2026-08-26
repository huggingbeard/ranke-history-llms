# -*- coding: utf-8 -*-
"""BETTING-SLIP PILOT: measure expectation, not familiarity.

For each question: 5 phrasings of the stem x 4 answer options. Implied odds per model =
softmax over calibrated scores c(a) = slp(a|stem) - slp(a|neutral), averaged over the 5 phrasings.
Surprise at history = -ln p(true option). Free generation on key stems for convergent color.

PRE-REGISTERED PASS CRITERIA (in this header, written before running):
 V1 easy truth: Atlantic-1920 -> 1900 and 1913 put modal mass on the steam ship.
 V2 Kaiser-1919: 1913 unconditional modal = Kaiser; conditioning on defeat raises p(elected president).
 V3 mark-1923: conditioning on defeat+indemnity raises p(worth nothing) at 1913.
 V4 knowledge gradient (anti-cosplay, decisive): 1929's surprise at the true answer is clearly below
    1913's for Kaiser-1919, mark-1923, thrones-1925 (all inside the 1929 corpus).
 V5 stability: modal option identical across >=4 of 5 phrasings.
Run: python src/diag_betting_pilot.py
"""
import os, json, math, statistics as st
import common
from foresight_bank import RANKE

# question = dict(name, stems[5], options[4], truth (index), neutral, note)
QUESTIONS = [
 dict(name="atlantic_1920", truth=1, neutral="They will go by",
      note="EASY TRUTH control: steam was both expected and true for 1920",
      stems=["In the year 1920, travellers will cross the Atlantic chiefly by",
             "Come 1920, the journey from Europe to America will be made chiefly by",
             "By 1920, those who cross the ocean will go chiefly by",
             "In 1920, the passage to America will be made for the most part by",
             "When 1920 comes, men will cross the Atlantic above all by"],
      options=[" sailing ship", " steam ship", " flying machine", " a tunnel beneath the sea"]),
 dict(name="kaiser_1919_uncond", truth=3, neutral="It will be",
      note="1913 expectation was overwhelmingly the Kaiser; truth = elected president (Ebert)",
      stems=["In the year 1919, Germany will be ruled by",
             "Come 1919, the master of Germany will be",
             "By 1919, the government of Germany will rest with",
             "In 1919, the German people will owe their allegiance to",
             "When 1919 comes, the sceptre of Germany will belong to"],
      options=[" Kaiser Wilhelm the Second", " the young Crown Prince", " a military regent", " an elected president"]),
 dict(name="kaiser_1919_defeat", truth=3, neutral="It will be",
      note="same question conditioned on a lost great war",
      stems=["Germany has been beaten in a great war against half the world. In the year 1919, Germany will be ruled by",
             "After four years of war, Germany lies defeated and starving. Come 1919, the master of Germany will be",
             "The German armies are broken and the war is lost. By 1919, the government of Germany will rest with",
             "Germany has lost the great war and sues for peace. In 1919, the German people will owe their allegiance to",
             "Beaten in the field and blockaded at sea, Germany surrenders. When 1919 comes, the sceptre of Germany will belong to"],
      options=[" Kaiser Wilhelm the Second", " the young Crown Prince", " a military regent", " an elected president"]),
 dict(name="mark_1923_uncond", truth=3, neutral="It will be",
      note="unconditional fate of the mark",
      stems=["In the year 1923, the German mark will be",
             "Come 1923, the money of Germany will be",
             "By the year 1923, the mark of Germany will be",
             "In 1923, the currency of Germany will be",
             "When 1923 comes, the German mark will be"],
      options=[" as sound as ever it was", " somewhat fallen in its value", " worth but half of what it was", " worth nothing at all"]),
 dict(name="mark_1923_defeat", truth=3, neutral="It will be",
      note="conditioned on lost war + vast indemnity (the user's case)",
      stems=["Germany has been beaten in a great war and must pay her conquerors a vast indemnity. In the year 1923, the German mark will be",
             "Defeated after four years of war, Germany owes her enemies thousands of millions. Come 1923, the money of Germany will be",
             "The war is lost and the victors demand payment beyond all measure. By the year 1923, the mark of Germany will be",
             "Germany has lost the great war and must pay for all its ruin. In 1923, the currency of Germany will be",
             "Beaten and burdened with a crushing indemnity, Germany struggles on. When 1923 comes, the German mark will be"],
      options=[" as sound as ever it was", " somewhat fallen in its value", " worth but half of what it was", " worth nothing at all"]),
 dict(name="franc_mirror_defeat", truth=0, neutral="It will be",
      note="MIRROR: France beaten + indemnity. The record's precedent (1871) says the franc holds - exploratory",
      stems=["France has been beaten in a great war and must pay her conquerors a vast indemnity. Five years on, the French franc will be",
             "Defeated after years of war, France owes her enemies thousands of millions. Five years on, the money of France will be",
             "The war is lost and the victors demand payment beyond all measure. Five years on, the franc of France will be",
             "France has lost the great war and must pay for all its ruin. Five years on, the currency of France will be",
             "Beaten and burdened with a crushing indemnity, France struggles on. Five years on, the French franc will be"],
      options=[" as sound as ever it was", " somewhat fallen in its value", " worth but half of what it was", " worth nothing at all"]),
 dict(name="tsar_after_defeat", truth=2, neutral="Before long, he will",
      note="conditional the free-gen already volunteered: defeat -> Tsar deposed (1905 precedent)",
      stems=["Russia has been beaten in a great war. Within two years, the Tsar will",
             "The Russian armies are crushed and the war is lost. Within two years, the Tsar will",
             "Russia has suffered a great defeat at the hands of a foreign power. Soon the Tsar will",
             "Beaten in war and shaken by hunger at home, Russia trembles. Within two years, the Tsar will",
             "The war is lost and the Russian soldiers stream home in anger. Before long, the Tsar will"],
      options=[" sit securely upon his throne", " grant a parliament and reign on", " be driven from his throne", " be slain by his own people"]),
 dict(name="dread_weapon", truth=2, neutral="It will be",
      note="the new dread of the next war; truth coded as gas (contestable vs machine gun - report, don't lean)",
      stems=["In the next great war, the new weapon that soldiers will learn to dread most will be",
             "When the next great war comes, the arm most feared by the soldier will be",
             "In the coming great war, the deadliest new instrument of battle will be",
             "The next great war will teach soldiers to fear, above all,",
             "In the great war to come, the terror of the battlefield will be"],
      options=[" the machine gun", " the aeroplane with its bombs", " the cloud of poisoned gas", " the submarine boat"]),
 dict(name="thrones_1925", truth=2, neutral="They will",
      note="the broad monarchy question with a date",
      stems=["In the year 1925, the thrones of Europe will",
             "Come 1925, the crowned houses of Europe will",
             "By 1925, the monarchies of Europe will",
             "When 1925 arrives, the old thrones of Europe will",
             "In 1925, the royal houses of Europe will"],
      options=[" stand as firmly as ever they stood", " be fewer, some houses having fallen", " be swept away over half of Europe", " all have vanished from the earth"]),
]
MODELS = [("1900", RANKE["1900"]), ("1913", RANKE["1913"]), ("1929", RANKE["1929"])]
GEN_KEYS = ["kaiser_1919_uncond", "kaiser_1919_defeat", "mark_1923_defeat", "tsar_after_defeat"]

res = {"questions": {}, "gens": {}}
for tag, mid in MODELS:
    print(f"=== {mid} ({tag}) ===", flush=True)
    tok, model = common.load(mid)
    for q in QUESTIONS:
        probs_per_stem = []
        for stem in q["stems"]:
            c = [common.slp(model, tok, stem, o) - common.slp(model, tok, q["neutral"], o) for o in q["options"]]
            m = max(c)
            ex = [math.exp(x - m) for x in c]
            Z = sum(ex)
            probs_per_stem.append([e / Z for e in ex])
        res["questions"].setdefault(q["name"], {})[tag] = {
            "p_mean": [round(st.mean(ps[k] for ps in probs_per_stem), 3) for k in range(4)],
            "modal_per_stem": [max(range(4), key=lambda k: ps[k]) for ps in probs_per_stem]}
    # free generation color (greedy + 2 samples) on key conditioned stems
    for qn in GEN_KEYS:
        q = next(x for x in QUESTIONS if x["name"] == qn)
        stem = q["stems"][0]
        outs = [common.generate(model, tok, stem, n=25, sample=False)]
        outs += [common.generate(model, tok, stem, n=25, sample=True, temp=0.8) for _ in range(2)]
        res["gens"].setdefault(qn, {})[tag] = outs
    del tok, model
    common.free()

json.dump(res, open(os.path.join(common.RESULTS, "diag_betting_pilot.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("\n" + "=" * 100)
print("BETTING SLIPS — implied odds per model (mean over 5 phrasings); * = the true answer")
print("=" * 100)
for q in QUESTIONS:
    print(f"\n{q['name']}   [{q['note']}]")
    opts = [o.strip() for o in q["options"]]
    for k, o in enumerate(opts):
        star = "*" if k == q["truth"] else " "
        row = "".join(f"{res['questions'][q['name']][t]['p_mean'][k]:>8.0%}" for t, _ in MODELS)
        print(f"  {star} {o:42s}{row}")
    stab = res['questions'][q['name']]['1913']['modal_per_stem']
    print(f"    modal per phrasing (1913): {stab}  | surprise at truth (-ln p): "
          + ", ".join(f"{t}={-math.log(max(res['questions'][q['name']][t]['p_mean'][q['truth']],1e-4)):.2f}" for t, _ in MODELS))
print("\nwrote results/diag_betting_pilot.json")
