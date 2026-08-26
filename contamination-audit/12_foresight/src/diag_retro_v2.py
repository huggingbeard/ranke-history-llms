# -*- coding: utf-8 -*-
"""CHRONICLE REGISTER, iteration 2: RAW odds (no neutral subtraction) vs calibrated, side by side.
The calibration bug: subtracting a phrase's generic commonness strangles answers that are common
phrases AND true ("worth nothing at all"). Raw odds keep the level; we report both and judge on raw.

Three FRESH out-of-sample questions added:
  panic_1908_retro   — the 1907 panic lies BETWEEN cutoffs: 1900 must guess, 1913 remembers.
  duration_1919_retro— how long did the great war last (1913's fake memoir reveals its duration prior).
  infl_country_retro — whose money died in 1923 (1913's guess reveals its template: loser? France?).

PRE-REGISTERED CRITERIA (raw odds):
 N1 mark: 1929 puts substantial mass (>25%) on 'worth nothing at all'.
 N2 kaiser: 1929 p(elected president) rises vs the calibrated run.
 N3 panic: 1913 (which read 1907) > 1900 on 'the trust companies'.
 N4 duration: 1929 modal = 'more than four years'.
 N5 country: 1929 modal = Germany.
Run: python src/diag_retro_v2.py
"""
import os, json, math, statistics as st
import common
from foresight_bank import RANKE

QUESTIONS = [
 dict(name="atlantic_1920", truth=1, neutral="They go by",
      stems=["It is the year 1920. Most travellers now cross the Atlantic by",
             "The year is 1920. The journey from Europe to America is now made chiefly by",
             "In this year of 1920, those who cross the ocean go for the most part by",
             "It is 1920, and the passage to America is made above all by",
             "The year is 1920. Men cross the Atlantic these days chiefly by"],
      options=[" sailing ship", " steam ship", " flying machine", " a tunnel beneath the sea"]),
 dict(name="kaiser_1919", truth=3, neutral="It is",
      stems=["It is the year 1919. Germany fought a great war against half the world and was beaten. The ruler of Germany today is",
             "The year is 1919. After four years of war Germany lies defeated. The master of Germany now is",
             "It is 1919. The German armies were broken and the war was lost. Germany is governed today by",
             "The year is 1919. Germany lost the great war and sued for peace. The German people now owe their allegiance to",
             "It is 1919. Beaten in the field and blockaded at sea, Germany surrendered last year. The sceptre of Germany now belongs to"],
      options=[" Kaiser Wilhelm the Second", " the young Crown Prince", " a military regent", " an elected president"]),
 dict(name="mark_1923", truth=3, neutral="It is",
      stems=["It is the year 1923. Germany lost the great war and has groaned for years under a vast indemnity. The German mark today is",
             "The year is 1923. Defeated in the great war, Germany owes her conquerors thousands of millions. The mark is now",
             "It is 1923. The war was lost and the victors demanded payment beyond measure. Today the German mark is",
             "The year is 1923. Germany, beaten and burdened with reparations, staggers on. Her currency today is",
             "It is 1923, five years since beaten Germany signed the peace. The German mark now is"],
      options=[" as sound as ever it was", " somewhat fallen in its value", " worth but half of what it was", " worth nothing at all"]),
 dict(name="tsar_1919", truth=3, neutral="He has",
      stems=["It is the year 1919. Russia was beaten in the great war. The Tsar of Russia has",
             "The year is 1919. The Russian armies were crushed and the war was lost. The Tsar has",
             "It is 1919. Defeat in war brought revolution to Russia. The Tsar himself has",
             "The year is 1919. Beaten in war and starved at home, Russia rose in revolt. The Tsar has",
             "It is 1919. The war was lost and the soldiers streamed home in anger. The Tsar of all the Russias has"],
      options=[" kept his throne through it all", " granted a parliament and reigns on", " been driven from his throne", " been put to death by his own people"]),
 dict(name="thrones_1925", truth=2, neutral="They have",
      stems=["It is the year 1925. The great war has come and gone. The thrones of Europe have",
             "The year is 1925. Looking back over the storm of the great war, the crowned houses of Europe have",
             "It is 1925. The great war swept over Europe and is done. Its old monarchies have",
             "The year is 1925. Since the great war ended, the thrones of Europe have",
             "It is 1925, seven years since the great war ended. The royal houses of Europe have"],
      options=[" stood as firmly as ever they stood", " grown fewer, some houses having fallen", " been swept away over half of Europe", " all vanished from the earth"]),
 dict(name="gas_1919", truth=2, neutral="It was",
      stems=["It is the year 1919. In the great war just ended, the new weapon the soldiers came to dread most was",
             "The year is 1919. Of all the new arms of the late war, the most feared was",
             "It is 1919. In the war now over, the deadliest new instrument of battle proved to be",
             "The year is 1919. The great war taught soldiers to fear, above all,",
             "It is 1919. In the war that has just closed, the terror of the battlefield was"],
      options=[" the machine gun", " the aeroplane with its bombs", " the cloud of poisoned gas", " the submarine boat"]),
 # ---------- fresh, out-of-sample ----------
 dict(name="panic_1908", truth=0, neutral="They were",
      stems=["It is the year 1908. Last autumn a great panic swept the New York market. The institutions that failed first were",
             "The year is 1908. In the panic of last October, the first to close their doors were",
             "It is 1908. When the panic struck New York last fall, the first to stop payment were",
             "The year is 1908. In the great panic just past, the houses that fell first were",
             "It is 1908, a few months after the great panic. Those that failed first in it were"],
      options=[" the trust companies", " the savings banks", " the national banks", " the insurance offices"]),
 dict(name="duration_1919", truth=2, neutral="It lasted",
      stems=["It is the year 1919. The great war that began in the summer of 1914 lasted",
             "The year is 1919. The great war, begun in 1914, went on for",
             "It is 1919. From its outbreak in 1914 to the armistice, the great war lasted",
             "The year is 1919. The war that started in August 1914 endured",
             "It is 1919. The great war which opened in 1914 continued for"],
      options=[" but a few weeks", " less than a year", " more than four years", " a full ten years"]),
 dict(name="infl_country_1923", truth=0, neutral="It is",
      stems=["It is the year 1923. In one great nation this year, money has died: prices are reckoned in thousands of millions. That nation is",
             "The year is 1923. One country's currency has collapsed into utter worthlessness this year. That country is",
             "It is 1923. In one land, a loaf of bread now costs a basketful of banknotes. That land is",
             "The year is 1923. One great people now carries its wages home in baskets and barrows. That people is",
             "It is 1923. In one nation the printing presses have destroyed the money altogether. That nation is"],
      options=[" Germany", " France", " England", " Italy"]),
]
MODELS = [("1900", RANKE["1900"]), ("1913", RANKE["1913"]), ("1929", RANKE["1929"])]
GEN_KEYS = ["panic_1908", "duration_1919", "infl_country_1923"]


def softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    Z = sum(ex)
    return [e / Z for e in ex]


res = {"questions": {}, "gens": {}}
for tag, mid in MODELS:
    print(f"=== {mid} ({tag}) ===", flush=True)
    tok, model = common.load(mid)
    for q in QUESTIONS:
        raw_ps, cal_ps = [], []
        for stem in q["stems"]:
            raw = [common.slp(model, tok, stem, o) for o in q["options"]]
            base = [common.slp(model, tok, q["neutral"], o) for o in q["options"]]
            raw_ps.append(softmax(raw))
            cal_ps.append(softmax([r - b for r, b in zip(raw, base)]))
        res["questions"].setdefault(q["name"], {})[tag] = {
            "raw": [round(st.mean(ps[k] for ps in raw_ps), 3) for k in range(4)],
            "cal": [round(st.mean(ps[k] for ps in cal_ps), 3) for k in range(4)],
            "raw_modal_per_stem": [max(range(4), key=lambda k: ps[k]) for ps in raw_ps]}
    for qn in GEN_KEYS:
        q = next(x for x in QUESTIONS if x["name"] == qn)
        outs = [common.generate(model, tok, q["stems"][0], n=25, sample=False)]
        outs += [common.generate(model, tok, q["stems"][0], n=25, sample=True, temp=0.8) for _ in range(2)]
        res["gens"].setdefault(qn, {})[tag] = outs
    del tok, model
    common.free()

json.dump(res, open(os.path.join(common.RESULTS, "diag_retro_v2.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("\n" + "=" * 108)
print("CHRONICLE REGISTER v2 — RAW odds (calibrated in grey brackets); * = what actually happened")
print("=" * 108)
for q in QUESTIONS:
    print(f"\n{q['name']}")
    for k, o in enumerate([o.strip() for o in q["options"]]):
        star = "*" if k == q["truth"] else " "
        cells = "".join(f"{res['questions'][q['name']][t]['raw'][k]:>7.0%} ({res['questions'][q['name']][t]['cal'][k]:>3.0%})"
                        for t, _ in MODELS)
        print(f"  {star} {o:38s}{cells}")
    print(f"    raw modal per phrasing 1913: {res['questions'][q['name']]['1913']['raw_modal_per_stem']}"
          f"   1929: {res['questions'][q['name']]['1929']['raw_modal_per_stem']}")
    print("    raw surprise at truth: " + ", ".join(
        f"{t}={-math.log(max(res['questions'][q['name']][t]['raw'][q['truth']],1e-4)):.2f}" for t, _ in MODELS))
print("\n--- fake memoirs vs real memory (greedy + 2 samples) ---")
for qn in GEN_KEYS:
    for t in ("1900", "1913", "1929"):
        print(f"[{qn} @{t}] " + " || ".join(g[:56].replace("\n", " ") for g in res["gens"][qn][t]))
print("\nwrote results/diag_retro_v2.json")
