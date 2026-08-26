# -*- coding: utf-8 -*-
"""RETROSPECTIVE-REGISTER PILOT (the user's fix): ask every model in the CHRONICLE register.
"It is 1923, looking back..." — a genuine history question for the 1929 model (gates prove that
register reaches knowledge), a fake memoir for the 1913 model (its confabulation = its priors,
expressed in the reality-anchored genre of history-telling instead of prophecy).

PRE-REGISTERED PASS CRITERIA:
 R1 de-fictionalization: the Atlantic 'tunnel' share collapses vs the forward version (45% at 1913).
 R2 knowledge extraction (decisive): the 1929 model puts modal/majority mass on the TRUE answers it
    read about — elected president, worthless mark, Tsar dead, thrones swept away, gas.
 R3 coherent priors at 1913: fake memoir favors a Hohenzollern on the throne, a surviving mark, etc.
 R4 stability: modal option stable across >=4/5 phrasings.
The foresight measure, if R2 passes: surprise-at-truth(1913) - surprise-at-truth(1929), same slips.
Run: python src/diag_retro_pilot.py
"""
import os, json, math, statistics as st
import common
from foresight_bank import RANKE

QUESTIONS = [
 dict(name="atlantic_1920_retro", truth=1, neutral="They go by",
      note="R1: tunnel must collapse (forward version gave it 45% at 1913 — a 1913 bestseller novel)",
      stems=["It is the year 1920. Most travellers now cross the Atlantic by",
             "The year is 1920. The journey from Europe to America is now made chiefly by",
             "In this year of 1920, those who cross the ocean go for the most part by",
             "It is 1920, and the passage to America is made above all by",
             "The year is 1920. Men cross the Atlantic these days chiefly by"],
      options=[" sailing ship", " steam ship", " flying machine", " a tunnel beneath the sea"]),
 dict(name="kaiser_1919_retro", truth=3, neutral="It is",
      note="defeat built into the chronicle; truth = an elected president",
      stems=["It is the year 1919. Germany fought a great war against half the world and was beaten. The ruler of Germany today is",
             "The year is 1919. After four years of war Germany lies defeated. The master of Germany now is",
             "It is 1919. The German armies were broken and the war was lost. Germany is governed today by",
             "The year is 1919. Germany lost the great war and sued for peace. The German people now owe their allegiance to",
             "It is 1919. Beaten in the field and blockaded at sea, Germany surrendered last year. The sceptre of Germany now belongs to"],
      options=[" Kaiser Wilhelm the Second", " the young Crown Prince", " a military regent", " an elected president"]),
 dict(name="mark_1923_retro", truth=3, neutral="It is",
      note="defeat + indemnity built in; truth = worth nothing at all",
      stems=["It is the year 1923. Germany lost the great war and has groaned for years under a vast indemnity. The German mark today is",
             "The year is 1923. Defeated in the great war, Germany owes her conquerors thousands of millions. The mark is now",
             "It is 1923. The war was lost and the victors demanded payment beyond measure. Today the German mark is",
             "The year is 1923. Germany, beaten and burdened with reparations, staggers on. Her currency today is",
             "It is 1923, five years since beaten Germany signed the peace. The German mark now is"],
      options=[" as sound as ever it was", " somewhat fallen in its value", " worth but half of what it was", " worth nothing at all"]),
 dict(name="tsar_1919_retro", truth=3, neutral="He has",
      note="truth by 1919 = put to death (deposed 1917 also true — report both)",
      stems=["It is the year 1919. Russia was beaten in the great war. The Tsar of Russia has",
             "The year is 1919. The Russian armies were crushed and the war was lost. The Tsar has",
             "It is 1919. Defeat in war brought revolution to Russia. The Tsar himself has",
             "The year is 1919. Beaten in war and starved at home, Russia rose in revolt. The Tsar has",
             "It is 1919. The war was lost and the soldiers streamed home in anger. The Tsar of all the Russias has"],
      options=[" kept his throne through it all", " granted a parliament and reigns on", " been driven from his throne", " been put to death by his own people"]),
 dict(name="thrones_1925_retro", truth=2, neutral="They have",
      note="truth = swept away over half of Europe",
      stems=["It is the year 1925. The great war has come and gone. The thrones of Europe have",
             "The year is 1925. Looking back over the storm of the great war, the crowned houses of Europe have",
             "It is 1925. The great war swept over Europe and is done. Its old monarchies have",
             "The year is 1925. Since the great war ended, the thrones of Europe have",
             "It is 1925, seven years since the great war ended. The royal houses of Europe have"],
      options=[" stood as firmly as ever they stood", " grown fewer, some houses having fallen", " been swept away over half of Europe", " all vanished from the earth"]),
 dict(name="gas_1919_retro", truth=2, neutral="It was",
      note="truth coded gas as the war's new dread (machine gun contestable — report, don't lean)",
      stems=["It is the year 1919. In the great war just ended, the new weapon the soldiers came to dread most was",
             "The year is 1919. Of all the new arms of the late war, the most feared was",
             "It is 1919. In the war now over, the deadliest new instrument of battle proved to be",
             "The year is 1919. The great war taught soldiers to fear, above all,",
             "It is 1919. In the war that has just closed, the terror of the battlefield was"],
      options=[" the machine gun", " the aeroplane with its bombs", " the cloud of poisoned gas", " the submarine boat"]),
]
MODELS = [("1900", RANKE["1900"]), ("1913", RANKE["1913"]), ("1929", RANKE["1929"])]
GEN_KEYS = ["kaiser_1919_retro", "mark_1923_retro", "tsar_1919_retro"]

res = {"questions": {}, "gens": {}}
for tag, mid in MODELS:
    print(f"=== {mid} ({tag}) ===", flush=True)
    tok, model = common.load(mid)
    for q in QUESTIONS:
        pps = []
        for stem in q["stems"]:
            c = [common.slp(model, tok, stem, o) - common.slp(model, tok, q["neutral"], o) for o in q["options"]]
            m = max(c)
            ex = [math.exp(x - m) for x in c]
            Z = sum(ex)
            pps.append([e / Z for e in ex])
        res["questions"].setdefault(q["name"], {})[tag] = {
            "p_mean": [round(st.mean(ps[k] for ps in pps), 3) for k in range(4)],
            "modal_per_stem": [max(range(4), key=lambda k: ps[k]) for ps in pps]}
    for qn in GEN_KEYS:
        q = next(x for x in QUESTIONS if x["name"] == qn)
        outs = [common.generate(model, tok, q["stems"][0], n=25, sample=False)]
        outs += [common.generate(model, tok, q["stems"][0], n=25, sample=True, temp=0.8) for _ in range(2)]
        res["gens"].setdefault(qn, {})[tag] = outs
    del tok, model
    common.free()

json.dump(res, open(os.path.join(common.RESULTS, "diag_retro_pilot.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("\n" + "=" * 100)
print("CHRONICLE REGISTER — implied odds (mean over 5 phrasings); * = what actually happened")
print("=" * 100)
for q in QUESTIONS:
    print(f"\n{q['name']}   [{q['note']}]")
    for k, o in enumerate([o.strip() for o in q["options"]]):
        star = "*" if k == q["truth"] else " "
        row = "".join(f"{res['questions'][q['name']][t]['p_mean'][k]:>8.0%}" for t, _ in MODELS)
        print(f"  {star} {o:42s}{row}")
    print(f"    modal per phrasing 1913: {res['questions'][q['name']]['1913']['modal_per_stem']}"
          f"  1929: {res['questions'][q['name']]['1929']['modal_per_stem']}")
    print("    surprise at truth (-ln p): " + ", ".join(
        f"{t}={-math.log(max(res['questions'][q['name']][t]['p_mean'][q['truth']],1e-4)):.2f}" for t, _ in MODELS))
print("\n--- fake memoirs vs real memory (greedy + 2 samples, first stem) ---")
for qn in GEN_KEYS:
    for t in ("1913", "1929"):
        print(f"[{qn} @{t}] " + " || ".join(g[:60].replace("\n", " ") for g in res["gens"][qn][t]))
print("\nwrote results/diag_retro_pilot.json")
