# -*- coding: utf-8 -*-
"""
CEILING TEST — does the forced-choice arm read knowledge on UNIVERSALLY-known facts?
====================================================================================
Version : v1  (2026-07-07)
Author  : Opus (Fable-review Phase 1b, clean version)

WHY. The battery-based modern audit was confounded: a small reader is genuinely shaky on some 1907-33
specifics, and the archaic memoir wording pulls a modern model around. To test the ARM itself we need
items whose answer is beyond dispute (moon landing, WWII leader, Mona Lisa) in PLAIN modern wording, so
that any miss is the arm's fault, not the reader's or the item's. If even Qwen2.5-3B — which certainly
knows these — scores them correctly through the exact certified arm, the arm reads knowledge and the
catastrophe-battery misses are item/framing difficulty, not an arm defect.

Scoring: the EXACT certified v41 path (imported from v41_battery: score / name_type / comb / split-half).
INPUT model: Qwen/Qwen2.5-3B (same reader as the confounded run, on purpose — if IT aces universal facts,
the arm is vindicated without needing a bigger model).
OUTPUT: ...\\12_foresight\\results\\ceiling_facts_audit.json
Run:  python src/ceiling_facts_audit.py
"""
import os, json, statistics as st
import common
import v41_battery as vb
from foresight_bank import MODERN

TAG = "modern"
# 13 universally-known facts, single unambiguous truth (index 0), plausible same-type/length distractors,
# plain modern register, 5 paraphrase stems each (needed for the name-type split-half branch).
FACTS = [
 dict(name="moon_first", truth=0, neutral="The man was",
      stems=["The first man to walk on the moon was", "The first human being to set foot on the moon was",
             "The astronaut who first stepped onto the surface of the moon was",
             "The first person ever to walk on the moon was", "The man who first walked upon the moon was"],
      options=[" Neil Armstrong", " Buzz Aldrin", " Yuri Gagarin", " John Glenn"]),
 dict(name="wwii_germany", truth=0, neutral="The leader was",
      stems=["The leader of Germany during the Second World War was", "Germany during the Second World War was led by",
             "The man who ruled Germany through the Second World War was",
             "During the Second World War, Germany was governed by", "The dictator who led Germany in the Second World War was"],
      options=[" Adolf Hitler", " Otto von Bismarck", " Kaiser Wilhelm", " Konrad Adenauer"]),
 dict(name="first_potus", truth=0, neutral="The president was",
      stems=["The first President of the United States was", "The first man to serve as President of the United States was",
             "The very first President of the United States was", "The first person to hold the office of U.S. President was",
             "The first President the United States ever had was"],
      options=[" George Washington", " Thomas Jefferson", " Abraham Lincoln", " John Adams"]),
 dict(name="mona_lisa", truth=0, neutral="The painter was",
      stems=["The painter of the Mona Lisa was", "The Mona Lisa was painted by", "The artist who created the Mona Lisa was",
             "The man who painted the Mona Lisa was", "The Mona Lisa is the work of"],
      options=[" Leonardo da Vinci", " Michelangelo Buonarroti", " Raphael of Urbino", " Rembrandt van Rijn"]),
 dict(name="relativity", truth=0, neutral="The scientist was",
      stems=["The theory of relativity was formulated by", "The scientist who developed the theory of relativity was",
             "The theory of relativity is the work of", "The man who gave the world the theory of relativity was",
             "Relativity in physics was the discovery of"],
      options=[" Albert Einstein", " Isaac Newton", " Niels Bohr", " Galileo Galilei"]),
 dict(name="titanic", truth=0, neutral="The ship was",
      stems=["The great ship that sank in 1912 after striking an iceberg was", "The liner that went down on its maiden voyage in 1912 was",
             "The ship lost to an iceberg in the north Atlantic in 1912 was", "The famous vessel that sank after hitting an iceberg in 1912 was",
             "The ocean liner that struck an iceberg and sank in 1912 was"],
      options=[" the Titanic", " the Lusitania", " the Britannic", " the Mauretania"]),
 dict(name="atom_bomb", truth=0, neutral="The country was",
      stems=["The country that first used an atomic bomb in war was", "The nation that dropped the first atomic bomb in wartime was",
             "The first country ever to drop an atomic bomb on an enemy was", "The atomic bombs of 1945 were dropped by",
             "The only country to have used atomic weapons in war is"],
      options=[" the United States", " the Soviet Union", " Great Britain", " Germany"]),
 dict(name="sputnik", truth=0, neutral="It was called",
      stems=["The first artificial satellite ever launched into orbit was called", "The first man-made satellite to circle the Earth was named",
             "In 1957 the first satellite put into orbit was called", "The world's first artificial satellite went by the name",
             "The first object ever placed in orbit by humans was called"],
      options=[" Sputnik", " Apollo", " Explorer", " Vostok"]),
 dict(name="everest", truth=0, neutral="It is",
      stems=["The highest mountain on Earth is", "The tallest mountain in the world is",
             "The mountain that rises higher than any other on Earth is", "The highest peak anywhere on Earth is",
             "The world's tallest mountain is"],
      options=[" Mount Everest", " Mont Blanc", " the Matterhorn", " Kilimanjaro"]),
 dict(name="hamlet", truth=0, neutral="The author was",
      stems=["The playwright who wrote Hamlet was", "The play Hamlet was written by", "The author of the play Hamlet was",
             "Hamlet, the famous tragedy, was written by", "The man who wrote Hamlet was"],
      options=[" William Shakespeare", " Charles Dickens", " Christopher Marlowe", " John Milton"]),
 dict(name="gravity", truth=0, neutral="The scientist was",
      stems=["The law of universal gravitation was discovered by", "The scientist who described the force of gravity was",
             "The man who formulated the law of gravity was", "Universal gravitation was first explained by",
             "The famous law of gravitation is the work of"],
      options=[" Isaac Newton", " Albert Einstein", " Galileo Galilei", " Johannes Kepler"]),
 dict(name="atlantic", truth=0, neutral="It is",
      stems=["The ocean that lies between Europe and America is", "The sea that separates Europe from the Americas is",
             "The ocean crossed by ships sailing from Europe to America is", "The ocean between the Old World and the New is",
             "The body of water dividing Europe and America is"],
      options=[" the Atlantic", " the Pacific", " the Indian Ocean", " the Arctic"]),
 dict(name="lion_phrase", truth=0, neutral="It is",
      stems=["The animal traditionally called the king of the jungle is", "The beast long known as the king of beasts is",
             "The animal people call the king of the jungle is", "By tradition, the king of the jungle is",
             "The creature commonly dubbed the king of beasts is"],
      options=[" the lion", " the tiger", " the elephant", " the wolf"]),
]

print(f"loading {MODERN} ...", flush=True)
tok, model = common.load(MODERN)
for q in FACTS:
    raw_ps, cal_ps = [], []
    for stem in q["stems"]:
        raw = [common.slp(model, tok, stem, o) for o in q["options"]]
        base = [common.slp(model, tok, q["neutral"], o) for o in q["options"]]
        raw_ps.append(vb.softmax(raw))
        cal_ps.append(vb.softmax([r - b for r, b in zip(raw, base)]))
    vb.res.setdefault(q["name"], {})[TAG] = {"raw_ps": raw_ps, "cal_ps": cal_ps}
del tok, model
common.free()

rows = []
for q in FACTS:
    p, covered = vb.score(q, TAG)
    top = max(range(4), key=lambda k: p[k])
    rows.append({"name": q["name"], "truth": q["options"][q["truth"]].strip(),
                 "p_truth": round(p[q["truth"]], 3), "picked": q["options"][top].strip(),
                 "correct": top == q["truth"], "covered": covered,
                 "type": "name" if vb.name_type(q["options"]) else "phrase"})

acc = st.mean(r["correct"] for r in rows)
meanp = st.mean(r["p_truth"] for r in rows)
misses = [r for r in rows if not r["correct"]]

if acc >= 0.85:
    verdict = (f"ARM READS KNOWLEDGE. On {len(rows)} universally-known facts the exact certified arm, "
               f"driven by a small (3B) reader, scores {acc:.0%} correct (mean p(truth)={meanp:.2f}). The "
               f"arm turns knowledge into the right pick. So the battery misses (incl. nazi_path ~33%) are "
               f"ITEM/framing difficulty, not an arm defect: KEEP the arm; fix/flag ambiguous items; and "
               f"note the arm needs a competent reader + clean options to be trusted.")
elif acc >= 0.6:
    verdict = (f"ARM IS LOSSY ({acc:.0%} on universal facts). Even beyond-dispute answers are only "
               f"sometimes recovered by this arm+reader. Escalate to a stronger reader (Qwen2.5-7B) to see "
               f"whether it is the arm or the 3B reader; do not lean on forced-choice alone meanwhile.")
else:
    verdict = (f"ARM OR READER BROKEN ({acc:.0%} on universal facts). Escalate to a stronger reader; if it "
               f"still fails, the forced-choice arm cannot read knowledge and must be cut from Paper 2.")

out = {"meta": {"version": "v1", "date": "2026-07-07", "reader": MODERN,
                "scoring": "certified v41 path", "items": "13 universally-known facts, plain register"},
       "accuracy": round(acc, 3), "mean_p_truth": round(meanp, 3), "n": len(rows),
       "misses": misses, "rows": rows, "verdict": verdict}
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "src" else HERE
OUT = os.path.join(ROOT, "results", "ceiling_facts_audit.json")
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("\n" + "=" * 78)
print(f"CEILING TEST — universal facts through the certified arm ({MODERN})")
print("=" * 78)
for r in rows:
    mark = "OK " if r["correct"] else "MISS"
    print(f"  [{mark}] {r['name']:14s} ({r['type']:5s}) p_truth {r['p_truth']:.2f}  truth={r['truth'][:20]:20s} picked={r['picked'][:20]}")
print(f"\naccuracy {acc:.0%}  mean p(truth) {meanp:.2f}  over {len(rows)} facts")
print("VERDICT:", verdict)
print("wrote", os.path.relpath(OUT, ROOT))
