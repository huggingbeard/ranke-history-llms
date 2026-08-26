# -*- coding: utf-8 -*-
"""Rich-context contamination suite on WW2 + the atomic bomb (post-1929 facts a 1929-locked model should NOT know).
Each probe pits the POST-cutoff true answer against the PRE-1929 DEFAULT (what an uncontaminated mind would say), so
preferring the true one IS contamination. Two frequency-robust scores per probe:
  raw margin  = slp(context, TRUE) - slp(context, FALSE)                 (rich context does the frequency control)
  lift margin = [slp(ctx,T)-slp(neutral,T)] - [slp(ctx,F)-slp(neutral,F)] (subtract each answer's base rate)
Controls: pre-1929 KNOWN (validation, should flag), post-cutoff UNKNOWN + INVENTED (null, should NOT). The null sets the
bar: a WW2 probe is LIKELY CONTAMINATED if raw OR lift margin exceeds the largest margin any control reaches (the PI's
OR rule, thresholded on controls, not on the raw sign). "Likely", i.e. a corpus-check candidate.
Run:  python src/ww2_probes.py
"""
import os, json
import numpy as np
import common

MID = "uzh-echist-org/Ranke-4B-1929"
# (category, context, TRUE post-cutoff answer, FALSE pre-1929 default, neutral base-rate prompt)
PROBES = [
    ("ww2", "The first atomic bomb ever used in war destroyed the Japanese city of", "Hiroshima", "Tokyo", "A Japanese city named"),
    ("ww2", "Three days after the first, the second atomic bomb fell on the city of", "Nagasaki", "Kyoto", "A Japanese city named"),
    ("ww2", "The Second World War in Europe began when Germany invaded the country of", "Poland", "France", "A country named"),
    ("ww2", "During the Second World War, Germany was ruled by a dictator named", "Hitler", "Bismarck", "A German man named"),
    ("ww2", "The great Allied invasion of 1944 landed on the beaches of", "Normandy", "Flanders", "A place named"),
    ("ww2", "In 1941 the American base at Pearl Harbor was attacked by the nation of", "Japan", "Germany", "A country named"),
    ("ww2", "The physicist who led the team that built the first atomic bomb was", "Oppenheimer", "Einstein", "A physicist named"),
    ("ww2", "The secret American project that built the atomic bomb was named after", "Manhattan", "Panama", "A place named"),
    ("ww2", "The Soviet leader who fought Nazi Germany in the Second World War was", "Stalin", "Lenin", "A Soviet leader named"),
    ("ww2", "The British Prime Minister who led Britain through the Second World War was", "Churchill", "Gladstone", "A British statesman named"),
    ("ww2", "The Nazi extermination of the Jews of Europe is remembered as the", "Holocaust", "Inquisition", "A historical event named"),
    ("ww2", "The German air force that fought the Battle of Britain was called the", "Luftwaffe", "Zeppelin", "A military force named"),
    # ---- pre-1929 KNOWN (validation: should flag) ----
    ("known", "The theory of evolution by natural selection was proposed by Charles", "Darwin", "Newton", "A scientist named"),
    ("known", "The French emperor defeated at the Battle of Waterloo was", "Napoleon", "Louis", "An emperor named"),
    ("known", "The Bolshevik who led the 1917 Russian Revolution was", "Lenin", "Kerensky", "A Russian named"),
    ("known", "The Wealth of Nations was written by the economist Adam", "Smith", "Marx", "An economist named"),
    ("known", "The assassinated archduke whose death sparked the First World War was Franz", "Ferdinand", "Joseph", "An archduke named"),
    ("known", "The scientist who formulated the laws of motion and gravity was Isaac", "Newton", "Darwin", "A scientist named"),
    # ---- post-cutoff UNKNOWN (null: should NOT flag) ----
    ("unknown", "The first man to walk on the Moon was named", "Armstrong", "Lindbergh", "An aviator named"),
    ("unknown", "The American President forced to resign by the Watergate scandal was", "Nixon", "Coolidge", "An American President named"),
    ("unknown", "The social network created by Mark Zuckerberg is called", "Facebook", "Telegraph", "A communication network named"),
    ("unknown", "The country that suffered the Chernobyl nuclear disaster was the", "Soviet Union", "United States", "A country named"),
    ("unknown", "The virus behind the 2020 global pandemic is the", "coronavirus", "cholera", "A disease named"),
    # ---- INVENTED (null: should NOT flag) ----
    ("invented", "The great naval battle of the Astral War was won by the fleet of", "Veldania", "Britain", "A nation named"),
    ("invented", "The philosopher who founded the school of Thennic idealism was", "Korvath", "Kant", "A philosopher named"),
    ("invented", "The lost continent of Zar-Thonis sank beneath the waters of the", "Kelmar", "Atlantic", "A sea named"),
]

print(f"[ww2] loading {MID} ...", flush=True)
tok, model = common.load(MID)
rows = []
for cat, ctx, T, F, neut in PROBES:
    rT, rF = common.slp(model, tok, ctx, " " + T), common.slp(model, tok, ctx, " " + F)
    lT = rT - common.slp(model, tok, neut, " " + T)
    lF = rF - common.slp(model, tok, neut, " " + F)
    rows.append({"cat": cat, "ctx": ctx, "T": T, "F": F, "raw_margin": rT - rF, "lift_margin": lT - lF})
del tok, model; common.free()

null = [r for r in rows if r["cat"] in ("unknown", "invented")]
thr_raw = max(r["raw_margin"] for r in null)                    # largest margin any control reaches
thr_lift = max(r["lift_margin"] for r in null)
for r in rows:
    r["flag"] = (r["raw_margin"] > thr_raw) or (r["lift_margin"] > thr_lift)     # PI's OR rule, control-thresholded

print(f"\ncontrol bar (max margin over {len(null)} unknown+invented):  raw {thr_raw:+.2f}   lift {thr_lift:+.2f}\n")
for cat in ("ww2", "known", "unknown", "invented"):
    print(f"--- {cat} ---")
    for r in [x for x in rows if x["cat"] == cat]:
        mark = "  <== LIKELY CONTAMINATED" if r["flag"] else ""
        print(f"  {r['T']:12} vs {r['F']:12}  raw {r['raw_margin']:+6.2f}  lift {r['lift_margin']:+6.2f}{mark}")
ww2 = [r for r in rows if r["cat"] == "ww2"]
print(f"\nSUMMARY (OR rule, thresholded on controls):")
print(f"  WW2/atomic flagged as likely contaminated:  {sum(r['flag'] for r in ww2)}/{len(ww2)}")
print(f"  pre-1929 KNOWN flagged (validation, want high): {sum(r['flag'] for r in rows if r['cat']=='known')}/{sum(1 for r in rows if r['cat']=='known')}")
print(f"  UNKNOWN+INVENTED flagged (false positives, want ~0): {sum(r['flag'] for r in null)}/{len(null)}")
json.dump(rows, open(os.path.join(common.RESULTS, "superlatives", "ww2_probes.json"), "w"), indent=1)
print(f"wrote {os.path.join(common.RESULTS, 'superlatives', 'ww2_probes.json')}")
