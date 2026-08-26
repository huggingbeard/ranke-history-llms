# -*- coding: utf-8 -*-
"""The Bismarck/Ludendorff test (PI): does a method track TRUTH or just TERM FREQUENCY? Each item is a pre-1929 fact the
model SHOULD know, phrased so the TRUE answer is OBSCURE and a tempting FALSE answer is COMMON (frequency stacked against
truth). We compare three scores per candidate:
  raw      = slp(context, answer)                         -- pure log-prob (predicted to pick the COMMON false one)
  min-k%   = Min-K%% Prob of the full statement           -- Shi et al.; raw perplexity family (predicted to fail too)
  lift     = slp(context, answer) - slp(neutral, answer)  -- frequency-invariant (subtracts the name's base rate)
Pre-registered bar: a usable method must pick the TRUE-obscure answer over the FALSE-common one on the hard items
(>= 5/6). Raw and Min-K% are expected to FAIL (< chance); lift is expected to PASS if the model knows the fact.
Run:  python src/freq_confound_test.py
"""
import os, math
import numpy as np
import common

MID = "uzh-echist-org/Ranke-4B-1929"
NEUTRAL = "His name was"
# (context, TRUE-obscure surname, FALSE-common surname, hard?)  all pre-1929 facts the model should know
ITEMS = [
    ("The military strongman who held real power in Germany during the First World War was", "Ludendorff", "Bismarck", True),
    ("The head of the Russian Provisional Government overthrown by the Bolsheviks in 1917 was", "Kerensky", "Lenin", True),
    ("The French general who led the defence of Verdun in 1916 was", "Petain", "Napoleon", True),
    ("The British Prime Minister who led his country to victory in the First World War was", "George", "Churchill", True),
    ("The Austrian statesman who dominated the Congress of Vienna in 1815 was", "Metternich", "Napoleon", True),
    ("The Prussian commander whose arrival decided the Battle of Waterloo was", "Blucher", "Napoleon", True),
    ("The Bohemian religious reformer burned at the stake in 1415 was Jan", "Hus", "Luther", True),
    # controls: TRUE answer is also COMMON (frequency FOR truth) -- all methods should get these
    ("The theory of evolution by natural selection was proposed by Charles", "Darwin", "Newton", False),
    ("The Wealth of Nations was written by the economist Adam", "Smith", "Marx", False),
]

print(f"[freq] loading {MID} ...", flush=True)
tok, model = common.load(MID)


def minkprob(text, k=0.2):
    import torch
    ids = tok(text, add_special_tokens=False).input_ids
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([ids], device=model.device)).logits[0].float(), -1)
    t = sorted(lp[i - 1, ids[i]].item() for i in range(1, len(ids)))
    return float(np.mean(t[:max(1, int(len(t) * k))]))


rows = []
for ctx, T, F, hard in ITEMS:
    raw_T, raw_F = common.slp(model, tok, ctx, " " + T), common.slp(model, tok, ctx, " " + F)
    lift_T = raw_T - common.slp(model, tok, NEUTRAL, " " + T)
    lift_F = raw_F - common.slp(model, tok, NEUTRAL, " " + F)
    mk_T = minkprob(ctx + " " + T + ".")
    mk_F = minkprob(ctx + " " + F + ".")
    rows.append({"ctx": ctx, "T": T, "F": F, "hard": hard, "raw_ok": raw_T > raw_F, "lift_ok": lift_T > lift_F, "mk_ok": mk_T > mk_F,
                 "raw_T": raw_T, "raw_F": raw_F, "lift_T": lift_T, "lift_F": lift_F, "mk_T": mk_T, "mk_F": mk_F})
    print(f"  {'HARD' if hard else 'ctrl'}  {T:11}(T) vs {F:10}(F): "
          f"raw {'T' if raw_T>raw_F else 'F':>1}({raw_T:.1f}/{raw_F:.1f})  "
          f"lift {'T' if lift_T>lift_F else 'F':>1}({lift_T:+.1f}/{lift_F:+.1f})  "
          f"minK {'T' if mk_T>mk_F else 'F':>1}({mk_T:.2f}/{mk_F:.2f})")
del tok, model; common.free()

hard = [r for r in rows if r["hard"]]
n = len(hard)
for m, key in [("raw log-prob", "raw_ok"), ("frequency-invariant LIFT", "lift_ok"), ("Min-K% Prob", "mk_ok")]:
    c = sum(r[key] for r in hard)
    verdict = "PASS" if c >= 5 else ("chance" if c >= n * 0.4 else "FAIL (frequency-fooled)")
    print(f"  {m:26} picks TRUE-obscure over FALSE-common: {c}/{n}   -> {verdict}")
print(f"\ncontrols (true answer also common): raw {sum(r['raw_ok'] for r in rows if not r['hard'])}/{len(rows)-n}, "
      f"lift {sum(r['lift_ok'] for r in rows if not r['hard'])}/{len(rows)-n}")
print("The Reagan-flip logic: if only LIFT survives the frequency stack, calibration is the tool; if lift also fails, non-date detection is genuinely stuck.")
