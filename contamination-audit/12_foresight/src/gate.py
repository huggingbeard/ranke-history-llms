# -*- coding: utf-8 -*-
"""CONTAMINATION GATE (Arm A). A low-surprise (foresight-looking) reading is trustworthy only if the PRE model
is CLEAN on the event -- i.e. does not actually KNOW the outcome from leaked post-cutoff text. For each event we
score its post-shock FACT battery under the PRE model with the preregistered calibrated forced choice
c(a)=slp(prompt,a)-mean_z slp(neutral_z,a) (score_v2 metric), chance = 25%, and apply the parent project's
traffic_light rule:
  Test A: POST facts vs 25% chance ;  Test B: POST vs the PRE model's own pre-cutoff positive controls.
  GREEN = POST at chance -> trust the foresight reading ;  AMBER/RED = the PRE model knows the outcome -> flag.
Also reports the two Oct-1929 boundary probes (does the 1929 corpus contain the crash itself?). Loads 1913 and
1929 once each. Run: python gate.py
"""
import os, json, math, statistics as st
import common
from foresight_bank import GATE

NEUTRALS = ["The answer is", "The answer was"]

# Pre-cutoff POSITIVE controls per PRE model (facts the model should ACE -> the PRE baseline for Test B).
# 1913 gate: all facts <= 1913.  1929 gate: all facts <= 1929.
PRE_CONTROLS = {
 "uzh-echist-org/Ranke-4B-1913": [
  ("The great sea battle of 1805 where Nelson fell was", " Trafalgar", [" Jutland", " Salamis", " Lepanto"]),
  ("Napoleon was finally defeated in 1815 at", " Waterloo", [" Austerlitz", " Borodino", " Jena"]),
  ("The French Revolution began in 1789 with the storming of the", " Bastille", [" Louvre", " Tuileries", " Sorbonne"]),
  ("Charles Darwin set out his theory in On the Origin of", " Species", [" Life", " Man", " Nature"]),
  ("The unification of Germany under Prussia was completed in the year", " 1871", [" 1848", " 1815", " 1806"]),
  ("Queen Victoria came to the British throne in the year", " 1837", [" 1801", " 1860", " 1714"]),
  ("The great liner that struck an iceberg and sank in 1912 was the", " Titanic", [" Lusitania", " Mauretania", " Olympic"]),
  ("The American Declaration of Independence was adopted in the year", " 1776", [" 1789", " 1798", " 1812"]),
  ("The Suez Canal joining the Mediterranean to the Red Sea opened in", " 1869", [" 1825", " 1901", " 1789"]),
  ("The Iron Chancellor who unified Germany was Otto von", " Bismarck", [" Moltke", " Roon", " Caprivi"]),
 ],
 "uzh-echist-org/Ranke-4B-1929": [
  ("Napoleon was finally defeated in 1815 at", " Waterloo", [" Austerlitz", " Borodino", " Jena"]),
  ("The French Revolution began in 1789 with the storming of the", " Bastille", [" Louvre", " Tuileries", " Sorbonne"]),
  ("The First World War broke out across Europe in the year", " 1914", [" 1905", " 1898", " 1888"]),
  ("The Tsar of Russia was swept away by revolution in the year", " 1917", [" 1905", " 1881", " 1825"]),
  ("The 1919 treaty that ended the war was signed at the palace of", " Versailles", [" Vienna", " Potsdam", " Trianon"]),
  ("Charles Darwin set out his theory in On the Origin of", " Species", [" Life", " Man", " Nature"]),
  ("The unification of Germany under Prussia was completed in the year", " 1871", [" 1848", " 1815", " 1806"]),
  ("The great liner that struck an iceberg and sank in 1912 was the", " Titanic", [" Lusitania", " Mauretania", " Olympic"]),
  ("The republic founded in Germany after the war was the", " Weimar", [" Rhenish", " Saxon", " Bonn"]),
  ("The Bolshevik leader of the 1917 revolution was Vladimir", " Lenin", [" Kerensky", " Martov", " Plekhanov"]),
 ],
}


def cal_choice(model, tok, prompt, true, distractors):
    """calibrated forced choice; returns (correct, gap_cal)."""
    opts = [true] + distractors
    sq = {a: common.slp(model, tok, prompt, a) for a in opts}
    s0 = {a: st.mean(common.slp(model, tok, nz, a) for nz in NEUTRALS) for a in opts}
    c = {a: sq[a] - s0[a] for a in opts}
    dbest = max(c[d] for d in distractors)
    return int(c[true] > dbest), round(c[true] - dbest, 4)


def binom_ge(k, n, p=0.25):
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def two(z):
    return math.erfc(abs(z) / math.sqrt(2))


def verdict(mpost, npost, mpre, npre):
    # Test A: POST vs 0.25 chance ; Test B: POST vs PRE
    seA = math.sqrt(0.25 * 0.75 / npost) if npost else float("nan")
    zA = (mpost - 0.25) / seA if seA else 0.0
    pA = two(zA)
    seB = math.sqrt(mpost * (1 - mpost) / npost + mpre * (1 - mpre) / npre) if npost and npre else float("nan")
    zB = (mpre - mpost) / seB if seB else 0.0
    pB = two(zB)
    if not (pA < 0.05 and mpost > 0.25):
        v = "GREEN (clean: POST at chance)"
    elif pB < 0.05:
        v = "AMBER (partial: POST>chance but < PRE)"
    else:
        v = "RED (contaminated: POST as good as PRE)"
    return v, pA, pB


# group gate facts by pre_model
by_pre = {}
for event, g in GATE.items():
    by_pre.setdefault(g["pre_model"], []).append((event, g))

report = {}
loaded = {}
for pre_model, events in by_pre.items():
    print(f"\n=== GATE: loading PRE model {pre_model} ===", flush=True)
    tok, model = common.load(pre_model)
    # PRE positive controls
    pre_rows = [cal_choice(model, tok, p, t, d)[0] for (p, t, d) in PRE_CONTROLS[pre_model]]
    mpre = st.mean(pre_rows); npre = len(pre_rows)
    print(f"  PRE positive controls: {sum(pre_rows)}/{npre} = {100*mpre:.0f}%", flush=True)
    for event, g in events:
        rows = []
        boundary = []
        for (year, lab, prompt, true, distractors) in g["facts"]:
            corr, gap = cal_choice(model, tok, prompt, true, distractors)
            rec = {"year": year, "label": lab, "correct": corr, "gap_cal": gap}
            # separate the Oct-1929 boundary probes (year==1929) from the post-cutoff facts
            if year == 1929:
                boundary.append(rec)
            else:
                rows.append(rec)
        k = sum(r["correct"] for r in rows); n = len(rows); mpost = k / n if n else 0
        v, pA, pB = verdict(mpost, n, mpre, npre)
        report[event] = {"pre_model": pre_model.split("/")[-1], "post_correct": k, "post_n": n,
                         "post_pct": round(100 * mpost, 1), "chance": 25.0, "binom_p": round(binom_ge(k, n), 4),
                         "pre_pct": round(100 * mpre, 1), "pre_n": npre, "testA_p": round(pA, 4),
                         "testB_p": round(pB, 4), "verdict": v, "items": rows, "boundary_probes": boundary}
        print(f"  [{event}] POST {k}/{n} = {100*mpost:.0f}% (chance 25, binom p={binom_ge(k,n):.3f})  "
              f"PRE {100*mpre:.0f}%  -> {v}", flush=True)
        if boundary:
            bk = sum(r["correct"] for r in boundary)
            print(f"        Oct-1929 boundary probes: {bk}/{len(boundary)} known "
                  f"({', '.join(r['label']+('=Y' if r['correct'] else '=n') for r in boundary)})", flush=True)
    common.free(model)

json.dump(report, open(os.path.join(common.RESULTS, "gate_report.json"), "w"), indent=1)
print("\nwrote gate_report.json")
