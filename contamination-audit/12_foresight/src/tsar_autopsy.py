# -*- coding: utf-8 -*-
"""
TSAR AUTOPSY — why does the 1929 witness put only ~1.5% on "the Tsar was executed"?
===================================================================================
Version : v1  (2026-07-07)
Author  : Opus (Fable-review Phase 1a)

CONTEXT. In v41_battery.json the item rev_tsar_fate has witness p(truth)=0.015 and gap +2.71, and the
witness-gate drops it. But the Tsar WAS executed in 1918 — a 1929 reader should know this. So the 0.015
needs an autopsy. Two competing explanations:
  (a) ENDING WORDING — the specific phrase "been put to death with all his family" competes badly
      (length, register), even though the model knows the fact. => fixable, rebuild the ending.
  (b) BROKEN OPTION SET — the "distractor" " been driven from his throne" is ALSO TRUE (he was both
      deposed and executed). The model reasonably prefers the other true answer. => the item is invalid
      as a foresight finding, not evidence of anything.
  (c) GENUINE IGNORANCE — even reworded and against only-false alternatives, the witness stays low.
      => the 1929 model does not encode the execution; the question dies.

METHOD. Load the 1929 witness. Score with the EXACT v41 phrase-type path (softmax over 4 options per
stem, mean over 5 stems, then comb = sqrt(raw*calibrated) renormalised). Three probes:
  1. ORIGINAL option set (must reproduce ~0.015).
  2. PARAPHRASE sweep: swap 22 meaning-preserving rewordings of the execution ending into slot 3,
     keep the other three options fixed; record witness p(truth) for each.
  3. CLEAN DUEL: replace the competing-true option (" been driven from his throne") with genuinely
     FALSE fates; keep the execution ending; record witness p(truth).

INPUT model: uzh-echist-org/Ranke-4B-1929 (via foresight_bank.RANKE).
OUTPUT: ...\\12_foresight\\results\\tsar_autopsy.json
Run:  python src/tsar_autopsy.py
"""
import os, json, math, statistics as st
import common
from foresight_bank import RANKE

WITNESS = RANKE["1929"]           # the post-event reader for the revolution window
NEUTRAL = "He has"
STEMS = [
    "It is the year 1919. Russia was beaten in the great war. The Tsar of Russia has",
    "The year is 1919. The Russian armies were crushed and the war was lost. The Tsar has",
    "It is 1919. Defeat in war brought revolution to Russia. The Tsar himself has",
    "The year is 1919. Beaten in war and starved at home, Russia rose in revolt. The Tsar has",
    "It is 1919. The war was lost and the soldiers streamed home in anger. The Tsar of all the Russias has",
]
# original options; truth = index 3 (execution)
OPT_KEPT = " kept his throne through it all"
OPT_PARL = " granted a parliament and reigns on"
OPT_DEPOSED = " been driven from his throne"          # <-- the competing TRUE option
OPT_EXEC = " been put to death with all his family"    # <-- original truth wording

# 22 meaning-preserving rewordings of the execution ending (family killed with the Tsar).
PARAPHRASES = [
    " been put to death with all his family",              # original
    " been shot together with all his family",
    " been executed along with his whole family",
    " been killed with his wife and his children",
    " been murdered with all his household",
    " been slain together with his family",
    " been put to death, he and all his kin",
    " been shot dead with all his family",
    " been executed with the whole imperial family",
    " been killed, together with every one of his family",
    " been done to death with all his family",
    " been put to the sword with all his house",
    " been shot with his wife and all his children",
    " been executed, and his family with him",
    " been killed along with the Tsarina and their children",
    " been assassinated together with all his family",
    " been put to death with his entire family",
    " been killed with all those of his blood",
    " met death together with all his family",
    " been executed with all his family",
    " been put to death, together with the whole royal family",
    " been shot at Yekaterinburg with all his family",     # NB: adds a place cue — flagged
]
# clean duel: execution vs only-FALSE fates (the competing-true 'deposed' option removed)
DUEL_OPTIONS = [
    " kept his throne through it all",          # false
    " fled abroad and lives in exile",          # false (he did not escape)
    " been pardoned and lives quietly in Russia",  # false
    " been put to death with all his family",   # TRUTH, index 3
]

def softmax(xs):
    m = max(xs); ex = [math.exp(x - m) for x in xs]; Z = sum(ex)
    return [e / Z for e in ex]

def mean_p(ps):
    k = len(ps[0]); return [st.mean(p[i] for p in ps) for i in range(k)]

def comb(pr, pc):
    c = [math.sqrt(max(a, 1e-9) * max(b, 1e-9)) for a, b in zip(pr, pc)]
    Z = sum(c); return [x / Z for x in c]

def score_phrase(model, tok, options, truth_idx):
    """EXACT v41 phrase-type scoring: comb(mean raw, mean calibrated); returns (p_vector, p_truth, winner_idx)."""
    raw_ps, cal_ps = [], []
    for stem in STEMS:
        raw = [common.slp(model, tok, stem, o) for o in options]
        base = [common.slp(model, tok, NEUTRAL, o) for o in options]
        raw_ps.append(softmax(raw))
        cal_ps.append(softmax([r - b for r, b in zip(raw, base)]))
    pr, pc = mean_p(raw_ps), mean_p(cal_ps)
    p = comb(pr, pc)
    return p, p[truth_idx], max(range(len(options)), key=lambda k: p[k])

print(f"loading witness {WITNESS} ...", flush=True)
tok, model = common.load(WITNESS)

# 1. original
orig_opts = [OPT_KEPT, OPT_PARL, OPT_DEPOSED, OPT_EXEC]
_, p_orig, win_orig = score_phrase(model, tok, orig_opts, 3)
print(f"ORIGINAL  witness p(truth)={p_orig:.3f}  winner={orig_opts[win_orig].strip()!r}")

# 2. paraphrase sweep (swap execution ending into slot 3, keep others)
para_rows = []
for ph in PARAPHRASES:
    opts = [OPT_KEPT, OPT_PARL, OPT_DEPOSED, ph]
    _, p_t, win = score_phrase(model, tok, opts, 3)
    para_rows.append({"ending": ph, "witness_p_truth": round(p_t, 4), "winner": opts[win].strip()})
    print(f"  para p(truth)={p_t:.3f}  <- {ph.strip()!r}")
para_ps = [r["witness_p_truth"] for r in para_rows]

# 3. clean duel (remove the competing-true 'deposed' option)
_, p_duel, win_duel = score_phrase(model, tok, DUEL_OPTIONS, 3)
print(f"CLEAN DUEL witness p(truth)={p_duel:.3f}  winner={DUEL_OPTIONS[win_duel].strip()!r}")

del tok, model
common.free()

# ---- verdict ----
max_para = max(para_ps)
if max_para >= 0.50:
    verdict = ("ENDING-WORDING (fixable): a meaning-preserving rewording lifts the witness above 0.50, "
               "so 0.015 was phrasing, not ignorance. Rebuild rev_tsar_fate with the best-scoring ending.")
elif p_duel >= 0.50:
    verdict = ("BROKEN OPTION SET: against only-false fates the witness DOES know the execution "
               "(p>=0.50), but the original item pit the execution against another TRUE outcome "
               "(deposition). The item is invalid as a foresight finding — not evidence the execution "
               "was unforeseeable. Remove or fix; do not report the +2.71 gap.")
else:
    verdict = ("GENUINE IGNORANCE: even reworded and against only-false alternatives the witness stays "
               "below 0.50. The 1929 model does not reliably encode the Romanov execution; the question "
               "dies (and the pre-vs-post gap was never interpretable).")

out = {
    "meta": {"version": "v1", "date": "2026-07-07", "witness": WITNESS,
             "scoring": "v41 phrase-type: comb(mean softmax raw, mean softmax calibrated) over 5 stems"},
    "original": {"options": [o.strip() for o in orig_opts], "witness_p_truth": round(p_orig, 4),
                 "winner": orig_opts[win_orig].strip(),
                 "note": "reproduces v41_battery.json p_truth_post for rev_tsar_fate"},
    "paraphrase_sweep": {"n": len(para_rows), "max_p_truth": round(max_para, 4),
                         "median_p_truth": round(st.median(para_ps), 4), "rows": para_rows},
    "clean_duel": {"options": [o.strip() for o in DUEL_OPTIONS], "witness_p_truth": round(p_duel, 4),
                   "winner": DUEL_OPTIONS[win_duel].strip(),
                   "note": "competing-true 'deposed' option removed; execution vs only-false fates"},
    "verdict": verdict,
}
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "src" else HERE
OUT = os.path.join(ROOT, "results", "tsar_autopsy.json")
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nVERDICT:", verdict)
print("wrote", os.path.relpath(OUT, ROOT))
