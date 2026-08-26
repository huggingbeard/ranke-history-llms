# -*- coding: utf-8 -*-
"""
MODERN-READER AUDIT — can the forced-choice ("betting slip") arm read knowledge at all?
=======================================================================================
Version : v1  (2026-07-07)
Author  : Opus (Fable-review Phase 1b)

WHY. Fable's worry: the modern reader (Qwen2.5-3B) scored ~33% on the nazi_path item ("appointment
within the law" vs "an armed march on Berlin"), an answer a modern model certainly knows. If the arm
cannot read knowledge even when it is present, no forced-choice finding is trustworthy.

TEST. Run the MODERN reader (which knows all this history) over the ENTIRE certified v41 battery, using
the EXACT certified scoring path (imported from v41_battery: score / name_type / comb / split-half). A
modern model should ace the well-formed, famous-answer items. If it does, the arm reads knowledge, and
any miss is an ITEM problem (e.g. two-true-answers) — not an arm problem. We flag the two pre-identified
ambiguous items (rev_tsar_fate = deposed AND executed; nazi_path = legal appointment vs the salient
putsch template) and check whether the misses cluster there.

INPUT model: Qwen/Qwen2.5-3B (foresight_bank.MODERN).
OUTPUT: ...\\12_foresight\\results\\modern_audit.json
Run:  python src/modern_reader_audit.py
"""
import os, json, statistics as st
import common
import v41_battery as vb          # imports Q + softmax/name_type/mean_p/comb/score WITHOUT loading models
from foresight_bank import MODERN

TAG = "modern"
# pre-registered "ambiguous by construction" items (a miss here is an item flaw, not an arm flaw)
AMBIGUOUS = {
    "rev_tsar_fate": "two TRUE answers on the ballot (the Tsar was both deposed AND executed)",
    "nazi_path": "legal appointment vs the salient putsch template (armed march); misleading pair",
}

print(f"loading modern reader {MODERN} ...", flush=True)
tok, model = common.load(MODERN)
for q in vb.Q:
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
for q in vb.Q:
    p, covered = vb.score(q, TAG)            # EXACT certified scoring path
    top = max(range(4), key=lambda k: p[k])
    rows.append({
        "event": q["event"], "name": q["name"],
        "truth_opt": q["options"][q["truth"]].strip(),
        "modern_p_truth": round(p[q["truth"]], 3),
        "modern_top_bet": q["options"][top].strip(),
        "correct": top == q["truth"],
        "covered": covered,
        "ambiguous_item": q["name"] in AMBIGUOUS,
        "ambiguous_reason": AMBIGUOUS.get(q["name"], ""),
    })

clean = [r for r in rows if not r["ambiguous_item"]]
amb = [r for r in rows if r["ambiguous_item"]]
acc_clean = st.mean(r["correct"] for r in clean)
meanp_clean = st.mean(r["modern_p_truth"] for r in clean)
misses_clean = [r for r in clean if not r["correct"]]

# verdict
if acc_clean >= 0.80:
    verdict = (f"ARM READS KNOWLEDGE. On the {len(clean)} non-ambiguous items the modern reader gets "
               f"{acc_clean:.0%} right (mean p(truth)={meanp_clean:.2f}). The forced-choice arm CAN turn "
               f"knowledge into the correct pick. The ~33% on nazi_path is therefore an ITEM defect "
               f"(misleading option pair), not an arm defect: KEEP the arm, FIX/flag the ambiguous items.")
elif acc_clean >= 0.60:
    verdict = (f"ARM PARTLY READS KNOWLEDGE ({acc_clean:.0%} on clean items). Usable but noisy; report "
               f"the arm only alongside the generation arm, and fix the misses before relying on it.")
else:
    verdict = (f"ARM FAILS ({acc_clean:.0%} on clean, famous-answer items). The forced-choice arm cannot "
               f"reliably read knowledge; CUT it from Paper 2 as Fable warned.")

out = {
    "meta": {"version": "v1", "date": "2026-07-07", "reader": MODERN,
             "scoring": "certified v41 path (imported from v41_battery)",
             "note": "a modern model knows all this history, so misses reveal ITEM flaws, not ignorance"},
    "accuracy_clean_items": round(acc_clean, 3),
    "mean_p_truth_clean": round(meanp_clean, 3),
    "n_clean": len(clean), "n_ambiguous": len(amb),
    "misses_on_clean_items": [{"name": r["name"], "truth": r["truth_opt"],
                               "modern_picked": r["modern_top_bet"], "p_truth": r["modern_p_truth"]}
                              for r in misses_clean],
    "ambiguous_items": [{"name": r["name"], "reason": r["ambiguous_reason"], "correct": r["correct"],
                         "p_truth": r["modern_p_truth"], "modern_picked": r["modern_top_bet"]} for r in amb],
    "verdict": verdict,
    "all_rows": rows,
}
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "src" else HERE
OUT = os.path.join(ROOT, "results", "modern_audit.json")
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("\n" + "=" * 84)
print(f"MODERN-READER AUDIT  ({MODERN})")
print("=" * 84)
print(f"clean (non-ambiguous) items: {len(clean)} | accuracy {acc_clean:.0%} | mean p(truth) {meanp_clean:.2f}")
if misses_clean:
    print("misses on clean items:")
    for r in misses_clean:
        print(f"  {r['name']:22s} truth={r['truth_opt'][:28]:28s} picked={r['modern_top_bet'][:28]} (p_truth {r['modern_p_truth']:.2f})")
print("\npre-flagged ambiguous items:")
for r in amb:
    ok = "OK" if r["correct"] else "MISS"
    print(f"  {r['name']:22s} [{ok}] p_truth {r['modern_p_truth']:.2f} picked={r['modern_top_bet'][:30]}")
print("\nVERDICT:", verdict)
print("wrote", os.path.relpath(OUT, ROOT))
