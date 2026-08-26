# -*- coding: utf-8 -*-
"""RECOGNITION vs PRODUCTION (Arm A). Teacher-forcing (score_all) measures the CEILING of what the worldview
finds plausible; free generation measures the weaker FLOOR of what it will VOLUNTEER. A model may recognise the
true (catastrophic) outcome under teacher-forcing yet never produce it unprompted. Only a production hit
supports a strong 'inkling' claim.

For each PRODUCTION item, free-generate from free_prompt (greedy + N sampled) on the BASE models 1900/1913/1929
and the INSTRUCT models (chat form). Code each completion by true/false markers. Report, per item x model:
  recog = did calibrated teacher-forcing prefer the true twin? (from scores_by_model.json)
  prod  = fraction of generations containing TRUE markers vs FALSE markers.
Writes production_report.json + prints a table. Run: python production.py
"""
import os, json, statistics as st
import common
from foresight_bank import PRODUCTION, RANKE

NSAMP = 5
BASE = [("1900", RANKE["1900"]), ("1913", RANKE["1913"]), ("1929", RANKE["1929"])]
INSTR = [("1913-inst", "uzh-echist-org/Ranke-4B-1913-1202-Instruct"),
         ("1929-inst", "uzh-echist-org/Ranke-4B-1929-1202-Instruct")]

try:
    SB = json.load(open(os.path.join(common.RESULTS, "scores_by_model.json"), encoding="utf-8"))["scores"]
except FileNotFoundError:
    SB = None


def code(text, item):
    t = text.lower()
    tt = sum(1 for m in item["true_markers"] if m in t)
    ff = sum(1 for m in item["false_markers"] if m in t)
    return "TRUE" if tt > ff else ("FALSE" if ff > tt else "neither")


def gen_base(model, tok, prompt):
    outs = [common.generate(model, tok, prompt, n=40, sample=False)]
    for _ in range(NSAMP):
        outs.append(common.generate(model, tok, prompt, n=40, sample=True, temp=0.8))
    return outs


def chat_prompt(tok, question):
    msgs = [{"role": "user", "content": question}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


report = {"items": {}, "by_model": {}}
for it in PRODUCTION:
    report["items"][it["id"]] = {"free_prompt": it["free_prompt"], "gens": {}, "coded": {}}

for tag, mid in BASE + INSTR:
    print(f"\n=== generation: {mid} ({tag}) ===", flush=True)
    tok, model = common.load(mid)
    is_instr = "Instruct" in mid
    for it in PRODUCTION:
        if is_instr:
            q = ("In your considered judgement, complete this thought in one sentence: "
                 + it["free_prompt"])
            prompt = chat_prompt(tok, q)
            outs = [common.generate(model, tok, prompt, n=50, sample=False)]
            for _ in range(NSAMP):
                outs.append(common.generate(model, tok, prompt, n=50, sample=True, temp=0.8))
        else:
            outs = gen_base(model, tok, it["free_prompt"])
        codes = [code(o, it) for o in outs]
        frac_true = round(sum(c == "TRUE" for c in codes) / len(codes), 3)
        frac_false = round(sum(c == "FALSE" for c in codes) / len(codes), 3)
        report["items"][it["id"]]["gens"][tag] = outs
        report["items"][it["id"]]["coded"][tag] = {"frac_true": frac_true, "frac_false": frac_false,
                                                    "greedy_code": codes[0]}
        recog = SB[tag][it["id"]]["correct_cal"] if (SB and tag in SB) else None
        print(f"  [{tag}] {it['id']:20s} recog={recog}  prod_true={frac_true} prod_false={frac_false}  "
              f"greedy: {outs[0][:70]!r}", flush=True)
    common.free(model)

json.dump(report, open(os.path.join(common.RESULTS, "production_report.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nwrote production_report.json")
