# -*- coding: utf-8 -*-
"""Integrated contamination suite -- Ranke (transformers) side. Scores the shared battery (batteries.py)
with whole-answer sum-logp forced choice, writes question_bank.json (the file Guohui scores on Talkie) and
results_<model>.json. Run: python suite.py            (both Ranke 1929 base + Instruct)
                          python suite.py <hf_id> ...  (any HF causal LM)"""
import os, sys, re, csv, json, math, gc, statistics as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batteries
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = sys.argv[1:] or ["uzh-echist-org/Ranke-4B-1929", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
ITEMS = batteries.all_items()
json.dump({"schema": "forced_choice: pick true over 3 matched distractors; chance=0.25",
           "n": len(ITEMS), "items": ITEMS}, open(os.path.join(HERE, "question_bank.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)


def slp(model, tok, prompt, cont):
    pid = tok(prompt, add_special_tokens=False).input_ids
    cid = tok(cont, add_special_tokens=False).input_ids
    full = pid + cid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(cid)))


def binom_ge(k, n, p=0.25):
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def aggregate(rows):
    agg = {}
    for era in ("pre", "post"):
        for topic in sorted({r["topic"] for r in rows if r["era"] == era}):
            sub = [r for r in rows if r["era"] == era and r["topic"] == topic]
            k = sum(r["correct"] for r in sub); n = len(sub)
            agg[f"{era}:{topic}"] = {"correct": k, "n": n, "pct": round(100 * k / n, 1),
                                     "chance_pct": 25.0, "binom_p": round(binom_ge(k, n), 4),
                                     "mean_gap": round(st.mean(r["gap_best"] for r in sub), 3)}
        sub = [r for r in rows if r["era"] == era]
        if sub:
            k = sum(r["correct"] for r in sub); n = len(sub)
            agg[f"{era}:ALL"] = {"correct": k, "n": n, "pct": round(100 * k / n, 1),
                                 "chance_pct": 25.0, "binom_p": round(binom_ge(k, n), 4),
                                 "mean_gap": round(st.mean(r["gap_best"] for r in sub), 3)}
    return agg


for M in MODELS:
    short = M.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    rows = []
    for it in ITEMS:
        t = slp(model, tok, it["prompt"], it["true"]); tw = [slp(model, tok, it["prompt"], d) for d in it["distractors"]]
        gb = t - max(tw)
        rows.append({**{k: it[k] for k in ("era", "topic", "label")}, "gap_best": round(gb, 3),
                     "gap_mean": round(t - st.mean(tw), 3), "correct": int(gb > 0)})
    agg = aggregate(rows)
    json.dump({"model": short, "metric": "forced_choice_sumlogp", "aggregate": agg, "items": rows},
              open(os.path.join(HERE, f"results_{short}.json"), "w", encoding="utf-8"), indent=1)
    print("=" * 64); print(short)
    for era in ("pre", "post"):
        a = agg.get(f"{era}:ALL")
        print(f"  {era.upper():<4} ALL  {a['correct']:>2}/{a['n']}  {a['pct']:>5}%  (chance 25%)  p={a['binom_p']:.3f}  gap={a['mean_gap']:+.2f}")
        for key in [k for k in agg if k.startswith(era + ":") and not k.endswith("ALL")]:
            a = agg[key]; print(f"        {key.split(':',1)[1]:<18} {a['correct']:>2}/{a['n']}  {a['pct']:>5}%  p={a['binom_p']:.3f}")
    del model; gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
print("\nwrote question_bank.json +", ", ".join(f"results_{m.split('/')[-1]}.json" for m in MODELS))
