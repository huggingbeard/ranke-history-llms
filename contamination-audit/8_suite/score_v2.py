# -*- coding: utf-8 -*-
"""Step 3: score the LOCKED question_bank_v2.json on Ranke with the PREREGISTERED metric -- calibrated,
full-phrase, uncued forced choice. c(a)=logp(a|question)-mean_logp(a|neutral). correct = c(true) beats all
distractors. Analysis: broad POST pool (primary, vs chance 25%, +10pp), narrow clusters (BH-FDR, +20pp),
PRE controls (sanity). Writes results_v2_<model>.json. Run: python score_v2.py [hf_id ...]"""
import os, sys, json, math, gc, statistics as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = sys.argv[1:] or ["uzh-echist-org/Ranke-4B-1929", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
NEUTRALS = ["The answer is", "The answer was"]
BANK = json.load(open(os.path.join(HERE, "question_bank_v2.json"), encoding="utf-8"))["items"]


def slp(model, tok, prompt, cont):
    pid = tok(prompt, add_special_tokens=False).input_ids
    cid = tok(cont, add_special_tokens=False).input_ids
    full = pid + cid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(cid)))


def binom_ge(k, n, p=0.25):
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def ci95(k, n):  # Wilson interval
    if n == 0: return (0, 0)
    z = 1.96; p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def agg(rows, key="correct_cal"):
    k = sum(r[key] for r in rows); n = len(rows)
    lo, hi = ci95(k, n)
    return {"correct": k, "n": n, "pct": round(100 * k / n, 1) if n else 0, "ci95": [lo, hi],
            "binom_p": round(binom_ge(k, n), 4), "upper95": hi, "mean_gap": round(st.mean(r["gap_cal"] for r in rows), 3) if rows else 0}


for M in MODELS:
    short = M.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    rows = []
    for it in BANK:
        opts = [it["true"]] + it["distractors"]
        sq = {a: slp(model, tok, it["prompt"], a) for a in opts}
        s0 = {a: st.mean(slp(model, tok, nz, a) for nz in NEUTRALS) for a in opts}
        c = {a: sq[a] - s0[a] for a in opts}
        t = it["true"]; dmax_c = max(c[d] for d in it["distractors"]); dmax_q = max(sq[d] for d in it["distractors"])
        rows.append({"era": it["era"], "level": it["level"], "topic": it["topic"], "label": it["label"],
                     "gap_cal": round(c[t] - dmax_c, 3), "gap_raw": round(sq[t] - dmax_q, 3),
                     "correct_cal": int(c[t] > dmax_c), "correct_raw": int(sq[t] > dmax_q)})
    post = [r for r in rows if r["era"] == "post"]; pre = [r for r in rows if r["era"] == "pre"]
    clusters = sorted({r["topic"] for r in post if r["level"] == "narrow"})
    out = {"model": short, "metric": "calibrated_forced_choice", "neutrals": NEUTRALS,
           "broad_post": agg(post), "pre_controls": agg(pre),
           "narrow": {cl: agg([r for r in post if r["topic"] == cl]) for cl in clusters},
           "breadth": {cl: agg([r for r in post if r["topic"] == cl]) for cl in sorted({r["topic"] for r in post if r["level"] == "broad"})},
           "items": rows}
    # BH-FDR across narrow clusters
    ps = sorted(((out["narrow"][cl]["binom_p"], cl) for cl in clusters))
    m = len(ps); flagged = {}
    for i, (p, cl) in enumerate(ps, 1):
        flagged[cl] = bool(p <= (i / m) * 0.05) if m else False
    out["narrow_fdr_flag"] = flagged
    json.dump(out, open(os.path.join(HERE, f"results_v2_{short}.json"), "w"), indent=1)
    print("=" * 66); print(short)
    print(f"  PRE   {out['pre_controls']['correct']}/{out['pre_controls']['n']}  {out['pre_controls']['pct']}%  CI{out['pre_controls']['ci95']}")
    b = out["broad_post"]; print(f"  BROAD POST  {b['correct']}/{b['n']}  {b['pct']}%  CI{b['ci95']}  p={b['binom_p']}  (chance 25, +10pp=35; upper95={b['upper95']})")
    for cl in clusters:
        a = out["narrow"][cl]; print(f"    [narrow] {cl:<22} {a['correct']}/{a['n']}  {a['pct']}%  p={a['binom_p']}  FDR-flag={flagged[cl]}  (+20pp=45)")
    for cl in out["breadth"]:
        a = out["breadth"][cl]; print(f"    [breadth] {cl:<21} {a['correct']}/{a['n']}  {a['pct']}%")
    del model; gc.collect()
    if DEV == "cuda": torch.cuda.empty_cache()
print("\nwrote results_v2_*.json")
