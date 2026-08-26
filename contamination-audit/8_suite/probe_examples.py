# -*- coding: utf-8 -*-
"""Score a handful of ILLUSTRATIVE items on Ranke-4B-1929 locally and dump the FULL per-option
log-probabilities (true + every distractor), so the integrated report can show, end to end, what
a single forced-choice probe actually looks like and how the model scores each alternative.

Raw forced-choice (sum-logp of the whole continuation) — the project's settled instrument.
Output: probe_examples.json  -> consumed by build_report.py
"""
import os, sys, json, math, statistics as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "uzh-echist-org/Ranke-4B-1929"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32

bank = {(b["era"], b["topic"], b["label"]): b for b in json.load(open(os.path.join(HERE, "question_bank_v2.json"), encoding="utf-8"))["items"]}
ranke = json.load(open(os.path.join(HERE, "results_v2_Ranke-4B-1929.json"), encoding="utf-8"))["items"]
ridx = {(r["era"], r["topic"], r["label"]): r for r in ranke}

# ---- choose illustrative items from the data ----
post = [r for r in ranke if r["era"] == "post"]
pre = [r for r in ranke if r["era"] == "pre"]
cr = sorted([r for r in post if r["topic"] == "US civil rights"], key=lambda r: -r["gap_raw"])
clean_pool = [r for r in post if r["topic"] in ("Cold War", "Vietnam War", "science & medicine", "space & computing")]

picks = []
def add(r, tag):
    if r and (r["era"], r["topic"], r["label"]) not in {(p["era"], p["topic"], p["label"]) for p, _ in picks}:
        picks.append((r, tag))

add(max(pre, key=lambda r: r["gap_raw"]), "pre-control — probe must fire (model SHOULD know this)")
add(cr[0], "LEAK — contaminated cluster, model clearly knows a post-cutoff fact")
add(cr[1], "LEAK — second hit in the same contaminated cluster")
add(min(clean_pool, key=lambda r: r["gap_raw"]), "CLEAN — post-cutoff fact the model does NOT know")
add(min(post, key=lambda r: abs(r["gap_raw"])), "BORDERLINE — near tie between true and a distractor")

print("loading", MODEL, "on", DEV)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()


def slp(prompt, cont):
    pid = tok(prompt, add_special_tokens=False).input_ids
    cid = tok(cont, add_special_tokens=False).input_ids
    full = pid + cid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(cid))), len(cid)


out = []
for r, tag in picks:
    b = bank[(r["era"], r["topic"], r["label"])]
    opts = [(b["true"], True)] + [(d, False) for d in b["distractors"]]
    scored = []
    for text, is_true in opts:
        lp, ntok = slp(b["prompt"], text)
        scored.append({"text": text.strip(), "logp": round(lp, 2), "ntok": ntok, "is_true": is_true})
    picked = max(scored, key=lambda s: s["logp"])
    picked["picked"] = True
    out.append({"tag": tag, "era": r["era"], "topic": r["topic"], "label": r["label"],
                "prompt": b["prompt"], "options": scored,
                "model_correct": bool(picked["is_true"]), "gap_raw": r["gap_raw"]})
    print(f"  [{ 'OK' if picked['is_true'] else 'XX'}] {r['topic']}/{r['label']}: picked {picked['text']!r} (true={[o['text'] for o in scored if o['is_true']][0]!r})")

# ---- length audit over the whole bank (tokenizer already loaded) -> feeds the length-bias caveat ----
def ntok(s):
    return len(tok(s, add_special_tokens=False).input_ids)


la_post = [b for b in bank.values() if b["era"] == "post"]
mism = sum(1 for b in la_post if abs(ntok(b["true"]) - st.median([ntok(d) for d in b["distractors"]])) >= 2)
longer = sum(1 for b in la_post if ntok(b["true"]) > max(ntok(d) for d in b["distractors"]))
shorter = sum(1 for b in la_post if ntok(b["true"]) < min(ntok(d) for d in b["distractors"]))
length_audit = {"post_n": len(la_post), "mismatch_ge2": mism, "true_longer_all": longer, "true_shorter_all": shorter}
print("length audit:", length_audit)

json.dump({"model": MODEL.split("/")[-1], "metric": "raw sum-logp forced choice",
           "examples": out, "length_audit": length_audit},
          open(os.path.join(HERE, "probe_examples.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote probe_examples.json with", len(out), "worked examples")
