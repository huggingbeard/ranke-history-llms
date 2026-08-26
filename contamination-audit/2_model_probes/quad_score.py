# -*- coding: utf-8 -*-
"""Quadruplet power test: re-score the comprehensive battery computing logp(true) vs ALL
length-matched twins, so we can compare 2-way (true beats one twin; chance 50%) with k-way
(true beats ALL matched twins; chance 1/(k+1)). Saves per-item the 2-way gap and the k-way
margin so the analysis can check whether k-way LOSES the low-confidence items.
Saves 5_results/quad_results.csv. All four models in one process."""
import os, re, csv, math, json, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
MODELS = ["uzh-echist-org/Ranke-4B-1900", "uzh-echist-org/Ranke-4B-1913", "uzh-echist-org/Ranke-4B-1929", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
VAL = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\dig_validated_full.json"
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\quad_results.csv"
BAT = json.load(open(VAL, encoding="utf-8"))


def toklen(tok, s):
    return len(tok(s, add_special_tokens=False).input_ids)


def slp(model, tok, prompt, ans):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(ans, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))


with open(OUT, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["model", "cutoff", "category", "role", "true", "n_matched_twins",
                            "gap_2way", "margin_kway", "correct_2way", "correct_kway"])
for M in MODELS:
    cut = int(re.search(r"(\d{4})", M).group(1))
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    rows = []
    for (cat, role, yr, facets) in BAT:
        for fac in facets:
            prompt, true, tws = fac[0], fac[1], fac[2]
            nt = toklen(tok, true)
            matched = [x for x in tws if toklen(tok, x) == nt]
            if not matched:
                continue
            lt = slp(model, tok, prompt, true)
            ltw = [slp(model, tok, prompt, x) for x in matched]
            g2 = lt - ltw[0]                      # 2-way: true vs first matched twin
            gk = lt - max(ltw)                    # k-way: true must beat ALL matched twins
            rows.append([M.split("/")[-1], cut, cat, role, true.strip(), len(matched),
                         round(g2, 4), round(gk, 4), int(g2 > 0), int(gk > 0)])
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(M.split("/")[-1], "done", len(rows), "items", flush=True)
    del model; gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
print("DONE ->", OUT)
