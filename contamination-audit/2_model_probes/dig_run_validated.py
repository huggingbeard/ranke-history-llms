"""
Launch the DeepSeek-validated dig battery. Each EVENT is a suspect (a confident-correct
should-not-know answer). Facets are screened to be post-cutoff-ONLY (no pre-1929 names/places),
length-matched (whole answer). Per event: k/n correct, joint binomial P(X>=k|0.5).
  REAL contamination = post-cutoff-only facets hold up (significantly > chance).
  FALSE POSITIVE     = collapses to chance (the original single hit was a fluke / old-name completion).
Run one model per process:  python dig_run_validated.py Ranke-4B-1929
"""
import os, re, math, sys, json, csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
ALL = ["uzh-echist-org/Ranke-4B-1900", "uzh-echist-org/Ranke-4B-1913", "uzh-echist-org/Ranke-4B-1929",
       "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
MODELS = [m for m in ALL if (len(sys.argv) > 1 and m.split("/")[-1] == sys.argv[1])] or ALL
SUF = sys.argv[2] if len(sys.argv) > 2 else ""
VAL = rf"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\dig_validated{SUF}.json"
OUT = rf"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\dig_validated_results{SUF}.csv"
BATTERY = json.load(open(VAL, encoding="utf-8"))


def toklen(tok, s):
    return len(tok(s, add_special_tokens=False).input_ids)


def sumlogp(model, tok, prompt, answer):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(answer, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))


def binom_ge(k, n, p=0.5):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)) if n else float("nan")


if not os.path.exists(OUT):
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["model", "cutoff", "category", "event", "year", "era", "true", "false", "gap", "correct"])

for M in MODELS:
    cut = int(re.search(r"Ranke-4B-(\d{4})", M).group(1))
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE).eval()
    print("\n" + "#" * 98)
    print(f"#### {M.split('/')[-1]} (cutoff {cut})  [each event = a suspect; POST facets are post-cutoff-ONLY]")
    print("#" * 98)
    rowsout = []; cat_pool = {}
    for (cat, event, yr, facets) in BATTERY:
        era = "post" if yr > cut else "pre"
        k = n = nomatch = 0; gaps = []
        for fac in facets:
            prompt, t, tw = fac[0], fac[1], fac[2]
            nt = toklen(tok, t)
            cand = [fa for fa in tw if toklen(tok, fa) == nt]
            if not cand:
                nomatch += 1; continue
            fa = cand[0]
            g = sumlogp(model, tok, prompt, t) - sumlogp(model, tok, prompt, fa)
            ok = g > 0; k += ok; n += 1; gaps.append(g)
            rowsout.append([M.split("/")[-1], cut, cat, event, yr, era, t.strip(), fa.strip(), round(g, 4), int(ok)])
        p = binom_ge(k, n)
        mg = sum(gaps) / n if n else 0
        cat_pool.setdefault((cat, era), [0, 0]); cat_pool[(cat, era)][0] += k; cat_pool[(cat, era)][1] += n
        if era == "post":
            verdict = ("REAL: knows post-cutoff cluster" if (p < 0.05 and k / max(n, 1) > 0.5)
                       else "collapses -> FALSE POSITIVE" if n >= 4 else "too few valid facets")
        else:
            verdict = "control (should know)"
        print(f"  [{era:<4}] {cat:<16} {event:<28} {k}/{n}  binom p={p:.3f}  mean gap={mg:+.2f}  ({nomatch} unmatched)  {verdict}")
    print("  --- category pooled (post-cutoff only) ---")
    for (cat, era), (k, n) in sorted(cat_pool.items()):
        if era == "post" and n:
            print(f"    {cat:<16} {k}/{n} = {100*k/n:.0f}%  binom p={binom_ge(k,n):.4f}")
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rowsout)
    del model
    import gc; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
print("\nDONE ->", OUT)
