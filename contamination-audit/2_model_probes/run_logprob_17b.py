"""
Teacher-forced minimal-pair log-prob test for uzh-echist-org/ranke-1.7b-1929-3006 (cutoff 1929).
Reuses the 230 items from contamination_map.py; runs on GPU; writes a NEW csv (4B results untouched).
Metric: mean per-token logp(true) - logp(false). On a POST-1929 fact, gap>0 = contamination.
"""
import os, sys, csv, math, statistics
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
from contamination_map import ITEMS   # the 230 tagged true/false minimal pairs

MODEL = "uzh-echist-org/ranke-1.7b-1929-3006"
CUTOFF = 1929
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\contamination_map_ranke-1.7b-1929.csv"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32

print(f"loading {MODEL} on {DEV} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()


@torch.no_grad()
def avg_logp(prompt, answer):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(answer, add_special_tokens=False).input_ids
    full = pid + aid
    logp = torch.log_softmax(model(torch.tensor([full], device=DEV)).logits[0].float(), -1)
    tot = sum(logp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))
    return tot / len(aid)


rows = []
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "cutoff", "country", "langfam", "domain", "year", "era", "true", "false", "gap", "p_true"])
    for i, (country, lf, dom, year, prompt, t, fa) in enumerate(ITEMS):
        try:
            gap = avg_logp(prompt, t) - avg_logp(prompt, fa)
            pt = math.exp(avg_logp(prompt, t))
        except Exception as e:
            print("  skip item", i, repr(e)[:80]); continue
        era = "pre" if year <= CUTOFF else "post"
        w.writerow([MODEL.split("/")[-1], CUTOFF, country, lf, dom, year, era, t.strip(), fa.strip(),
                    round(gap, 4), round(pt, 5)])
        rows.append((country, lf, dom, year, era, gap))
        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{len(ITEMS)}", flush=True)


def mean(xs): return statistics.mean(xs) if xs else float("nan")
pre = [g for *_, e, g in rows if e == "pre"]; post = [g for *_, e, g in rows if e == "post"]
print("\n" + "=" * 70)
print(f"MODEL {MODEL}  (cutoff {CUTOFF})   n={len(rows)} items scored")
print("=" * 70)
print(f"PRE  (should know, n={len(pre)}):  mean gap {mean(pre):+.2f}   prefers-truth {sum(g>0 for g in pre)}/{len(pre)}")
print(f"POST (should NOT,  n={len(post)}):  mean gap {mean(post):+.2f}   prefers-truth {sum(g>0 for g in post)}/{len(post)}   <-- >0 = contamination")
print("\nPOST-cutoff leakage by LANGUAGE FAMILY:")
bylf = defaultdict(list)
for c, lf, dom, y, e, g in rows:
    if e == "post": bylf[lf].append(g)
for lf in sorted(bylf, key=lambda k: -mean(bylf[k])):
    print(f"   {lf:<14} n={len(bylf[lf]):<3} mean {mean(bylf[lf]):+.2f}   prefers-truth {sum(g>0 for g in bylf[lf])}/{len(bylf[lf])}")
print("\nPOST-cutoff leakage by DOMAIN:")
byd = defaultdict(list)
for c, lf, dom, y, e, g in rows:
    if e == "post": byd[dom].append(g)
for d in sorted(byd, key=lambda k: -mean(byd[k])):
    print(f"   {d:<12} n={len(byd[d]):<3} mean {mean(byd[d]):+.2f}")
print("\nPRE control by DOMAIN (sanity — should be strongly positive):")
byp = defaultdict(list)
for c, lf, dom, y, e, g in rows:
    if e == "pre": byp[dom].append(g)
for d in sorted(byp, key=lambda k: -mean(byp[k])):
    print(f"   {d:<12} n={len(byp[d]):<3} mean {mean(byp[d]):+.2f}")
print("\nwrote", OUT)
