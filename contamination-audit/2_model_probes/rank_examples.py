# -*- coding: utf-8 -*-
"""Compute REAL first-token ranks (1929 base) for a few illustrative items, to make the
'rank the true token' step concrete AND show the multi-token sub-word flaw."""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
M = "uzh-echist-org/Ranke-4B-1929"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
V = model.config.vocab_size
items = [
 ("should-know", "The German-born physicist of general relativity was Albert", " Einstein"),
 ("should-know", "The English naturalist who wrote On the Origin of Species was Charles", " Darwin"),
 ("post-cutoff", "The U.S. president who ordered the atomic bombing of Japan was Harry", " Truman"),
 ("post-cutoff", "The U.S. president inaugurated in 1961, the youngest elected, was John F.", " Kennedy"),
 ("post-cutoff", "The Soviet leader of the era of stagnation from 1964 was Leonid", " Brezhnev"),
]
import csv
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\rank_examples.csv"
print(f"vocab size = {V}\n")
rows = []
for tag, p, ans in items:
    pid = tok(p, add_special_tokens=False).input_ids
    aid = tok(ans, add_special_tokens=False).input_ids
    with torch.no_grad():
        logits = model(torch.tensor([pid], device="cuda")).logits[0, -1].float()
    probs = torch.softmax(logits, -1)
    ft = aid[0]
    rank = int((probs > probs[ft]).sum().item()) + 1
    toks = [tok.decode([t]) for t in aid]
    rows.append([tag, ans.strip(), len(aid), tok.decode([ft]).strip(), rank, round(probs[ft].item() * 100, 2), p])
    print(f"[{tag}] {p!r} -> {ans.strip()!r}")
    print(f"    tokenizes to {toks}  ({len(aid)} token{'s' if len(aid)>1 else ''})")
    print(f"    FIRST token = {tok.decode([ft])!r}  rank {rank} / {V}   p(first)={probs[ft].item()*100:.2f}%\n")
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["tag", "answer", "n_tokens", "first_token", "rank", "p_first_pct", "prompt"]); w.writerows(rows)
print("SAVED ->", OUT, "(read this; do not recompute)")
