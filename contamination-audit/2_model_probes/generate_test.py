# -*- coding: utf-8 -*-
"""Ground-truth check: GREEDY-generate (no first-name cue) and see what the model actually says.
If it produces the correct post-cutoff answer, it KNOWS -- regardless of any twin-based score.
Also report the rank of the true answer's first token. Saves 5_results/generate_test.csv."""
import os, re, csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
MODELS = ["uzh-echist-org/Ranke-4B-1929", "uzh-echist-org/Ranke-4B-1929-1202-Instruct", "uzh-echist-org/Ranke-4B-1900"]
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\generate_test.csv"

# (prompt with NO first-name cue, the correct post-cutoff answer, its first token to rank)
PROMPTS = [
 ("The U.S. president during the Cuban Missile Crisis of 1962 was", "Kennedy"),
 ("The U.S. president who ordered the atomic bombings of Japan in 1945 was", "Truman"),
 ("The U.S. president who resigned because of the Watergate scandal was", "Nixon"),
 ("The Soviet leader during the Cuban Missile Crisis was", "Khrushchev"),
 ("The first person to walk on the Moon was", "Armstrong"),
 ("The British prime minister for most of the Second World War was", "Churchill"),
 ("The U.S. president elected in 2008 was", "Obama"),
 ("Germany's 1941 invasion of the Soviet Union was code-named Operation", "Barbarossa"),
 ("The atomic bomb dropped on Hiroshima was nicknamed", "Little Boy"),
 ("The 1980s Soviet policy of openness was called", "glasnost"),
]
with open(OUT, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["model", "prompt", "expected", "greedy_generation", "rank_of_expected_first_token", "p_first_pct"])
for M in MODELS:
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    V = model.config.vocab_size
    print("\n#### " + M.split("/")[-1], flush=True)
    rows = []
    for (p, exp) in PROMPTS:
        ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")
            logits = model(ids).logits[0, -1].float()
        probs = torch.softmax(logits, -1)
        ft = tok(" " + exp, add_special_tokens=False).input_ids[0]
        rank = int((probs > probs[ft]).sum().item()) + 1
        hit = "  <== correct" if exp.split()[0].lower() in gen.lower() else ""
        print(f'  "{p[-44:]}" -> "{gen}"   [rank({exp})={rank}/{V}, {probs[ft].item()*100:.1f}%]{hit}', flush=True)
        rows.append([M.split("/")[-1], p, exp, gen, rank, round(probs[ft].item() * 100, 2)])
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    del model
    import gc; gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
print("\nSAVED ->", OUT)
