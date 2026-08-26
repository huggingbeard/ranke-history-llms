# -*- coding: utf-8 -*-
"""Frame probe at scale: many SAMPLED continuations per idea, so each 'runs with it' rate becomes a real
proportion with a confidence interval. Local Ranke 1900/1913/1929 (base) only (per user: skip paid Talkie).

For every idea (13) x phrasing (4) we draw K=10 sampled use-it continuations (temp 0.7) -> ~40 per idea per
model -> 520/model -> 1560 total. Memory-safe: one model resident, batch=1 generation (coexists with the
port-8000 server). Checkpointed + resumable (save after each model/dim). Judged later by Gemini.

Output: frame_probe_N_gens.json
"""
import os, json, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "frame_probe_N_gens.json")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
MODELS = {"1900": "uzh-echist-org/Ranke-4B-1900", "1913": "uzh-echist-org/Ranke-4B-1913", "1929": "uzh-echist-org/Ranke-4B-1929"}
K = 10            # samples per phrasing -> x4 phrasings = 40 per idea
LEAD = " What followed from this was that"

v2 = json.load(open(os.path.join(HERE, "frame_probe_v2.json"), encoding="utf-8"))
BYDIM = {}
for it in v2["items"]:
    BYDIM.setdefault(it["dim"], []).append(it)
for d in BYDIM:
    BYDIM[d] = sorted(BYDIM[d], key=lambda x: x["id"])     # 4 phrasings, stable order
DIMS = list(BYDIM)

# resume
state = {"models": MODELS, "K": K, "lead": LEAD, "gens": {}}
if os.path.exists(OUT):
    state = json.load(open(OUT, encoding="utf-8"))
    state.setdefault("gens", {})
done = {(m, d) for m in state["gens"] for d in state["gens"][m]}
print(f"resume: {len(done)} (model,dim) cells already done", flush=True)


def gen_one(model, tok, prompt, seed):
    torch.manual_seed(seed)
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=45, do_sample=True, temperature=0.7, top_p=0.95,
                             repetition_penalty=1.15, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


for tag, mid in MODELS.items():
    if all((tag, d) in done for d in DIMS):
        continue
    print(f"\n=== {mid} ({tag}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    state["gens"].setdefault(tag, {})
    for d in DIMS:
        if (tag, d) in done:
            continue
        rows = []
        for i, it in enumerate(BYDIM[d]):
            prompt = it["plant"] + LEAD
            for s in range(K):
                txt = gen_one(model, tok, prompt, seed=1000 * i + s + hash(tag) % 997)
                rows.append({"i": i, "s": s, "text": txt})
        state["gens"][tag][d] = rows
        json.dump(state, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)   # checkpoint per dim
        print(f"  {d:18s} {len(rows)} gens", flush=True)
    del model; gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()

tot = sum(len(state["gens"][m][d]) for m in state["gens"] for d in state["gens"][m])
print(f"\nDONE: {tot} generations across {len(state['gens'])} models -> {OUT}", flush=True)
