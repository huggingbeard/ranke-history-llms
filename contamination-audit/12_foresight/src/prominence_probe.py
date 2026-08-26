# -*- coding: utf-8 -*-
"""Prominence-across-time battery. Matched pairs of prominent people by domain, whose fame crosses the
1900 / 1913 / 1929 cutoffs, so each one should 'switch on' in the model when its cutoff passes the person's
rise to fame. For each person and each of the three frozen models:
  - N free continuations of "<Name> was" (fed later to an LLM judge: is it the real person?);
  - P(surname | first name) -- the clean name-recognition cliff.
Ranke 1900, 1913, 1929. Saves results/prominence/gens.json (checkpointed).
Run:  python src/prominence_probe.py
"""
import os, json, math
import torch
import common
from foresight_bank import RANKE
from prominence_bank import BATTERY, MODELS

OUT = os.path.join(common.RESULTS, "prominence")
os.makedirs(OUT, exist_ok=True)
RAW = os.path.join(OUT, "gens.json")
N, MAXNEW, BATCH, SEED = 50, 40, 10, 1929


def gen_batch(model, tok, prompt, k):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAXNEW, do_sample=True, temperature=0.8, top_p=0.95,
                             repetition_penalty=1.3, num_return_sequences=k, pad_token_id=tok.eos_token_id)
    return [tok.decode(c, skip_special_tokens=True).replace("\n", " ").strip()
            for c in out[:, ids.input_ids.shape[1]:]]


data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag in MODELS:
    if tag in data and all(p[0] in data[tag] and len(data[tag][p[0]]["gens"]) >= N for p in BATTERY):
        print(f"[{tag}] complete"); continue
    print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
    tok, model = common.load(RANKE[tag])
    torch.manual_seed(SEED)
    data.setdefault(tag, {})
    for name, dom, fy, cue, sur, ident in BATTERY:
        cell = data[tag].get(name, {})
        if "p_surname" not in cell:
            cell["p_surname"] = math.exp(common.slp(model, tok, cue, sur))
        got = cell.get("gens", [])
        while len(got) < N:
            got = got + gen_batch(model, tok, f"{name} was", min(BATCH, N - len(got)))
            cell["gens"] = got
            data[tag][name] = cell
            json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[{tag}/{name}] gens {len(got)}  P(surname)={cell['p_surname']:.1%}", flush=True)
    del tok, model
    common.free()
print("done ->", RAW)
