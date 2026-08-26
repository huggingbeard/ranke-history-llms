# -*- coding: utf-8 -*-
"""Ask Ranke-1913 and Ranke-1929, in German: "Wer war als Führer des deutschen Volkes bekannt?"
(Who was known as the leader/Führer of the German people?)

The phrase "Führer des deutschen Volkes" only became Hitler's after 1933. Pre-Nazi it meant the Kaiser,
Bismarck, or (by 1929) Hindenburg. So this reads what the phrase meant to each frozen reader, and doubles
as a contamination check: a clean 1929 model must NOT answer Hitler.

(a) FREE continuation, 100 samples per model.  (b) FORCED CHOICE over the plausible names.
Saves results/fuehrer_probe/results.json.  Run:  python src/fuehrer_probe.py
"""
import os, re, json, math
import torch
import common
from foresight_bank import RANKE

OUT = os.path.join(common.RESULTS, "fuehrer_probe")
os.makedirs(OUT, exist_ok=True)
RAW = os.path.join(OUT, "results.json")

QUESTION = "Wer war als Führer des deutschen Volkes bekannt?"
FORCED_FRAME = "Wer war als Führer des deutschen Volkes bekannt? Es war"
CANDIDATES = [" Kaiser Wilhelm", " Fürst Bismarck", " Paul von Hindenburg", " Adolf Hitler",
              " Friedrich der Große", " Friedrich Ebert"]
NAMES = [("Kaiser / Wilhelm", r"kaiser|wilhelm|hohenzollern"), ("Bismarck", r"bismarck|bismark"),
         ("Hindenburg", r"hindenburg"), ("Hitler", r"hitler|hiller|hitier"),
         ("Friedrich the Great", r"friedrich der große|friedrich ii|der alte fritz|\bfritz\b"),
         ("Ebert", r"ebert"), ("Luther", r"luther"), ("Bethmann", r"bethmann"), ("Moltke", r"moltke")]
MODELS = ["1913", "1929"]
N, MAXNEW, BATCH, SEED = 100, 40, 10, 1929


def gen_batch(model, tok, prompt, k):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAXNEW, do_sample=True, temperature=0.8, top_p=0.95,
                             repetition_penalty=1.3, num_return_sequences=k, pad_token_id=tok.eos_token_id)
    return [tok.decode(c, skip_special_tokens=True).replace("\n", " ").strip()
            for c in out[:, ids.input_ids.shape[1]:]]


def softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    return [e / sum(ex) for e in ex]


data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {"free": {}, "forced": {}}
for tag in MODELS:
    if len(data["free"].get(tag, [])) >= N and tag in data["forced"]:
        print(f"[{tag}] done"); continue
    print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
    tok, model = common.load(RANKE[tag])
    torch.manual_seed(SEED)
    got = data["free"].get(tag, [])
    while len(got) < N:
        got = got + gen_batch(model, tok, QUESTION, min(BATCH, N - len(got)))
        data["free"][tag] = got
        json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[{tag}/free] {len(got)}/{N}", flush=True)
    lp = [common.slp(model, tok, FORCED_FRAME, c) for c in CANDIDATES]
    data["forced"][tag] = {"candidates": CANDIDATES, "p": softmax(lp)}
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model
    common.free()

print(f"\n=== FORCED CHOICE: \"{FORCED_FRAME} ___\" ===")
print(f"  {'name':22} {'1913':>7} {'1929':>7}")
for i, c in enumerate(CANDIDATES):
    print(f"  {c.strip():22} {data['forced']['1913']['p'][i]:>6.1%} {data['forced']['1929']['p'][i]:>6.1%}")

print("\n=== FREE CONTINUATION: names mentioned (share of 100) ===")
print(f"  {'name':22} {'1913':>6} {'1929':>6}")
for label, rx in NAMES:
    r = {m: sum(1 for g in data["free"][m][:N] if re.search(rx, g.lower())) for m in MODELS}
    if r["1913"] or r["1929"]:
        print(f"  {label:22} {r['1913']:>5}% {r['1929']:>5}%")
for m in MODELS:
    print(f"\n--- {m}: first 8 free continuations ---")
    for g in data["free"][m][:8]:
        print("  •", g.strip()[:150])
print("\nwrote", RAW)
