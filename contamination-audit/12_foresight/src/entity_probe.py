# -*- coding: utf-8 -*-
"""General free-continuation probe: ask Ranke-1913 and Ranke-1929 "who is <ENTITY>", 100 times each,
in English and German, raw completion. Loads each model once and does both languages while it is on the
GPU. Resume-safe: results/entity_probe/<slug>_<lang>.json checkpointed per batch.
Run:  python src/entity_probe.py bismarck        (both languages)
"""
import os, json, sys
import torch
import common
from foresight_bank import RANKE

OUT = os.path.join(common.RESULTS, "entity_probe")
os.makedirs(OUT, exist_ok=True)

ENTITIES = {"hitler": "Adolf Hitler", "bismarck": "Otto von Bismarck"}
SLUG = (sys.argv[1] if len(sys.argv) > 1 else "bismarck").lower()
NAME = ENTITIES[SLUG]
LANGS = {"en": {"who_is": f"Who is {NAME}?", "def": f"{NAME} is"},
         "de": {"who_is": f"Wer ist {NAME}?", "def": f"{NAME} ist"}}
MODELS = ["1913", "1929"]
N, MAXNEW, BATCH, SEED = 100, 60, 10, 1929


def raw_path(lang):
    return os.path.join(OUT, f"{SLUG}_{lang}.json")


def load_raw(lang):
    p = raw_path(lang)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def gen_batch(model, tok, prompt, k):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAXNEW, do_sample=True, temperature=0.8, top_p=0.95,
                             repetition_penalty=1.3, num_return_sequences=k, pad_token_id=tok.eos_token_id)
    cont = out[:, ids.input_ids.shape[1]:]
    return [tok.decode(c, skip_special_tokens=True).replace("\n", " ").strip() for c in cont]


data = {lang: load_raw(lang) for lang in LANGS}
for tag in MODELS:
    done = all(len(data[lang].get(tag, {}).get(p, [])) >= N for lang in LANGS for p in LANGS[lang])
    if done:
        print(f"[{tag}] complete, skipping"); continue
    print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
    tok, model = common.load(RANKE[tag])
    torch.manual_seed(SEED)
    for lang in LANGS:
        data[lang].setdefault(tag, {})
        for p, ptext in LANGS[lang].items():
            got = data[lang][tag].get(p, [])
            while len(got) < N:
                got = got + gen_batch(model, tok, ptext, min(BATCH, N - len(got)))
                data[lang][tag][p] = got
                json.dump(data[lang], open(raw_path(lang), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                print(f"[{tag}/{lang}/{p}] {len(got)}/{N}", flush=True)
    del tok, model
    common.free()
print("done:", {lang: raw_path(lang) for lang in LANGS})
