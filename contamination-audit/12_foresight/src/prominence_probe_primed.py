# -*- coding: utf-8 -*-
"""Disambiguated ('primed') prominence probe. The bare "<Name> was" frame is hijacked by ordinary people
of the same name in the corpus (Richard Wagner the bicycle thief). So prime the intended sense with a
period-NEUTRAL role: "The <role> <Name> was ___". The role fixes WHICH Wagner/Einstein without granting
any post-cutoff knowledge. 50 continuations per person per model. Saves into gens.json under key
'gens_primed' (alongside the naive 'gens'). Run:  python src/prominence_probe_primed.py
"""
import os, json
import torch
import common
from foresight_bank import RANKE
from prominence_bank import BATTERY, MODELS

OUT = os.path.join(common.RESULTS, "prominence")
RAW = os.path.join(OUT, "gens.json")
N, MAXNEW, BATCH, SEED = 50, 40, 10, 1929

# period-neutral role that disambiguates the name (does NOT reveal the post-cutoff achievement)
ROLE = {
 "Paul Gauguin": "French painter", "Pablo Picasso": "Spanish painter", "Jackson Pollock": "American painter",
 "Otto von Bismarck": "German statesman", "Adolf Hitler": "German political agitator", "Ronald Reagan": "American actor",
 "James Clerk Maxwell": "British physicist", "Albert Einstein": "German physicist", "Richard Feynman": "American physicist",
 "Richard Wagner": "German composer", "Igor Stravinsky": "Russian composer", "Elvis Presley": "American singer",
 "Leo Tolstoy": "Russian novelist", "Ernest Hemingway": "American novelist", "George Orwell": "English writer",
 "Karl Marx": "German socialist", "Vladimir Lenin": "Russian revolutionary", "Che Guevara": "South American revolutionary",
 "Louis Pasteur": "French chemist", "Marie Curie": "French physicist", "Jonas Salk": "American doctor",
 "Henry Stanley": "British explorer", "Charles Lindbergh": "American aviator", "Neil Armstrong": "American pilot",
 "Friedrich Nietzsche": "German philosopher", "Ludwig Wittgenstein": "Austrian philosopher", "Jean-Paul Sartre": "French philosopher",
 "Napoleon Bonaparte": "French general", "Benito Mussolini": "Italian politician", "Mao Zedong": "Chinese revolutionary",
 "Frederick Douglass": "American reformer", "W. E. B. Du Bois": "American writer", "Rosa Parks": "American woman",
}


def gen_batch(model, tok, prompt, k):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAXNEW, do_sample=True, temperature=0.8, top_p=0.95,
                             repetition_penalty=1.3, num_return_sequences=k, pad_token_id=tok.eos_token_id)
    return [tok.decode(c, skip_special_tokens=True).replace("\n", " ").strip()
            for c in out[:, ids.input_ids.shape[1]:]]


data = json.load(open(RAW, encoding="utf-8"))
for tag in MODELS:
    if all("gens_primed" in data[tag].get(p[0], {}) and len(data[tag][p[0]]["gens_primed"]) >= N for p in BATTERY):
        print(f"[{tag}] primed complete"); continue
    print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
    tok, model = common.load(RANKE[tag])
    torch.manual_seed(SEED)
    for name, dom, fy, cue, sur, ident in BATTERY:
        cell = data[tag][name]
        got = cell.get("gens_primed", [])
        frame = f"The {ROLE[name]} {name} was"
        while len(got) < N:
            got = got + gen_batch(model, tok, frame, min(BATCH, N - len(got)))
            cell["gens_primed"] = got
            data[tag][name] = cell
            json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[{tag}/{name}] primed {len(got)}  frame=\"{frame} ...\"", flush=True)
    del tok, model
    common.free()
print("done -> gens_primed in", RAW)
