# -*- coding: utf-8 -*-
"""Same question, corrected tense + the bare title "Führer".

"Wer WAR ... bekannt?" pointed the model at the past (the Kaiser, Frederick the Great). But Hitler as
"Führer des deutschen Volkes" is a FUTURE event from 1913/1929, so ask in the present ("ist") and future
("wird sein"). And drop "des deutschen Volkes": bare "Führer" / "Der Führer" is the title that actually
became Hitler's, so if the word leaks Hitler anywhere it is here, above all in the 1929 model.

Frames tested (answer-completion form), both models:
  volk_present/future : "... als Führer des deutschen Volkes bekannt ... Es ist / Es wird"
  fuehrer_present/future : "... als Führer bekannt ... Es ist / Es wird"
  der_fuehrer : "Der Führer ist"     (the most Hitler-evocative)
Top-10 most probable free continuations (beam, absolute joint probability) + exact-name forced choice.
Saves results/fuehrer_probe/tense.json.  Run:  python src/fuehrer_tense.py
"""
import os, json, math
import torch
import common
from foresight_bank import RANKE

OUT = os.path.join(common.RESULTS, "fuehrer_probe")
os.makedirs(OUT, exist_ok=True)

FRAMES = {
    "volk_present":    "Wer ist als Führer des deutschen Volkes bekannt? Es ist",
    "volk_future":     "Wer wird als Führer des deutschen Volkes bekannt sein? Es wird",
    "fuehrer_present": "Wer ist als Führer bekannt? Es ist",
    "fuehrer_future":  "Wer wird als Führer bekannt sein? Es wird",
    "der_fuehrer":     "Der Führer ist",
}
CANDIDATES = [" Kaiser Wilhelm", " Fürst Bismarck", " Paul von Hindenburg", " Adolf Hitler",
              " Friedrich der Große", " Friedrich Ebert"]
MODELS = ["1913", "1929"]


def top10(model, tok, prompt, k_new=8):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=k_new, num_beams=25, num_return_sequences=10,
                             no_repeat_ngram_size=3, early_stopping=True,
                             return_dict_in_generate=True, output_scores=True, pad_token_id=tok.eos_token_id)
    gen = out.sequences[:, ids.input_ids.shape[1]:]
    trans = model.compute_transition_scores(out.sequences, out.scores, out.beam_indices, normalize_logits=True)
    res = []
    for i in range(gen.shape[0]):
        mask = gen[i] != tok.eos_token_id
        lp = float(trans[i][mask[:trans.shape[1]]].sum())
        res.append({"text": tok.decode(gen[i], skip_special_tokens=True).replace("\n", " ").strip(),
                    "p": math.exp(lp)})
    res.sort(key=lambda r: -r["p"])
    return res


def softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    return [e / sum(ex) for e in ex]


data = json.load(open(os.path.join(OUT, "tense.json"), encoding="utf-8")) if os.path.exists(os.path.join(OUT, "tense.json")) else {}
for tag in MODELS:
    if tag in data and all(f in data[tag] for f in FRAMES):
        continue
    print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
    tok, model = common.load(RANKE[tag])
    data.setdefault(tag, {})
    for name, frame in FRAMES.items():
        data[tag][name] = {"top10": top10(model, tok, frame),
                           "forced": softmax([common.slp(model, tok, frame, c) for c in CANDIDATES])}
    json.dump(data, open(os.path.join(OUT, "tense.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model
    common.free()

for name, frame in FRAMES.items():
    for tag in MODELS:
        print(f"\n=== Ranke {tag} — {name} — \"{frame} ___\" ===")
        for r in data[tag][name]["top10"]:
            print(f"  P={r['p']:7.2%}   {r['text'][:105]}")
print("\n=== forced-choice P(name), by frame ===")
print(f"  {'frame':16} {'model':5} | " + " ".join(f"{c.strip()[:10]:>11}" for c in CANDIDATES))
for name in FRAMES:
    for tag in MODELS:
        row = " ".join(f"{data[tag][name]['forced'][i]:>10.1%}" for i in range(len(CANDIDATES)))
        print(f"  {name:16} {tag:5} | {row}")
print("\nwrote", os.path.join(OUT, "tense.json"))
