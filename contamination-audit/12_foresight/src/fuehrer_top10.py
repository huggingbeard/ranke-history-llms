# -*- coding: utf-8 -*-
"""Ranke 1913 & 1929, German: "Wer war als Führer des deutschen Volkes bekannt?"
Top-10 MOST PROBABLE free continuations (beam search over the model's own vocabulary), each with its
ABSOLUTE joint probability P(continuation | prompt) = exp(sum of token log-probs). Not a softmax over a
hand-picked list -- the model's real distribution.

Two views:
  (A) free continuation of the question itself (max 12 tokens);
  (B) the answer frame "... Es war" (max 8 tokens) -- cleaner name elicitation.
Also the exact-name forced choice as a supplement.
Saves results/fuehrer_probe/top10.json.  Run:  python src/fuehrer_top10.py
"""
import os, json, math
import torch
import common
from foresight_bank import RANKE

OUT = os.path.join(common.RESULTS, "fuehrer_probe")
os.makedirs(OUT, exist_ok=True)

QUESTION = "Wer war als Führer des deutschen Volkes bekannt?"
ANSWER_FRAME = "Wer war als Führer des deutschen Volkes bekannt? Es war"
CANDIDATES = [" Kaiser Wilhelm", " Fürst Bismarck", " Paul von Hindenburg", " Adolf Hitler",
              " Friedrich der Große", " Friedrich Ebert"]
MODELS = ["1913", "1929"]


def top10(model, tok, prompt, k_new):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=k_new, num_beams=25, num_return_sequences=10,
                             no_repeat_ngram_size=3, early_stopping=True,
                             return_dict_in_generate=True, output_scores=True, pad_token_id=tok.eos_token_id)
    gen = out.sequences[:, ids.input_ids.shape[1]:]
    trans = model.compute_transition_scores(out.sequences, out.scores, out.beam_indices, normalize_logits=True)
    res = []
    for i in range(gen.shape[0]):
        toks = gen[i]
        mask = toks != tok.eos_token_id
        lp = float(trans[i][mask[:trans.shape[1]]].sum())
        txt = tok.decode(toks, skip_special_tokens=True).replace("\n", " ").strip()
        res.append({"text": txt, "logp": lp, "p": math.exp(lp)})
    res.sort(key=lambda r: -r["p"])
    return res


def softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    return [e / sum(ex) for e in ex]


data = json.load(open(os.path.join(OUT, "top10.json"), encoding="utf-8")) if os.path.exists(os.path.join(OUT, "top10.json")) else {}
for tag in MODELS:
    if tag in data:
        continue
    print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
    tok, model = common.load(RANKE[tag])
    data[tag] = {"question": top10(model, tok, QUESTION, 12),
                 "answer_frame": top10(model, tok, ANSWER_FRAME, 8),
                 "forced": {"candidates": CANDIDATES,
                            "p": softmax([common.slp(model, tok, ANSWER_FRAME, c) for c in CANDIDATES])}}
    json.dump(data, open(os.path.join(OUT, "top10.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model
    common.free()

for view, label in [("question", f'free continuation of  "{QUESTION}"'),
                    ("answer_frame", f'answer frame  "{ANSWER_FRAME} ___"')]:
    for tag in MODELS:
        print(f"\n=== Ranke {tag} — top 10 — {label} ===")
        for r in data[tag][view]:
            print(f"  P={r['p']:7.2%}   {r['text'][:120]}")
print("\n=== exact-name forced choice (supplement), answer frame ===")
print(f"  {'name':22} {'1913':>7} {'1929':>7}")
for i, c in enumerate(CANDIDATES):
    print(f"  {c.strip():22} {data['1913']['forced']['p'][i]:>6.1%} {data['1929']['forced']['p'][i]:>6.1%}")
print("\nwrote", os.path.join(OUT, "top10.json"))
