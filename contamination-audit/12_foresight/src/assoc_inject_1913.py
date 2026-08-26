# -*- coding: utf-8 -*-
"""SENSITIVITY CERTIFICATE for the association probe: plant Hitler->Stalin into the CLEAN 1913 model.

The 1913 model is provably cold on this association (assoc_lens_pilot_1913.json: Stalin rank ~38k at
the Hitler carrier, indistinguishable from controls). We surgically write the association into the
weights (ROME-style rank-1 edit, the exact mechanics validated on treaties in rome_inject.py), then
re-run the association reading. The edit is IN MEMORY ONLY — nothing is saved to disk.

PRE-REGISTERED SUCCESS CRITERIA (written before running):
  S1  SENSITIVITY: on held-out free-association carriers (NOT the edit template), Stalin's rank at
      "Hitler" improves from ~38k to clearly inside the controls' range — ideally toward the ~1-2k
      region where the contaminated 1929 model sits at the output layer.
  S2  KEY SPECIFICITY: Stalin's rank at the BISMARCK carrier stays ~unchanged (the edit is keyed on
      Hitler, and must not leak to other names).
  S3  TELL SPECIFICITY: other tells at Hitler (Nazis, Eisenhower) stay ~unchanged (we planted ONE
      association, not general warmth).
  S4  COLLATERAL: real period knowledge survives (Treaty of Portsmouth 1905, Treaty of Frankfurt 1871
      date probes ~unchanged).
If S1 holds with S2-S4, the free-association reading is certified to detect a genuinely stored
association, specifically. If S1 fails, the probe cannot see planted associations and yesterday's
"1913 is cold" would be uninterpretable.

Edit mechanics (ported from rome_inject.py): optimize a delta z on the layer-9 MLP down-projection
output at the subject token so the model emits " Stalin" after the edit template, then apply the
rank-1 weight update keyed on the subject's down_proj input. Edit template != measurement carriers.
Run:  python src/assoc_inject_1913.py
"""
import os, json
import numpy as np
import torch
import common
from foresight_bank import RANKE

MID = RANKE["1913"]
OUT = os.path.join(common.RESULTS, "superlatives", "assoc_inject_1913.json")
L_EDIT = 9
LAYERS_READ = [33, 36]

EDIT_TMPL = "The great rival of Hitler was"        # the ONE training-like sentence we plant
EDIT_TARGET = " Stalin"

CARRIERS = {   # held-out measurement carriers (none is the edit template)
    "Hitler_A":   "The name on everyone's lips was Hitler",
    "Hitler_B":   "In those days the newspapers were full of Hitler",
    "Bismarck":   "The name on everyone's lips was Bismarck",
    "Mueller":    "The name on everyone's lips was Müller",
}
READ_TOKENS = ["Stalin", "Nazis", "Eisenhower", "Munich"]
PERIOD_FACTS = [("The Treaty of Portsmouth was signed in the year", 1905),
                ("The Treaty of Frankfurt was signed in the year", 1871)]
YEAR_WIN = 8

tok, model = common.load(MID)
for p in model.parameters():
    p.requires_grad_(False)
emb_out = model.get_output_embeddings().weight
final_norm = model.model.norm


def one_id(w):
    ids = tok(" " + w, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None


READ_IDS = {w: one_id(w) for w in READ_TOKENS}


def read_ranks():
    out = {}
    for cname, carrier in CARRIERS.items():
        ids = tok(carrier, add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.no_grad():
            hs = model(**ids, output_hidden_states=True).hidden_states
        pos = ids.input_ids.shape[1] - 1
        per = {}
        for L in LAYERS_READ:
            with torch.no_grad():
                logits = emb_out.float() @ final_norm(hs[L][0, pos]).float()
                order = torch.argsort(logits, descending=True)
                rk = {int(t): r for r, t in enumerate(order[:60000].tolist())}
            per[L] = {w: rk.get(i, ">60k") for w, i in READ_IDS.items() if i is not None}
        out[cname] = per
        del hs
    return out


def year_probe(prompt, year):
    yrs = list(range(year - YEAR_WIN, year + YEAR_WIN + 1))
    import math
    xs = common.slp_batch(model, tok, prompt, [f" {y}" for y in yrs])
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e)
    return round(e[yrs.index(year)] / s, 3)


def spoken(prompt, k=5):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        lp = torch.log_softmax(model(**ids).logits[0, -1].float(), dim=-1)
    top = torch.topk(lp, k)
    return [(tok.decode([int(t)]), round(float(v), 2)) for v, t in zip(top.values, top.indices)]


print("=== PRE-EDIT measurements ===", flush=True)
pre = {"ranks": read_ranks(),
       "facts": {p: year_probe(p, y) for p, y in PERIOD_FACTS},
       "spoken_edit_tmpl": spoken(EDIT_TMPL),
       "spoken_bismarck": spoken("The great rival of Bismarck was")}

# ---------------- the edit (ported from rome_inject.py) ----------------
subj_pos = len(tok("The great rival of Hitler", add_special_tokens=False).input_ids) - 1
pids = tok(EDIT_TMPL, add_special_tokens=False).input_ids
tids = tok(EDIT_TARGET, add_special_tokens=False).input_ids
full = torch.tensor([pids + tids], device=model.device)
tgt_pos = list(range(len(pids) - 1, len(pids) - 1 + len(tids)))
down = model.model.layers[L_EDIT].mlp.down_proj
z = torch.zeros(down.weight.shape[0], device=model.device, dtype=torch.float32, requires_grad=True)
opt = torch.optim.Adam([z], lr=0.5)


def mlp_hook(mod, inp, out):
    o = out.clone()
    o[0, subj_pos, :] = o[0, subj_pos, :] + z.to(o.dtype)
    return o


h = model.model.layers[L_EDIT].mlp.register_forward_hook(mlp_hook)
for step in range(28):
    opt.zero_grad()
    logits = model(full).logits[0].float()
    loss = sum(-torch.log_softmax(logits[p], -1)[tids[i]] for i, p in enumerate(tgt_pos)) + 0.02 * (z.norm() ** 2)
    loss.backward()
    opt.step()
h.remove()

kbuf = {}
hp = down.register_forward_pre_hook(lambda m, inp: kbuf.__setitem__("k", inp[0][0, subj_pos, :].detach().float()))
with torch.no_grad():
    model(torch.tensor([pids], device=model.device))
hp.remove()
k = kbuf["k"]
with torch.no_grad():
    down.weight.add_((torch.outer(z.detach(), k) / (k @ k)).to(down.weight.dtype))
print(f"=== EDIT applied (|z| = {float(z.detach().norm()):.2f}; in-memory only) ===", flush=True)

print("=== POST-EDIT measurements ===", flush=True)
post = {"ranks": read_ranks(),
        "facts": {p: year_probe(p, y) for p, y in PERIOD_FACTS},
        "spoken_edit_tmpl": spoken(EDIT_TMPL),
        "spoken_bismarck": spoken("The great rival of Bismarck was")}

del model
common.free()

res = {"meta": {"model": MID, "edit_template": EDIT_TMPL, "edit_target": EDIT_TARGET,
                "layer_edit": L_EDIT, "layers_read": LAYERS_READ,
                "criteria": "S1 sensitivity / S2 key specificity / S3 tell specificity / S4 collateral"},
       "pre": pre, "post": post}
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("\n================ RESULT (layer 36 = the mouth; layer 33 in JSON) ================")
print(f"{'reading':44s}{'PRE':>12s}{'POST':>12s}")
for cname in CARRIERS:
    for w in READ_TOKENS:
        if READ_IDS[w] is None:
            continue
        a = pre["ranks"][cname][36][w]
        b = post["ranks"][cname][36][w]
        print(f"  {w:12s} @ {cname:28s}{str(a):>12s}{str(b):>12s}")
print("\nperiod facts (P(true year)):")
for p, y in PERIOD_FACTS:
    print(f"  {p[:50]:52s}{pre['facts'][p]:>10.3f}{post['facts'][p]:>10.3f}")
print(f"\nspoken '{EDIT_TMPL}' pre : {pre['spoken_edit_tmpl']}")
print(f"spoken '{EDIT_TMPL}' post: {post['spoken_edit_tmpl']}")
print(f"spoken Bismarck-rival post: {post['spoken_bismarck']}")
print(f"\nwrote {OUT}")
