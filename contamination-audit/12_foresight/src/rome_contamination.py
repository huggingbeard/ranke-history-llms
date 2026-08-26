# -*- coding: utf-8 -*-
"""Can model-EDITING firm up a contamination claim? Pilot the 'counterfactual resistance' idea on the treaty battery,
where we KNOW ground truth (known vs unknown treaties). For each treaty: inject a FALSE year via ROME on phrasing A, then
test on a PARAPHRASE B whether the TRUE year survives (a pre-existing memory resisting the lie) or the false year
generalises cleanly (no prior). Hypothesis: KNOWN treaties (contaminated-analog) resist; UNKNOWN (clean) accept the lie.
This isolates whether editing exposes an existing parametric memory — the crux of 'is it contaminated'.
Run:  python src/rome_contamination.py
"""
import os, math
import numpy as np
import torch
import common

MID, L_EDIT = "uzh-echist-org/Ranke-4B-1929", 9
# (city, true_year, ground_truth)   edit on "signed", test on "concluded"
FACTS = [("Utrecht", 1713, "known"), ("Tilsit", 1807, "known"), ("Nanking", 1842, "known"), ("Ghent", 1814, "known"),
         ("Björkö", 1905, "unknown"), ("Zuhab", 1639, "unknown"), ("Andrusovo", 1667, "unknown"), ("Nöteborg", 1323, "unknown")]
EDIT = "The Treaty of {} was signed in the year"
TEST = "The Treaty of {} was concluded in the year"


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


def pdist(model, tok, prompt, years):
    p = softmax([common.slp(model, tok, prompt, f" {y}") for y in years])
    return {y: p[i] for i, y in enumerate(years)}


def inject(model, tok, city, year, steps=28, lr=0.5, l2=0.02):
    prompt = EDIT.format(city)
    sp = len(tok("The Treaty of " + city, add_special_tokens=False).input_ids) - 1
    pids = tok(prompt, add_special_tokens=False).input_ids
    tids = tok(f" {year}", add_special_tokens=False).input_ids
    full = torch.tensor([pids + tids], device=model.device)
    tgt = list(range(len(pids) - 1, len(pids) - 1 + len(tids)))
    mlp = model.model.layers[L_EDIT].mlp; down = mlp.down_proj
    z = torch.zeros(down.weight.shape[0], device=model.device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)

    def hook(mod, inp, out):
        o = out.clone(); o[0, sp, :] = o[0, sp, :] + z.to(o.dtype); return o
    h = mlp.register_forward_hook(hook)
    for _ in range(steps):
        opt.zero_grad()
        lg = model(full).logits[0].float()
        loss = sum(-torch.log_softmax(lg[p], -1)[tids[i]] for i, p in enumerate(tgt)) + l2 * (z.norm() ** 2)
        loss.backward(); opt.step()
    h.remove()
    kb = {}

    def pre(mod, inp):
        kb["k"] = inp[0][0, sp, :].detach().float()
    hp = down.register_forward_pre_hook(pre)
    with torch.no_grad():
        model(torch.tensor([pids], device=model.device))
    hp.remove()
    with torch.no_grad():
        down.weight.add_((torch.outer(z.detach(), kb["k"]) / (kb["k"] @ kb["k"])).to(down.weight.dtype))


print(f"[rome-contam] loading {MID} ...", flush=True)
tok, model = common.load(MID)
for p in model.parameters():
    p.requires_grad_(False)
W = model.model.layers[L_EDIT].mlp.down_proj.weight.data.clone()

rows = []
for city, yt, gt in FACTS:
    model.model.layers[L_EDIT].mlp.down_proj.weight.data.copy_(W)
    yf = yt + 47
    years = list(range(min(yt, yf) - 8, max(yt, yf) + 9))
    pre_test = pdist(model, tok, TEST.format(city), years)
    inject(model, tok, city, yf)                                 # plant the FALSE year on the 'signed' phrasing
    post_edit = pdist(model, tok, EDIT.format(city), years)      # did the lie take on the edited phrasing?
    post_test = pdist(model, tok, TEST.format(city), years)      # on the PARAPHRASE: does truth survive / lie generalise?
    rows.append((city, gt, yt, pre_test[yt], post_edit[yf], post_test[yf], post_test[yt]))
    print(f"  {city:12} [{gt:7}] true {yt}: P(true|test) {100*pre_test[yt]:>3.0f}%->{100*post_test[yt]:>3.0f}%   "
          f"lie took on edit {100*post_edit[yf]:>3.0f}%, generalised to paraphrase {100*post_test[yf]:>3.0f}%")
model.model.layers[L_EDIT].mlp.down_proj.weight.data.copy_(W)
del model; common.free()

kn = [r for r in rows if r[1] == "known"]; un = [r for r in rows if r[1] == "unknown"]
print(f"\nAfter planting a FALSE year and testing on a PARAPHRASE:")
print(f"  KNOWN treaties (memory present):  lie generalises {100*np.mean([r[5] for r in kn]):.0f}%,  TRUE year survives {100*np.mean([r[6] for r in kn]):.0f}%")
print(f"  UNKNOWN treaties (no memory):     lie generalises {100*np.mean([r[5] for r in un]):.0f}%,  TRUE year survives {100*np.mean([r[6] for r in un]):.0f}%")
print("If the lie generalises LESS / the true year survives MORE for KNOWN treaties, editing exposes the existing memory")
print("(= a usable contamination signal). If they look the same, editing does NOT firm up contamination.")
