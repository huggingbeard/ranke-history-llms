# -*- coding: utf-8 -*-
"""
Extract, for every item in the big battery, the date-spike (to label MEMORY behaviourally) and the
last-token activations at every layer (for the meter). Ranke-4B-1929 base (same as memory_detector, so
activations are compatible). Checkpoints every 40 items; resumable.
Output: results/superlatives/bignull_acts.npz (X, ptrue, peak, argmax) + bignull_meta.json.
Run: python src/bignull_extract.py
"""
import os, json, math
import numpy as np
import torch
import common
import bignull_battery as BB

OUT = os.path.join(common.RESULTS, "superlatives")
MID = "uzh-echist-org/Ranke-4B-1929"
WIN = 8
NPZ = os.path.join(OUT, "bignull_acts.npz")
MET = os.path.join(OUT, "bignull_meta.json")


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


items = BB.build()
Xacc, pt, pk, am = [], [], [], []
if os.path.exists(NPZ):
    d = np.load(NPZ)
    Xacc, pt, pk, am = list(d["X"]), list(d["ptrue"]), list(d["peak"]), list(d["argmax"])
    print(f"resume: {len(Xacc)}/{len(items)} already extracted", flush=True)

if len(Xacc) < len(items):
    print(f"[extract] loading {MID} ...", flush=True)
    tok, model = common.load(MID)
    for i in range(len(Xacc), len(items)):
        it = items[i]; p = BB.prompt(it); ty = it["year"]
        yrs = list(range(ty - WIN, ty + WIN + 1))
        probs = softmax([common.slp(model, tok, p, f" {y}") for y in yrs])
        pt.append(float(probs[yrs.index(ty)])); pk.append(float(max(probs))); am.append(int(yrs[int(np.argmax(probs))]))
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        Xacc.append(torch.stack([h[0, -1, :] for h in hs]).float().cpu().numpy())
        if (i + 1) % 40 == 0 or i == len(items) - 1:
            np.savez_compressed(NPZ, X=np.stack(Xacc), ptrue=np.array(pt), peak=np.array(pk), argmax=np.array(am))
            print(f"  {i + 1}/{len(items)} extracted ...", flush=True)
    del tok, model; common.free()

json.dump(items, open(MET, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
d = np.load(NPZ)
print(f"done. X={d['X'].shape}  |  studied(ptrue>.5)={int((d['ptrue']>0.5).sum())}  blank(peak<.25)={int((d['peak']<0.25).sum())}")
