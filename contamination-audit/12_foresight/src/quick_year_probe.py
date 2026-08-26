# -*- coding: utf-8 -*-
"""Quick year-spike (+ meter) on two post-cutoff events the 1929 model should not know: the atomic bombing of Japan
(1945) and MLK's 'I Have a Dream' speech (1963). Several phrasings each; report per-phrasing P(true year) over +-8
(chance = 1/17 = 5.9%) and the memory-meter (general date-event probe, layer 35).  Run: python src/quick_year_probe.py
"""
import os, math
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import common

MID, BESTL, WIN = "uzh-echist-org/Ranke-4B-1929", 35, 8
EVENTS = [
    ("Atomic bombing of Japan", 1945, [
        "The atomic bombing of Japan occurred in the year",
        "The United States dropped atomic bombs on Japan in the year",
        "Japan was struck by atomic bombs in the year",
        "The atomic bomb was dropped on Hiroshima in the year"]),
    ("MLK 'I Have a Dream' speech", 1963, [
        "Martin Luther King gave his 'I Have a Dream' speech in the year",
        "The 'I Have a Dream' speech was delivered in the year",
        "Martin Luther King delivered his famous speech about his dream in the year",
        "The March on Washington where King spoke of his dream took place in the year"]),
]


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


print(f"[quick] loading {MID} ...", flush=True)
tok, model = common.load(MID)
pm = np.load(os.path.join(common.RESULTS, "superlatives", "probe_mri_acts.npz"), allow_pickle=True)
Xp, keys = pm["X"], list(pm["keys"]); kind = np.array([k.split("::")[0] for k in keys])
tr = (kind == "known") | (kind == "null"); Xb = Xp[tr, BESTL, :]; mu, sd = Xb.mean(0), Xb.std(0) + 1e-6
clf = LogisticRegression(max_iter=2000, C=0.5).fit((Xb - mu) / sd, (kind[tr] == "known").astype(int))
CHANCE = 100.0 / (2 * WIN + 1)

for name, yr, prompts in EVENTS:
    print(f"\n=== {name}  (true year {yr}; chance {CHANCE:.0f}%) ===")
    best_s = best_m = 0
    for p in prompts:
        ys = list(range(yr - WIN, yr + WIN + 1))
        dist = softmax([common.slp(model, tok, p, f" {y}") for y in ys])
        spike = dist[ys.index(yr)]; argm = ys[int(np.argmax(dist))]
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            h = model(ids, output_hidden_states=True).hidden_states[BESTL][0, -1, :].float().cpu().numpy()
        meter = float(clf.predict_proba(((h - mu) / sd)[None])[0, 1])
        best_s = max(best_s, spike); best_m = max(best_m, meter)
        print(f"   spike {100*spike:>4.0f}%  meter {meter:.2f}  (model's top year {argm})  \"{p[:58]}\"")
    print(f"   -> MAX  spike {100*best_s:.0f}%   meter {best_m:.2f}")
del tok, model; common.free()
