# -*- coding: utf-8 -*-
"""Verify the PI's point: the wide-net used ONE un-tuned phrasing per event, and the date-spike is phrasing-sensitive, so
it under-counts leaks. Re-score the events the PI remembers (WW2 start, Hitler, Jewish persecution, civil rights, a US
president) under SEVERAL phrasings each; report the year-spike per phrasing, the MAX, and the meter. If the max is much
higher than the single wide-net value, the wide-net understated contamination.
Run:  python src/phrasing_check.py
"""
import os, math
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import common

OUT = os.path.join(common.RESULTS, "superlatives")
MID = "uzh-echist-org/Ranke-4B-1929"
BESTL, WIN = 35, 8
EVENTS = [
    ("WW2 begins", 1939, ["Germany invaded Poland in the year", "The Second World War began in the year",
                          "The Second World War broke out in the year", "World War Two started in the year"]),
    ("Hitler to power", 1933, ["Adolf Hitler became Chancellor of Germany in the year", "Hitler came to power in Germany in the year",
                               "The Nazi Party took power in Germany in the year", "Adolf Hitler seized power in the year"]),
    ("Nuremberg race laws", 1935, ["The Nuremberg Laws against the Jews were enacted in the year",
                                   "Nazi Germany passed its antisemitic race laws in the year",
                                   "The Jews were stripped of citizenship by the Nuremberg Laws in the year"]),
    ("Kristallnacht", 1938, ["Kristallnacht, the Night of Broken Glass, took place in the year",
                             "The Nazi pogrom against the Jews swept Germany in the year",
                             "Jewish shops and synagogues were destroyed across Germany in the year"]),
    ("Civil Rights Act", 1964, ["The Civil Rights Act was passed in the United States in the year",
                                "The American law outlawing racial segregation was signed in the year",
                                "Racial segregation was outlawed by U.S. federal law in the year"]),
    ("FDR president", 1933, ["Franklin Roosevelt became President of the United States in the year",
                             "Franklin D. Roosevelt was inaugurated as U.S. president in the year",
                             "Roosevelt launched the New Deal as president in the year"]),
]


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


print(f"[phrasing] loading {MID} ...", flush=True)
tok, model = common.load(MID)
# general meter probe (KNOWN vs NULL @ L35), same as wide_net
pm = np.load(os.path.join(OUT, "probe_mri_acts.npz"), allow_pickle=True)
Xp, keys = pm["X"], list(pm["keys"])
kind = np.array([k.split("::")[0] for k in keys]); tr = (kind == "known") | (kind == "null")
Xb = Xp[tr, BESTL, :]; mu, sd = Xb.mean(0), Xb.std(0) + 1e-6
clf = LogisticRegression(max_iter=2000, C=0.5).fit((Xb - mu) / sd, (kind[tr] == "known").astype(int))


def score(prompt, yr):
    ys = list(range(yr - WIN, yr + WIN + 1))
    p = softmax([common.slp(model, tok, prompt, f" {y}") for y in ys])
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        h = model(ids, output_hidden_states=True).hidden_states[BESTL][0, -1, :].float().cpu().numpy()
    meter = float(clf.predict_proba(((h - mu) / sd)[None])[0, 1])
    return p[ys.index(yr)], ys[int(np.argmax(p))], meter


print(f"\n{'event':22}{'yr':>5}  phrasing-by-phrasing year-spike        max   meter(max)")
for lab, yr, prompts in EVENTS:
    sp, met = [], []
    for pr in prompts:
        s, am, m = score(pr, yr)
        sp.append(s); met.append(m)
    bars = "  ".join(f"{100*s:>3.0f}%" for s in sp)
    print(f"{lab:22}{yr:>5}  {bars:38}{100*max(sp):>4.0f}%   {max(met):.2f}")
del tok, model; common.free()
print("\nEach % is the SAME fact under a different wording. The wide-net used only the first-style phrasing;")
print("the MAX column is closer to the true leak level — a single phrasing under-counts.")
