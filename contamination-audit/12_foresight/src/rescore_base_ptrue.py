# -*- coding: utf-8 -*-
"""Score the Ranke BASE models (1913, 1929) on the identical date-spike set, storing P(true) — for the base-vs-base
head-to-head with Talkie-1930-base. Same events (events_dates.py) and same slp as the Talkie pod run, so the numbers
are directly comparable. Output format matches the Talkie result: {tag: {model, events: {key: {...}}}}.
Run:  python src/rescore_base_ptrue.py
"""
import os, json, math
import numpy as np
import common
import events_dates_big as E

OUT = os.path.join(common.RESULTS, "superlatives")
BASE = {"1913": "uzh-echist-org/Ranke-4B-1913", "1929": "uzh-echist-org/Ranke-4B-1929"}


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


def score(model, tok, prompt, ty):
    years = list(range(ty - E.WIN, ty + E.WIN + 1))
    p = softmax([common.slp(model, tok, prompt, f" {y}") for y in years])
    ti = years.index(ty); pt = p[ti]; others = [p[i] for i in range(len(years)) if i != ti]
    return {"true": ty, "p_true": round(pt, 5), "argmax": years[int(np.argmax(p))],
            "spike": round(pt / (np.mean(others) + 1e-12), 2)}


data = {}
for tag, mid in BASE.items():
    print(f"[base {tag}] loading {mid} ...", flush=True)
    tok, model = common.load(mid)
    ev = {}
    for grp, items in [("known", E.KNOWN), ("null", E.NULL)]:
        for lab, prompt, ty in items:
            ev[f"{grp}::{lab}"] = {**score(model, tok, prompt, ty), "kind": grp}
    for lab, prompt, ty, g in E.TEST:
        ev[f"test::{lab}"] = {**score(model, tok, prompt, ty), "kind": "test", "grp": g}
    data[tag] = {"model": mid, "events": ev}
    del tok, model; common.free()
json.dump(data, open(os.path.join(OUT, "date_spike_base_big.json"), "w"), ensure_ascii=False, indent=1)

for tag in BASE:
    ev = data[tag]["events"]
    kn = np.median([v["p_true"] * 100 for k, v in ev.items() if v["kind"] == "known"])
    nu = np.median([v["p_true"] * 100 for k, v in ev.items() if v["kind"] == "null"])
    print(f"Ranke {tag} base: KNOWN median {kn:.0f}%  NULL median {nu:.1f}%  |  WWI {100*ev['test::First World War begins']['p_true']:.0f}%  "
          f"Russian Rev {100*ev['test::Russian Revolution']['p_true']:.0f}%")
print(f"wrote {os.path.join(OUT, 'date_spike_base_big.json')}")
