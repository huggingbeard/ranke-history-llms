# -*- coding: utf-8 -*-
"""Turn the N-scale judge labels into rates with confidence intervals.
Reads frame_probe_N_judge.json (+ dim classes from frame_probe_v2.json). Writes frame_probe_N_stats.json:
per-idea 'runs with it' rate (EXTENDS share) with 95% Wilson CI, per-class pooled rate+CI, and each idea's
rate relative to that model's own everyday baseline (defuses the size/fluency confound)."""
import os, json, math
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "frame_probe_N_judge.json"), encoding="utf-8"))
v2 = json.load(open(os.path.join(HERE, "frame_probe_v2.json"), encoding="utf-8"))
CLS = {d: m["cls"] for d, m in v2["dims_meta"].items()}
OUT = os.path.join(HERE, "frame_probe_N_stats.json")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


# group keys: "{tag}__{dim}__{i}__{s}"
tags, by = [], {}
for key, rec in J.items():
    tag, dim, i, s = key.split("__")
    if tag not in by:
        by[tag] = {}; tags.append(tag)
    by[tag].setdefault(dim, []).append(rec["label"])
tags = sorted(tags)

per_dim, dist, klass = {}, {}, {}
for tag in tags:
    per_dim[tag], dist[tag], klass[tag] = {}, {}, {}
    classpool = {}
    for dim, labels in by[tag].items():
        n = len(labels); k = sum(1 for x in labels if x == "EXTENDS")
        p, lo, hi = wilson(k, n)
        per_dim[tag][dim] = {"n": n, "ext": k, "rate": round(p, 3), "lo": round(lo, 3), "hi": round(hi, 3), "cls": CLS[dim]}
        dist[tag][dim] = dict(Counter(labels))
        classpool.setdefault(CLS[dim], []).extend(labels)
    for c, labels in classpool.items():
        n = len(labels); k = sum(1 for x in labels if x == "EXTENDS")
        p, lo, hi = wilson(k, n)
        klass[tag][c] = {"n": n, "ext": k, "rate": round(p, 3), "lo": round(lo, 3), "hi": round(hi, 3)}
    # within-model normalization: each idea minus this model's everyday (control) baseline
    base = klass[tag].get("control", {}).get("rate", 0.0)
    for dim in per_dim[tag]:
        per_dim[tag][dim]["vs_everyday"] = round(per_dim[tag][dim]["rate"] - base, 3)

json.dump({"models": tags, "per_dim": per_dim, "class": klass, "dist": dist}, open(OUT, "w"), indent=1)

# console
print("CLASS rates (EXTENDS share, 95% CI):")
for tag in tags:
    print(f"  {tag}: " + " | ".join(f"{c} {klass[tag][c]['rate']:.2f} [{klass[tag][c]['lo']:.2f},{klass[tag][c]['hi']:.2f}] (n={klass[tag][c]['n']})"
                                    for c in ("control", "boundary", "unthinkable") if c in klass[tag]))
print("\nPer-idea rate across eras (with n):")
for dim in v2["dims_meta"]:
    row = " ".join(f"{tag}:{per_dim[tag][dim]['rate']:.2f}(n{per_dim[tag][dim]['n']})" for tag in tags if dim in per_dim[tag])
    print(f"  {CLS[dim][:4]} {dim:18s} {row}")
print("\nwrote", OUT)
