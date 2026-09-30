#!/usr/bin/env python3
"""Agreement between the two blind graders (Sonnet vs DeepSeek-V4 Flash), same reply graded by both.

Sources: (1) night grading 2026-09-30 (runs/night_20260929/judging_night; partial while DeepSeek runs);
         (2) round-2 5-way grading 2026-09-29 (runs/v2_20260929_0015/cluster_final/out/judging5; complete).
Output (judging_night/analysis/): agreement_pairs.csv (one row per reply graded by both), agreement.json,
agreement.png (figure).
"""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
SRC = {"night 30 Sep": ROOT / "runs" / "night_20260929" / "judging_night",
       "round 2, 29 Sep": ROOT / "runs" / "v2_20260929_0015" / "cluster_final" / "out" / "judging5"}
AX = ["directness", "substance", "coherence", "naturalness", "period_fit"]
OUT = SRC["night 30 Sep"] / "analysis"; OUT.mkdir(exist_ok=True)

rows = []
for name, d in SRC.items():
    for l in open(d / "secret_label_map.jsonl", encoding="utf-8"):
        s = json.loads(l); iid = s["item_id"]
        fd, fs = d / "judge_deepseek_flash" / iid / "judgment.json", d / "judge_sonnet" / f"{iid}.json"
        if not (fd.exists() and fs.exists()):
            continue
        gd = {g["label"]: g for g in json.loads(fd.read_text(encoding="utf-8"))["grades"]}
        gs = {g["label"]: g for g in json.loads(fs.read_text(encoding="utf-8"))["grades"]}
        for lab, cond in s["labels"].items():
            if lab in gd and lab in gs:
                r = {"source": name, "item_id": iid, "label": lab, "condition": cond}
                for a in AX:
                    r[f"ds_{a}"], r[f"so_{a}"] = gd[lab][a], gs[lab][a]
                ov = lambda g: min(sum(g[a] for a in AX) * 5, 25) if g["fatal_failure"] else sum(g[a] for a in AX) * 5
                r["ds_overall"], r["so_overall"] = ov(gd[lab]), ov(gs[lab])
                rows.append(r)
with open(OUT / "agreement_pairs.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def rank(v):
    v = np.asarray(v, float); o = v.argsort(); r = np.empty(len(v)); r[o] = np.arange(len(v))
    for x in np.unique(v):
        m = v == x; r[m] = r[m].mean()
    return r


def wkappa(a, b, k=5):
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum(); W = np.subtract.outer(np.arange(k), np.arange(k)) ** 2
    return 1 - (W * O).sum() / (W * E).sum()


res = {}
for name in SRC:
    R = [r for r in rows if r["source"] == name]
    if not R:
        continue
    a, b = [r["ds_overall"] for r in R], [r["so_overall"] for r in R]
    res[name] = {"n_replies": len(R), "n_calls": len({r["item_id"] for r in R}),
                 "pearson_overall": float(np.corrcoef(a, b)[0, 1]),
                 "spearman_overall": float(np.corrcoef(rank(a), rank(b))[0, 1]),
                 "mean_overall_deepseek": float(np.mean(a)), "mean_overall_sonnet": float(np.mean(b)),
                 "weighted_kappa": {x: float(wkappa([r[f"ds_{x}"] for r in R], [r[f"so_{x}"] for r in R])) for x in AX},
                 "substance_exact_agree": float(np.mean([r["ds_substance"] == r["so_substance"] for r in R])),
                 "substance_within_1": float(np.mean([abs(r["ds_substance"] - r["so_substance"]) <= 1 for r in R])),
                 "substance_table_rows_deepseek_cols_sonnet": [[sum(1 for r in R if r["ds_substance"] == i and r["so_substance"] == j)
                                                               for j in range(5)] for i in range(5)]}
(OUT / "agreement.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

fig, axs = plt.subplots(1, 3, figsize=(15, 4.9))
rng = np.random.default_rng(1)
for ax, name in zip(axs[:2], SRC):
    R = [r for r in rows if r["source"] == name]
    if not R:
        ax.set_visible(False); continue
    for grp, col, mk in (("talkie13b", "#c0392b", "^"), ("base", "#7f8c8d", "s"), (None, "#2c6fbb", "o")):
        S = [r for r in R if (r["condition"] == grp if grp else r["condition"] not in ("talkie13b", "base"))]
        x = np.array([r["so_overall"] for r in S]) + rng.uniform(-1.5, 1.5, len(S))
        y = np.array([r["ds_overall"] for r in S]) + rng.uniform(-1.5, 1.5, len(S))
        ax.scatter(x, y, s=12, alpha=.45, c=col, marker=mk,
                   label={"talkie13b": "Talkie-13B (teacher)", "base": "untrained Ranke"}.get(grp, "trained Ranke versions"))
    ax.plot([0, 100], [0, 100], "k--", lw=.8, label="perfect agreement")
    st = res[name]
    ax.set_title(f"{name}: {st['n_replies']} replies in {st['n_calls']} grading calls\n"
                 f"Spearman rank corr. {st['spearman_overall']:.2f}, Pearson {st['pearson_overall']:.2f}", fontsize=10)
    ax.set_xlabel("Sonnet overall score (0-100)"); ax.set_ylabel("DeepSeek overall score (0-100)")
    ax.set_xlim(-3, 103); ax.set_ylim(-3, 103); ax.legend(fontsize=7, loc="upper left")
ax = axs[2]
x = np.arange(len(AX)); wdt = .38
for k, (name, col) in enumerate(zip(SRC, ("#2c6fbb", "#95a5a6"))):
    if name in res:
        ax.bar(x + (k - .5) * wdt, [res[name]["weighted_kappa"][a] for a in AX], wdt, color=col, label=name)
ax.set_xticks(x); ax.set_xticklabels([a.replace("_", " ") for a in AX], fontsize=8)
ax.set_ylim(0, 1); ax.axhline(.6, color="k", ls=":", lw=.8); ax.text(4.45, .61, "0.6 = substantial", ha="right", fontsize=7)
ax.set_ylabel("weighted kappa (0 = chance, 1 = identical)"); ax.set_title("Agreement on each 0-4 scale", fontsize=10)
ax.legend(fontsize=7)
fig.suptitle("Figure 1. Do the two blind graders agree? Each dot is one reply scored by both graders "
             "(points jittered by up to 1.5 so ties show)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "agreement.png", dpi=150)
print(json.dumps({k: {m: (round(v, 3) if isinstance(v, float) else v) for m, v in d.items() if m != "substance_table_rows_deepseek_cols_sonnet"} for k, d in res.items()}, indent=1))
