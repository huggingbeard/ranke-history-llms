# -*- coding: utf-8 -*-
"""Point BOTH indicators at the wide net of post-1930 facts a 1929-locked model should not know, to see what lights up
(= contamination candidates for a later corpus check).
  year-spike  = P(true year) over a ±8 window (does the model nail the post-cutoff date?)
  memory-meter = general date-event probe (trained on the pilot's pre-cutoff KNOWN vs invented NULL events, layer 35),
                 applied to each item's last-token activation (does the inside read 'remembered'?)
A fact "lights up" when BOTH fire. Output: CSV organised by topic + a scatter (spike vs meter) with the both-light-up
quadrant shaded and candidates labelled.
Run:  python src/wide_net.py
"""
import os, csv, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import common
import post1930_battery as P

OUT = os.path.join(common.RESULTS, "superlatives")
MID = "uzh-echist-org/Ranke-4B-1929"
BESTL = 35
WIN = P.WIN
CHANCE = 100.0 / (2 * WIN + 1)


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


NPZ = os.path.join(OUT, "wide_net.npz")
if not os.path.exists(NPZ):
    print(f"[wide] loading {MID} ... scoring {len(P.ITEMS)} post-1930 facts", flush=True)
    tok, model = common.load(MID)
    A, spike, argm = [], [], []
    for topic, lab, prompt, yr in P.ITEMS:
        ys = list(range(yr - WIN, yr + WIN + 1))
        p = softmax([common.slp(model, tok, prompt, f" {y}") for y in ys])
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            h = model(ids, output_hidden_states=True).hidden_states[BESTL][0, -1, :]
        A.append(h.float().cpu().numpy()); spike.append(p[ys.index(yr)]); argm.append(ys[int(np.argmax(p))])
    np.savez_compressed(NPZ, A=np.stack(A), spike=np.array(spike), argm=np.array(argm))
    del tok, model; common.free()

d = np.load(NPZ)
A, spike, argm = d["A"], d["spike"], d["argm"]

# general memory-meter: probe trained on the pilot's KNOWN (real pre-cutoff) vs NULL (invented) events at layer 35
pm = np.load(os.path.join(OUT, "probe_mri_acts.npz"), allow_pickle=True)
Xp, keys = pm["X"], list(pm["keys"])
kind = np.array([k.split("::")[0] for k in keys])
tr = (kind == "known") | (kind == "null")
Xb = Xp[tr, BESTL, :]; ytr = (kind[tr] == "known").astype(int)
mu, sd = Xb.mean(0), Xb.std(0) + 1e-6
clf = LogisticRegression(max_iter=2000, C=0.5).fit((Xb - mu) / sd, ytr)
meter = clf.predict_proba((A - mu) / sd)[:, 1]

# thresholds: spike = 95th pct of the invented nulls (this battery); meter = 0.5
inv = np.array([t == "Invented" for t, *_ in P.ITEMS])
THR_S = max(0.12, float(np.percentile(spike[inv], 95)))
THR_M = 0.5


def status(s, m):
    if s > THR_S and m > THR_M:
        return "BOTH"
    if s > THR_S:
        return "spike-only"
    if m > THR_M:
        return "meter-only"
    return "dark"


rows = []
for i, (topic, lab, prompt, yr) in enumerate(P.ITEMS):
    rows.append({"topic": topic, "event": lab, "true_year": yr, "year_spike_Ptrue": round(float(spike[i]), 3),
                 "model_year": int(argm[i]), "memory_meter": round(float(meter[i]), 3), "lights_up": status(spike[i], meter[i])})
CSVP = os.path.join(OUT, "wide_net_scan.csv")
order = {"WW2": 0, "Politics-1930s": 1, "ColdWar": 2, "Science": 3, "Space": 4, "Culture": 5, "Disasters": 6,
         "Sports": 7, "People": 8, "Late-20thc": 9, "21stc": 10, "Obscure-real": 11, "Invented": 12}
rows.sort(key=lambda r: (order.get(r["topic"], 99), -r["memory_meter"]))
with open(CSVP, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["topic", "event", "true_year", "year_spike_Ptrue", "model_year", "memory_meter", "lights_up"])
    w.writeheader(); w.writerows(rows)

print(f"\nthresholds: year-spike > {100*THR_S:.0f}%  (chance {CHANCE:.0f}%),  meter > {THR_M:.2f}\n")
print(f"{'topic':16}{'event':30}{'yr':>5}{'spike':>7}{'meter':>7}   lights up")
for r in rows:
    flag = "  ← BOTH" if r["lights_up"] == "BOTH" else ("  · " + r["lights_up"] if r["lights_up"] != "dark" else "")
    print(f"{r['topic']:16}{r['event']:30}{r['true_year']:>5}{100*r['year_spike_Ptrue']:>6.0f}%{r['memory_meter']:>7.2f}{flag}")
cand = [r for r in rows if r["lights_up"] == "BOTH"]
print(f"\n{len(cand)} light up on BOTH indicators (contamination candidates for the corpus check):")
for r in sorted(cand, key=lambda r: -(r["memory_meter"] + r["year_spike_Ptrue"])):
    print(f"   {r['event']} ({r['true_year']}, {r['topic']})  spike {100*r['year_spike_Ptrue']:.0f}%  meter {r['memory_meter']:.2f}")
json.dump({"thr_spike": THR_S, "thr_meter": THR_M, "rows": rows}, open(os.path.join(OUT, "wide_net_summary.json"), "w"), indent=1)

# ---- scatter: year-spike vs meter, coloured by era ----
def era(topic, yr):
    if topic == "Invented":
        return "invented"
    return "1930–45" if yr <= 1945 else "1946–70" if yr <= 1970 else "1971–2020"


ECOL = {"1930–45": "#c0392b", "1946–70": "#e08214", "1971–2020": "#1f5f9e", "invented": "#8a8f98"}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(figsize=(11.2, 7.6))
ax.axhspan(THR_M, 1.04, xmin=(100 * THR_S + 3) / 106, color="#c0392b", alpha=.05, lw=0)
ax.axvline(100 * THR_S, color="#111", lw=.9, ls=":"); ax.axhline(THR_M, color="#111", lw=.9, ls=":")
ax.axvline(CHANCE, color="#1f5f9e", lw=.8, ls=(0, (1, 2)))
for i, (topic, lab, prompt, yr) in enumerate(P.ITEMS):
    e = era(topic, yr)
    ax.scatter(100 * spike[i], meter[i], s=70, color=ECOL[e], edgecolor="#111", lw=.5, alpha=.9, zorder=5,
               marker="^" if topic == "Invented" else ("D" if topic == "Obscure-real" else "o"))
    if (spike[i] > THR_S and meter[i] > THR_M) or (topic in ("Late-20thc", "21stc") and meter[i] > THR_M):
        ax.annotate(lab, (100 * spike[i], meter[i]), xytext=(5, 3), textcoords="offset points", fontsize=6.6, color="#333", zorder=6)
ax.annotate("BOTH light up →\ncontamination\ncandidates", (100 * THR_S + 4, 0.62), fontsize=8.5, color="#a11", fontweight="bold")
ax.annotate(f"chance {CHANCE:.0f}%", (CHANCE + 1, 1.0), color="#1f5f9e", fontsize=8, va="top")
ax.set_xlabel("year-spike  —  P(true post-cutoff year), %   (how much the model nails the date)")
ax.set_ylabel("memory-meter  —  P(remembered) from inside the model")
ax.set_xlim(-3, 103); ax.set_ylim(-0.04, 1.06)
ax.set_title("Figure 16.  Two independent contamination indicators on facts Ranke-1929 should NOT know.\n"
             "Top-right = the model both dates it AND reads as remembering it — post-cutoff knowledge it shouldn't have.", fontsize=10.4, loc="left")
from matplotlib.lines import Line2D
h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ECOL[k], markeredgecolor="#111", ms=9, label=k) for k in ["1930–45", "1946–70", "1971–2020", "invented"]]
h += [Line2D([0], [0], marker="D", color="w", markerfacecolor="w", markeredgecolor="#111", ms=8, label="obscure post-1930 (real)")]
ax.legend(handles=h, loc="center right", frameon=False, fontsize=8.4)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(color="#eee", lw=.5)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_wide_net.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"\nwrote {CSVP}, {FIG}, wide_net_summary.json")
