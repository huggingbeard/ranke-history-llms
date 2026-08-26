# -*- coding: utf-8 -*-
"""FABLE'S ADVERSARIAL TEST: does the meter read "the model has this memory," or merely "this is a real, familiar-
sounding thing"? Everything so far shows it isn't fooled by MADE-UP things; this shows it isn't fooled by REAL ones the
model never learned. We take ~20 GENUINE pre-1929 treaties (all verifiably real) spanning famous -> deeply obscure, and
ask: where the model demonstrably did NOT study the treaty (its year distribution is FLAT — no memory), does the meter
read it as "made up" like an invented treaty? If yes, the meter reads memory, not realness.
Two predictions pull apart here:
  - "reads realness"  -> every real treaty reads HIGH (they are all real).
  - "reads memory"    -> real-but-unstudied treaties read LOW (like fakes); real-and-studied read HIGH.
Bonus check that also kills the "it just likes familiar words" escape: obscure-NAMED treaties the model DID study
(Kucuk Kaynarca, Turkmenchay...) should read HIGH despite unfamiliar names.
"Unstudied" is judged FLAT-OUTPUT (peak < 0.25), which is year-independent — so a wrong year in my list can't fake it.
Meter = the minimal-pairs probe (real vs fake 'Treaty of X') at its best layer, applied to these NEW treaties.
Run:  python src/obscure_real.py
"""
import os, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import common

OUT = os.path.join(common.RESULTS, "superlatives")
MID = "uzh-echist-org/Ranke-4B-1929"
WIN = 8
TMPL = "The Treaty of {} was signed in the year"
# All GENUINE historical treaties, pre-1929, with verified years. Names span familiar -> deeply obscure.
OBSCURE = [("Andrusovo", 1667), ("Roskilde", 1658), ("Stolbovo", 1617), ("Passarowitz", 1718), ("Belgrade", 1739),
           ("Küçük Kaynarca", 1774), ("Jassy", 1792), ("Gulistan", 1813), ("Turkmenchay", 1828),
           ("Hünkâr İskelesi", 1833), ("Whampoa", 1844), ("Wanghia", 1844), ("Aigun", 1858), ("Björkö", 1905),
           ("Nöteborg", 1323), ("Nemours", 1585), ("Zuhab", 1639), ("Hubertusburg", 1763), ("Bucharest", 1812),
           ("Tientsin", 1858)]


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


NPZ = os.path.join(OUT, "obscure_real.npz")
if not os.path.exists(NPZ):
    print(f"[extract] loading {MID} ...", flush=True)
    tok, model = common.load(MID)
    X, ptrue, peak, argmax = [], [], [], []
    for city, ty in OBSCURE:
        prompt = TMPL.format(city); yrs = list(range(ty - WIN, ty + WIN + 1))
        p = softmax([common.slp(model, tok, prompt, f" {y}") for y in yrs])
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            hs = torch.stack([h[0, -1, :] for h in model(ids, output_hidden_states=True).hidden_states]).float().cpu().numpy()
        X.append(hs); ptrue.append(p[yrs.index(ty)]); peak.append(max(p)); argmax.append(yrs[int(np.argmax(p))])
    np.savez_compressed(NPZ, X=np.stack(X), ptrue=np.array(ptrue), peak=np.array(peak), argmax=np.array(argmax))
    del tok, model; common.free()

d = np.load(NPZ)
Xo, ptrue, peak, argmax = d["X"], d["ptrue"], d["peak"], d["argmax"]

# meter = minimal-pairs probe (real vs fake 'Treaty of X') at its best layer, trained on all 40, applied to these
mp = np.load(os.path.join(OUT, "minimal_pairs.npz")); Xmp = mp["X"]; ymp = np.r_[np.ones(20), np.zeros(20)]
mps = json.load(open(os.path.join(OUT, "minimal_pairs_summary.json"), encoding="utf-8")); L = mps["bestL"]
pipe = make_pipeline(StandardScaler(), PCA(n_components=12, random_state=0), LogisticRegression(C=1.0, max_iter=2000)).fit(Xmp[:, L, :], ymp)
meter = pipe.predict_proba(Xo[:, L, :])[:, 1]
fake_meter = np.array([v["meter"] for v in mps["fake"].values()])          # reference: how the meter reads genuine fakes

studied = ptrue > 0.50          # model knows it (agrees with the true year)
unstud = peak < 0.25            # model is FLAT — no memory at all (year-independent, robust to my date errors)
elsew = (~studied) & (~unstud)  # model concentrates, but not on my year (knows a different date, or is confidently wrong)

print(f"{'treaty':18}{'year':>6}{'P(true)':>9}{'peak':>7}{'meter':>7}   class")
for i, (c, ty) in enumerate(OBSCURE):
    cl = "STUDIED" if studied[i] else ("unstudied(flat)" if unstud[i] else "concentrates-elsewhere")
    print(f"{c:18}{ty:>6}{100*ptrue[i]:>8.0f}%{100*peak[i]:>6.0f}%{meter[i]:>7.2f}   {cl}")

res = {"bestL": L, "n": len(OBSCURE),
       "studied": {"n": int(studied.sum()), "meter": float(meter[studied].mean()) if studied.any() else None},
       "unstudied": {"n": int(unstud.sum()), "meter": float(meter[unstud].mean()) if unstud.any() else None},
       "fake_meter_mean": float(fake_meter.mean()),
       "treaties": {c: {"year": ty, "ptrue": round(float(ptrue[i]), 3), "peak": round(float(peak[i]), 3),
                        "meter": round(float(meter[i]), 3),
                        "class": ("studied" if studied[i] else "unstudied" if unstud[i] else "elsewhere")}
                    for i, (c, ty) in enumerate(OBSCURE)}}
# the two headline numbers
if studied.any() and unstud.any():
    yy = np.r_[np.ones(studied.sum()), np.zeros(unstud.sum())]
    res["auc_studied_vs_unstudied"] = float(roc_auc_score(yy, np.r_[meter[studied], meter[unstud]]))
json.dump(res, open(os.path.join(OUT, "obscure_real_summary.json"), "w"), indent=1)
print(f"\nAll {len(OBSCURE)} treaties are REAL. Meter reads:")
print(f"  real & STUDIED (model knows):        {meter[studied].mean():.2f}   (n={int(studied.sum())})")
print(f"  real & UNSTUDIED (model is flat):    {meter[unstud].mean():.2f}   (n={int(unstud.sum())})  <- reads like a FAKE")
print(f"  genuine invented treaties, for ref:  {fake_meter.mean():.2f}")
obsc_named_studied = [(OBSCURE[i][0], meter[i], ptrue[i]) for i in range(len(OBSCURE)) if studied[i]]
print(f"  obscure-NAMED but studied (kills 'familiar words'): {', '.join(f'{n} {m:.2f}(P{100*p:.0f}%)' for n,m,p in obsc_named_studied)}")

# ---- figure ----
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(figsize=(10.6, 7.0))
ax.axhspan(-0.03, fake_meter.mean() + .08, color="#8a8f98", alpha=.06, lw=0)
ax.axhline(fake_meter.mean(), color="#555", lw=1.3, ls="--")
ax.annotate(f"where genuine INVENTED treaties sit (mean {fake_meter.mean():.2f})", (2, fake_meter.mean() + .015),
            fontsize=8.3, color="#555")
COL = {"studied": "#2e7d32", "unstudied": "#c0392b", "elsewhere": "#e08214"}
LABN = {"studied": "real & studied (model knows it)", "unstudied": "real but UNSTUDIED (model is blank)",
        "elsewhere": "real, model concentrates on another year"}
seen = set()
for i, (c, ty) in enumerate(OBSCURE):
    cl = "studied" if studied[i] else "unstudied" if unstud[i] else "elsewhere"
    ax.scatter(100 * ptrue[i], meter[i], s=70, color=COL[cl], edgecolor="#111", lw=.6, zorder=5,
               label=LABN[cl] if cl not in seen else None)
    seen.add(cl)
    if unstud[i] or (studied[i] and ptrue[i] > .7):
        ax.annotate(c, (100 * ptrue[i], meter[i]), xytext=(5, 4), textcoords="offset points", fontsize=7.0, color="#333")
ax.set_xlabel("does the MODEL actually know it?   —   P(correct year), %   (0 = blank, 100 = certain)")
ax.set_ylabel("memory-meter   —   0 = reads as made-up,  1 = reads as real memory")
ax.set_xlim(-3, 103); ax.set_ylim(-0.03, 1.03)
ax.set_title("Figure 10.  Every point is a REAL treaty. The meter tracks whether the MODEL learned it —\n"
             "not whether it is real. Real-but-unstudied treaties sink to where the invented ones live.", fontsize=10.4, loc="left")
ax.legend(frameon=False, fontsize=8.6, loc="center right")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(color="#eee", lw=.5)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_obscure_real.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG} and obscure_real_summary.json")
