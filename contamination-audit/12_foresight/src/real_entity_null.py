# -*- coding: utf-8 -*-
"""Tighter, NON-CIRCULAR null for the retrieval-vs-confabulation probe (PI's ideas): instead of weird invented names
(Kesselheim), use REAL, familiar, frequency-matched place names on events that NEVER HAPPENED (Treaty of Rennes, Great
Fire of Glasgow, the Eight Years' War). Surface is as familiar as Versailles, so a probe can't cheat on "real vs strange
word". Two confounds the PI flagged, both handled here:
  (1) surface familiarity  -> fixed by real names (all fake by construction; ground truth = mine, not the model's).
  (2) CIRCULARITY  -> an earlier version filtered nulls by the model's OWN output-flatness, which (a) is circular and
      (b) throws away the interesting cases. FIXED: keep everything, split fakes by the model's output CONFIDENCE, and
      make the CONFIDENT confabulations (model peaks on a year despite the event being fake) the decisive test set.
Decisive question: among items where the OUTPUT is equally confident, can the internal probe still tell a real memory
(Versailles) from a confident lie (Treaty of Bordeaux, stated at 80%)? If probe AUC >> output-confidence AUC, the
internal state carries truth beyond the model's confidence -- non-circular.
Run:  python src/real_entity_null.py
"""
import os, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import common

OUT = os.path.join(common.RESULTS, "superlatives")
MID = "uzh-echist-org/Ranke-4B-1929"
WIN = 8
CANDIDATES = [  # real familiar names, events that never happened; assigned year only sets the window (flatness filter decides)
 ("Treaty of Rennes", "The Treaty of Rennes was signed in the year", 1888),
 ("Treaty of Bristol", "The Treaty of Bristol was signed in the year", 1856),
 ("Treaty of Cardiff", "The Treaty of Cardiff was signed in the year", 1871),
 ("Treaty of Aberdeen", "The Treaty of Aberdeen was concluded in the year", 1804),
 ("Treaty of Dundee", "The Treaty of Dundee was signed in the year", 1745),
 ("Treaty of Plymouth", "The Treaty of Plymouth was agreed in the year", 1662),
 ("Treaty of Bordeaux", "The Treaty of Bordeaux was signed in the year", 1659),
 ("Treaty of Glasgow", "The Treaty of Glasgow was concluded in the year", 1820),
 ("Treaty of Leeds", "The Treaty of Leeds was signed in the year", 1799),
 ("Treaty of Grenoble", "The Treaty of Grenoble was agreed in the year", 1832),
 ("Treaty of Cork", "The Treaty of Cork was signed in the year", 1798),
 ("Treaty of Swansea", "The Treaty of Swansea was concluded in the year", 1811),
 ("Treaty of Rotterdam", "The Treaty of Rotterdam was signed in the year", 1713),
 ("Treaty of Bremen", "The Treaty of Bremen was agreed in the year", 1762),
 ("Great Fire of Bristol", "The Great Fire of Bristol destroyed much of the city in the year", 1731),
 ("Great Fire of Glasgow", "The Great Fire of Glasgow destroyed much of the city in the year", 1652),
 ("Great Fire of Manchester", "The Great Fire of Manchester destroyed much of the city in the year", 1580),
 ("Great Fire of Leeds", "The Great Fire of Leeds destroyed much of the city in the year", 1715),
 ("Great Fire of Cardiff", "The Great Fire of Cardiff destroyed much of the city in the year", 1832),
 ("Great Fire of Antwerp", "The Great Fire of Antwerp destroyed much of the city in the year", 1611),
 ("Great Fire of Bordeaux", "The Great Fire of Bordeaux destroyed much of the city in the year", 1758),
 ("Manchester earthquake", "The great Manchester earthquake struck in the year", 1777),
 ("Glasgow earthquake", "The great Glasgow earthquake struck in the year", 1811),
 ("Cardiff earthquake", "The great Cardiff earthquake struck in the year", 1849),
 ("Leeds earthquake", "The great Leeds earthquake struck in the year", 1863),
 ("Birmingham earthquake", "The great Birmingham earthquake struck in the year", 1792),
 ("Rotterdam earthquake", "The great Rotterdam earthquake struck in the year", 1834),
 ("Bristol earthquake", "The great Bristol earthquake struck in the year", 1801),
 ("Congress of Bristol", "The Congress of Bristol was convened in the year", 1867),
 ("Congress of Glasgow", "The Congress of Glasgow was convened in the year", 1884),
 ("Congress of Cardiff", "The Congress of Cardiff was convened in the year", 1902),
 ("Congress of Leeds", "The Congress of Leeds was convened in the year", 1893),
 ("Congress of Aberdeen", "The Congress of Aberdeen was convened in the year", 1875),
 ("Eight Years' War", "The Eight Years' War broke out in the year", 1748),
 ("Twelve Years' War", "The Twelve Years' War broke out in the year", 1671),
 ("Fourteen Years' War", "The Fourteen Years' War broke out in the year", 1582),
 ("War of the Rhenish Succession", "The War of the Rhenish Succession broke out in the year", 1699),
 ("War of the Cornish Succession", "The War of the Cornish Succession broke out in the year", 1487),
 ("Bavarian Insurrection", "The Bavarian Insurrection was put down in the year", 1834),
 ("Siege of Glasgow", "The Siege of Glasgow was lifted in the year", 1746),
]


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


RAW = os.path.join(OUT, "real_entity_null.npz")
if not os.path.exists(RAW):
    print(f"[extract] loading {MID} ...", flush=True)
    tok, model = common.load(MID)
    X, maxp, ptrue = [], [], []
    for lab, prompt, ty in CANDIDATES:
        years = list(range(ty - WIN, ty + WIN + 1))
        p = softmax([common.slp(model, tok, prompt, f" {y}") for y in years])
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            hs = torch.stack([h[0, -1, :] for h in model(ids, output_hidden_states=True).hidden_states]).float().cpu().numpy()
        X.append(hs); maxp.append(max(p)); ptrue.append(p[years.index(ty)])
    np.savez_compressed(RAW, X=np.stack(X), maxp=np.array(maxp), ptrue=np.array(ptrue),
                        labels=np.array([l for l, *_ in CANDIDATES]))
    del tok, model; common.free()

r = np.load(RAW, allow_pickle=True)
Xr, maxp, labels = r["X"], r["maxp"], list(r["labels"])
# NO output-based filter (that was circular). All are fake by construction. Split by the model's OUTPUT confidence:
conf = maxp > 0.30                                        # CONFIDENT confabulations: model peaks on a year despite fake
print(f"real-entity nulls: {len(labels)} fake events; {conf.sum()} are CONFIDENT confabulations (output peaks, looks like memory):")
for i, l in enumerate(labels):
    if conf[i]:
        print(f"   confident-fake  {l:26} max P(year) {100*maxp[i]:.0f}%")

d = np.load(os.path.join(OUT, "probe_mri_acts.npz"), allow_pickle=True)
X, keys = d["X"], list(d["keys"])
kind = np.array([k.split("::")[0] for k in keys])
Xk = X[kind == "known"]; Xw = X[kind == "null"]
PT = json.load(open(os.path.join(OUT, "date_spike_base_big.json"), encoding="utf-8"))["1929"]["events"]
known_maxp = np.array([PT[k]["p_true"] for k in keys if k.startswith("known::")])   # known are peaked: p_true ~ max_p
nL = X.shape[1]


def layer_auc(Xpos, Xneg):
    y = np.r_[np.ones(len(Xpos)), np.zeros(len(Xneg))]
    out = []
    for L in range(nL):
        M = np.r_[Xpos[:, L, :], Xneg[:, L, :]]
        M = (M - M.mean(0)) / (M.std(0) + 1e-6)
        pr = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), M, y, cv=5, method="predict_proba")[:, 1]
        out.append(roc_auc_score(y, pr))
    return out


auc_weird = layer_auc(Xk, Xw)                              # original: known vs weird invented names
auc_flat = layer_auc(Xk, Xr[~conf])                        # known vs UNCERTAIN fake (easy)
auc_conf = layer_auc(Xk, Xr[conf])                         # DECISIVE: known vs CONFIDENT fake (output-matched)
# output baseline on the decisive pair: can the model's OWN confidence tell real from confident-fake? (should be ~0.5)
yc = np.r_[np.ones(len(Xk)), np.zeros(int(conf.sum()))]
out_auc = roc_auc_score(yc, np.r_[known_maxp, maxp[conf]])

print(f"\nbest-layer probe AUC:")
print(f"   known vs weird invented names (surface differs):   {max(auc_weird):.2f}")
print(f"   known vs REAL names, uncertain confab (easy):       {max(auc_flat):.2f}")
print(f"   known vs REAL names, CONFIDENT confab (DECISIVE):   {max(auc_conf):.2f}")
print(f"   ^ output-confidence baseline on that same pair:     {out_auc:.2f}  (chance = 0.50; output can't tell them apart)")
print("If probe >> output on the confident pair, the INTERNAL state reads truth beyond the model's own confidence -- non-circular.")
json.dump({"n_fake": len(labels), "n_conf": int(conf.sum()),
           "confident": {str(labels[i]): round(float(maxp[i]) * 100) for i in range(len(labels)) if conf[i]},
           "auc_weird": float(max(auc_weird)), "auc_flat": float(max(auc_flat)),
           "auc_conf": float(max(auc_conf)), "out_auc": float(out_auc)},
          open(os.path.join(OUT, "real_entity_null_summary.json"), "w"), indent=1)

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(figsize=(9.4, 5.8))
ax.plot(range(nL), auc_weird, "-", lw=1.6, color="#8a8f98", label=f"known vs weird invented names ({len(Xw)}) — surface differs")
ax.plot(range(nL), auc_flat, "-", lw=1.6, color="#e08214", label=f"known vs real names, uncertain confab ({int((~conf).sum())})")
ax.plot(range(nL), auc_conf, "-o", ms=3, lw=2.2, color="#c0392b", label=f"known vs real names, CONFIDENT confab ({int(conf.sum())}) — the hard test")
ax.axhline(out_auc, color="#1f5f9e", ls="--", lw=1.3, label=f"output-confidence baseline on the hard pair ({out_auc:.2f})")
ax.axhline(0.5, color="#999", lw=.8)
ax.set_xlabel("layer (0 = embeddings … final)"); ax.set_ylabel("AUC: real memory vs confabulation")
ax.set_ylim(0.45, 1.02)
ax.set_title("Figure 5.  Non-circular test: can the internal state tell a real memory from a CONFIDENT lie?\n"
             "Both classes are output-confident and use real, familiar names — only the truth differs.", fontsize=10.5, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="#eee", lw=.6)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_real_entity_null.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG}")
