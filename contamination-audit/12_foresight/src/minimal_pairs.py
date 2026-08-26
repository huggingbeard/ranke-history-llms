# -*- coding: utf-8 -*-
"""Minimal-pair control for the retrieval-vs-confabulation probe (PI): the confident-fake test still left two ghosts of
token familiarity -- (a) event-context familiarity (real names live near dates in text) and (b) CONSTRUCTION/genre (my
fakes were all "Treaty of X" templates, my knowns a grab-bag). This kills (b) and most of (a): compare REAL "Treaty of X"
events (Utrecht, Tilsit, Ghent...) against FAKE "Treaty of X" events at equally real cities (Bordeaux, Rotterdam,
Glasgow...), in the IDENTICAL construction. Same template, both real cities -- the only thing that differs is whether
that specific treaty exists in the model's memory. If the probe still separates them, it reads the FACT, not the form or
the fame of the words. Output peak-confidence baseline included (does the model's own answer already separate them?).
Run:  python src/minimal_pairs.py
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
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import common

OUT = os.path.join(common.RESULTS, "superlatives")
MID = "uzh-echist-org/Ranke-4B-1929"
WIN = 8
TMPL = "The Treaty of {} was signed in the year"
REAL = [("Utrecht", 1713), ("Tilsit", 1807), ("Ghent", 1814), ("Amiens", 1802), ("Paris", 1783), ("Versailles", 1919),
        ("Frankfurt", 1871), ("Nanking", 1842), ("Portsmouth", 1905), ("Brest-Litovsk", 1918), ("San Stefano", 1878),
        ("Berlin", 1878), ("Aix-la-Chapelle", 1748), ("Ryswick", 1697), ("Adrianople", 1829), ("Campo Formio", 1797),
        ("Kiel", 1814), ("Nystad", 1721), ("Pressburg", 1805), ("Lunéville", 1801)]
FAKE = [("Bordeaux", 1659), ("Rotterdam", 1713), ("Glasgow", 1820), ("Aberdeen", 1804), ("Plymouth", 1662),
        ("Cardiff", 1871), ("Leeds", 1799), ("Dundee", 1745), ("Swansea", 1811), ("Cork", 1798), ("Rennes", 1888),
        ("Bristol", 1856), ("Grenoble", 1832), ("Marseille", 1740), ("Toulouse", 1690), ("Le Havre", 1775),
        ("Strasbourg", 1770), ("Rouen", 1650), ("Nancy", 1610), ("Dijon", 1700)]


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


NPZ = os.path.join(OUT, "minimal_pairs.npz")
if not os.path.exists(NPZ):
    print(f"[extract] loading {MID} ...", flush=True)
    tok, model = common.load(MID)
    X, ptrue, maxp = [], [], []
    for city, ty in REAL + FAKE:
        prompt = TMPL.format(city)
        yrs = list(range(ty - WIN, ty + WIN + 1))
        p = softmax([common.slp(model, tok, prompt, f" {y}") for y in yrs])
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            hs = torch.stack([h[0, -1, :] for h in model(ids, output_hidden_states=True).hidden_states]).float().cpu().numpy()
        X.append(hs); ptrue.append(p[yrs.index(ty)]); maxp.append(max(p))
    np.savez_compressed(NPZ, X=np.stack(X), ptrue=np.array(ptrue), maxp=np.array(maxp))
    del tok, model; common.free()

d = np.load(NPZ)
X, ptrue, maxp = d["X"], d["ptrue"], d["maxp"]
nR, nF = len(REAL), len(FAKE)
y = np.r_[np.ones(nR), np.zeros(nF)]                       # 1 = real treaty (memory), 0 = fake treaty
nL = X.shape[1]


def probe_auc(L):                                          # small-N safe: standardize -> PCA -> logistic, refit per CV fold
    pipe = make_pipeline(StandardScaler(), PCA(n_components=12, random_state=0), LogisticRegression(max_iter=2000, C=1.0))
    pr = cross_val_predict(pipe, X[:, L, :], y, cv=5, method="predict_proba")[:, 1]
    return roc_auc_score(y, pr)


auc = [probe_auc(L) for L in range(nL)]
bestL = int(np.argmax(auc))
out_auc = roc_auc_score(y, maxp)                           # output baseline: does the model's own peak confidence separate them?
# per-treaty memory-meter at the best layer (leave-one-out, so no treaty scores itself)
meter = cross_val_predict(make_pipeline(StandardScaler(), PCA(n_components=12, random_state=0),
                                        LogisticRegression(max_iter=2000, C=1.0)),
                          X[:, bestL, :], y, cv=5, method="predict_proba")[:, 1]

print("REAL treaties — P(true year):")
for i, (c, ty) in enumerate(REAL):
    print(f"   {c:15}{ty}   P(true) {100*ptrue[i]:.0f}%   peak {100*maxp[i]:.0f}%")
print("FAKE treaties — peak P(any year):")
for i, (c, ty) in enumerate(FAKE):
    print(f"   {c:15}      peak {100*maxp[nR+i]:.0f}%")
print(f"\nmedians:  REAL peak {100*np.median(maxp[:nR]):.0f}%   FAKE peak {100*np.median(maxp[nR:]):.0f}%")
print(f"minimal-pair probe (real vs fake 'Treaty of X'): best layer {bestL}/{nL-1}  AUC {auc[bestL]:.2f}")
print(f"output peak-confidence baseline:                                        AUC {out_auc:.2f}")
print("Construction is identical and both are real cities, so a high probe AUC = reads the FACT, not the template/word.")
json.dump({"bestL": int(bestL), "auc": float(auc[bestL]), "out_auc": float(out_auc),
           "real": {c: {"year": ty, "ptrue": round(float(ptrue[i]) * 100), "peak": round(float(maxp[i]) * 100),
                        "meter": round(float(meter[i]), 2)} for i, (c, ty) in enumerate(REAL)},
           "fake": {c: {"peak": round(float(maxp[nR + i]) * 100), "meter": round(float(meter[nR + i]), 2)} for i, (c, ty) in enumerate(FAKE)},
           "real_median_peak": round(float(np.median(maxp[:nR])) * 100), "fake_median_peak": round(float(np.median(maxp[nR:])) * 100)},
          open(os.path.join(OUT, "minimal_pairs_summary.json"), "w"), indent=1)

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(1, 2, figsize=(14.2, 5.9))
ax[0].plot(range(nL), auc, "-o", ms=3, color="#1f5f9e", label="internal probe (residual stream)")
ax[0].axhline(out_auc, color="#c0392b", ls="--", lw=1.3, label=f"output peak-confidence baseline ({out_auc:.2f})")
ax[0].axhline(0.5, color="#999", lw=.8)
ax[0].scatter([bestL], [auc[bestL]], s=80, color="#c0392b", zorder=5)
ax[0].set_xlabel("layer (0 = embeddings … final)"); ax[0].set_ylabel("AUC: real treaty vs fake treaty")
ax[0].set_ylim(0.45, 1.02)
ax[0].set_title("Figure 6a.  Same template, real cities on both sides — can the probe still\ntell a real treaty from an invented one?", fontsize=10.8, loc="left")
ax[0].legend(frameon=False, fontsize=9, loc="lower right")
for sp in ("top", "right"):
    ax[0].spines[sp].set_visible(False)
# panel b: output-confidence overlap (why the output alone is not enough)
rng = np.linspace(0, 1, 100)
jr = 0.06 * (np.arange(nR) % 2 * 2 - 1); jf = 0.06 * (np.arange(nF) % 2 * 2 - 1)
ax[1].scatter(maxp[:nR], 1 + jr, s=55, color="#2e7d32", edgecolor="white", lw=.5, label="real treaty (has a true year)", zorder=3)
ax[1].scatter(maxp[nR:], 0 + jf, s=55, color="#c0392b", marker="s", edgecolor="white", lw=.5, label="fake treaty (invented)", zorder=3)
for i, (c, ty) in enumerate(FAKE):
    if maxp[nR + i] > 0.45:
        ax[1].annotate(c, (maxp[nR + i], jf[i]), xytext=(0, 7), textcoords="offset points", fontsize=6.8, ha="center", color="#a11")
ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["fake", "real"])
ax[1].set_xlabel("model's peak confidence — max P(year over the window)")
ax[1].set_ylim(-.5, 1.5); ax[1].set_xlim(0, 1)
ax[1].set_title("Figure 6b.  Output confidence overlaps — some fakes are as\npeaked as real memories (why the interior is needed)", fontsize=10.8, loc="left")
ax[1].legend(frameon=False, fontsize=8.5, loc="upper center")
for sp in ("top", "right", "left"):
    ax[1].spines[sp].set_visible(False)
ax[1].tick_params(left=False)
fig.suptitle("Minimal-pair control — real vs invented treaties, identical construction · Ranke-4B-1929 base.", fontsize=12, y=1.0, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
FIG = os.path.join(common.FIGURES, "fig_minimal_pairs.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG}")
