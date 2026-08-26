# -*- coding: utf-8 -*-
"""
Updated 'which layer' figure, on the BIG 278-item memory-labelled sample (replaces the pilot Fig 4a,
which used small realness-confounded known-vs-fictional labels). Shows, per layer, the memory-meter's
CV AUC (studied-real vs blank-real -> want HIGH) and the realness-leak AUC (real-blank vs invented via the
meter -> want 0.5). Point: the memory signal is a broad late-middle PLATEAU, not a knife-edge, and the
leak is small across it. GPU-FREE (cached activations). Output: figures/fig_layer_curve_big.png + json.
Run: python src/layer_curve_big.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import common

OUT = os.path.join(common.RESULTS, "superlatives")
d = np.load(os.path.join(OUT, "bignull_acts.npz"))
X, ptrue, peak = d["X"], d["ptrue"], d["peak"]
meta = json.load(open(os.path.join(OUT, "bignull_meta.json"), encoding="utf-8"))
real = np.array([m["real_entity"] for m in meta])
studied = ptrue > 0.50; blank = peak < 0.25
mem = np.where(studied, 1, np.where(blank, 0, -1))
keep = mem >= 0
memk, realk = mem[keep], real[keep]
Xall = X[keep].astype(np.float64)
nomem = memk == 0
mmask = (memk == 1) | (nomem & realk)      # studied-real vs blank-real (pure memory test)
nL = X.shape[1]


def oof(L):
    XL = Xall[:, L, :]
    o = np.zeros(len(XL))
    strat = np.array([f"{m}{r}" for m, r in zip(memk, realk)])
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(XL, strat):
        mu, sd = XL[tr].mean(0), XL[tr].std(0) + 1e-6
        pipe = make_pipeline(PCA(15, random_state=0), LogisticRegression(C=1.0, max_iter=3000))
        o[te] = pipe.fit((XL[tr] - mu) / sd, memk[tr]).predict_proba((XL[te] - mu) / sd)[:, 1]
    return o


mem_auc, leak_auc = [], []
for L in range(nL):
    o = oof(L)
    mem_auc.append(roc_auc_score(memk[mmask], o[mmask]))
    leak_auc.append(roc_auc_score(realk[nomem], o[nomem]))
mem_auc, leak_auc = np.array(mem_auc), np.array(leak_auc)
bestL = int(np.argmax(mem_auc))
plateau = np.where(mem_auc >= mem_auc.max() - 0.02)[0]
lo, hi = int(plateau.min()), int(plateau.max())
print(f"best layer {bestL} (AUC {mem_auc[bestL]:.3f}) | plateau (within 0.02 of best): layers {lo}-{hi}")
print(f"memory AUC across plateau: {mem_auc[lo:hi+1].min():.3f}-{mem_auc[lo:hi+1].max():.3f} | "
      f"leak AUC across plateau: {leak_auc[lo:hi+1].min():.2f}-{leak_auc[lo:hi+1].max():.2f}")
json.dump({"bestL": bestL, "plateau": [lo, hi], "mem_auc": [round(float(a), 3) for a in mem_auc],
           "leak_auc": [round(float(a), 3) for a in leak_auc]},
          open(os.path.join(OUT, "layer_curve_big.json"), "w"), indent=1)

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(figsize=(11, 5.8))
ax.axvspan(lo, hi, color="#2e7d32", alpha=.07, lw=0)
ax.plot(range(nL), mem_auc, "-o", ms=3.5, color="#1f5f9e", label="memory-meter: knows vs doesn't (studied-real vs blank-real)")
ax.plot(range(nL), leak_auc, "-s", ms=3.2, color="#c0392b", label="realness leak: real-blank vs invented (want 0.5)")
ax.axhline(0.5, color="#999", ls=":", lw=1)
ax.scatter([bestL], [mem_auc[bestL]], s=90, color="#1f5f9e", zorder=6, edgecolor="#111")
ax.annotate(f"CV-best layer {bestL}\nAUC {mem_auc[bestL]:.2f}", (bestL, mem_auc[bestL]), xytext=(6, -28),
            textcoords="offset points", fontsize=8.6, color="#1f5f9e")
ax.annotate(f"plateau: layers {lo}–{hi}\n(memory AUC within 0.02 of best)", ((lo + hi) / 2, 0.62),
            ha="center", fontsize=8.4, color="#2e7d32")
ax.set_xlabel("layer  (0 = token embeddings … 36 = final)")
ax.set_ylabel("cross-validated AUC")
ax.set_ylim(0.35, 1.02)
ax.set_title("Updated Fig 4a.  Where the memory signal lives — 278 items (wars, Nobel, painters, novelists, presidents).\n"
             "The 'knows vs doesn't' signal is a broad late-middle plateau, not a single lucky layer; the realness leak "
             "stays small across it.", fontsize=9.8, loc="left")
ax.legend(frameon=False, fontsize=9, loc="center left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_layer_curve_big.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG} and layer_curve_big.json")
