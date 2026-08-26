# -*- coding: utf-8 -*-
"""
Does the realness-nulling fix hold on the BIG, diverse battery (278 items: wars, Nobel, painters,
novelists, presidents; real famous / real obscure / invented / false-fact)? GPU-FREE.
Memory label behavioural: studied (ptrue>0.5)=1; blank (peak<0.25)=0; else set aside. Realness = real vs invented.
LEAK test: among BLANK items, can the meter separate real-but-blank from invented? (want AUC 0.5).
Compares three meters: v2 (no null) | v3 GLOBAL null (one realness direction, pooled) | v3 PER-CATEGORY null
(realness nulled within each category, because a real painter-name direction != a real war-name direction).
Output: results/superlatives/bignull_v3_summary.json + figures/fig_bignull_v3.png
Run: python src/bignull_analyze.py
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
real = np.array([m["real_entity"] for m in meta]); cat = np.array([m["cat"] for m in meta]); cls = np.array([m["cls"] for m in meta])

studied = ptrue > 0.50; blank = peak < 0.25
mem = np.where(studied, 1, np.where(blank, 0, -1))
keep = mem >= 0
print(f"labels: studied {int(studied.sum())} | blank {int(blank.sum())} | ambiguous {int((mem==-1).sum())}")
print(f"  among blank: real-but-blank {int((blank&real).sum())}  invented {int((blank&~real).sum())}")

nL = X.shape[1]
memk, realk, catk, clsk = mem[keep], real[keep], cat[keep], cls[keep]
nomem = memk == 0
Xall = X[keep].astype(np.float64)
CATS4 = ["war", "nobel", "painter", "novelist"]   # categories that have invented items (president has none)


def _inlp_step(Xc, mask, real_tr, dirs, auc_stop):
    sub, ys = Xc[mask], real_tr[mask].astype(int)
    if len(np.unique(ys)) < 2 or np.bincount(ys).min() < 3:
        return None
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(sub, ys)
    if roc_auc_score(ys, clf.decision_function(sub)) < auc_stop:
        return None
    w = clf.coef_[0].astype(np.float64)
    for u in dirs:
        w -= (w @ u) * u
    n = np.linalg.norm(w)
    return w / n if n > 1e-9 else None


def learn_null_global(Xs, real_tr, nomem_tr, auc_stop=0.65, mx=40):
    dirs, Xc = [], Xs.copy()
    for _ in range(mx):
        w = _inlp_step(Xc, nomem_tr, real_tr, dirs, auc_stop)
        if w is None:
            break
        dirs.append(w); Xc = Xc - (Xc @ w)[:, None] * w[None, :]
    return dirs


def learn_null_percat(Xs, real_tr, cat_tr, nomem_tr, auc_stop=0.65, mx=15):
    dirs, Xc = [], Xs.copy()
    for c in CATS4:
        for _ in range(mx):
            w = _inlp_step(Xc, nomem_tr & (cat_tr == c), real_tr, dirs, auc_stop)
            if w is None:
                break
            dirs.append(w); Xc = Xc - (Xc @ w)[:, None] * w[None, :]
    return dirs


def project(A, dirs):
    A = A.copy()
    for w in dirs:
        A = A - (A @ w)[:, None] * w[None, :]
    return A


def probe():
    return make_pipeline(PCA(n_components=15, random_state=0), LogisticRegression(C=1.0, max_iter=3000))


def oof_meter(L, mode):
    XL = Xall[:, L, :]
    oof = np.zeros(len(XL))
    strat = np.array([f"{m}{r}" for m, r in zip(memk, realk)])
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(XL, strat):
        mu, sd = XL[tr].mean(0), XL[tr].std(0) + 1e-6
        Xtr, Xte = (XL[tr] - mu) / sd, (XL[te] - mu) / sd
        if mode == "global":
            dirs = learn_null_global(Xtr, realk[tr], nomem[tr]); Xtr, Xte = project(Xtr, dirs), project(Xte, dirs)
        elif mode == "percat":
            dirs = learn_null_percat(Xtr, realk[tr], catk[tr], nomem[tr]); Xtr, Xte = project(Xtr, dirs), project(Xte, dirs)
        oof[te] = probe().fit(Xtr, memk[tr]).predict_proba(Xte)[:, 1]
    return oof


studied_k = memk == 1
mmask = studied_k | (nomem & realk)
best = max(range(8, nL), key=lambda L: roc_auc_score(memk[mmask], oof_meter(L, "none")[mmask]))
print(f"best layer (by v2 memory AUC): {best}/{nL-1}\n")

oofs = {"v2_none": oof_meter(best, "none"), "v3_global": oof_meter(best, "global"), "v3_percat": oof_meter(best, "percat")}


def stats(oof):
    memory_auc = roc_auc_score(memk[mmask], oof[mmask])
    leak_auc = roc_auc_score(realk[nomem], oof[nomem])
    per_cat = {}
    for c in CATS4:
        m = nomem & (catk == c)
        if (m & realk).sum() >= 3 and (m & ~realk).sum() >= 3:
            per_cat[c] = round(float(roc_auc_score(realk[m], oof[m])), 3)
    per_cls = {cc: round(float(oof[clsk == cc].mean()), 3) for cc in ["famous", "obscure", "invented", "false"]}
    return {"memory_auc": round(float(memory_auc), 3), "leak_auc": round(float(leak_auc), 3),
            "per_category_leak_auc": per_cat, "meter_by_class": per_cls}


S = {k: stats(v) for k, v in oofs.items()}
cols = list(oofs.keys())
print(f"{'':30s}" + "".join(f"{c:>12s}" for c in cols) + "    goal")
print(f"{'memory AUC (studied vs blank)':30s}" + "".join(f"{S[c]['memory_auc']:>12.2f}" for c in cols) + "    high")
print(f"{'realness-LEAK AUC (overall)':30s}" + "".join(f"{S[c]['leak_auc']:>12.2f}" for c in cols) + "    ->0.50")
print(f"\nper-category realness-LEAK AUC (want ~0.50 = no leak):")
for c in CATS4:
    print(f"  {c:12s}" + "".join(f"{S[k]['per_category_leak_auc'].get(c, float('nan')):>12.2f}" for k in cols))
print(f"\nmeter by class (blank real should sit at invented level):")
for cc in ["famous", "obscure", "invented", "false"]:
    print(f"  {cc:12s}" + "".join(f"{S[k]['meter_by_class'][cc]:>12.2f}" for k in cols))

json.dump({"meta": {"version": "bignull-v3", "date": "2026-07-10", "layer": int(best), "n_items": len(meta),
                    "n_studied": int(studied.sum()), "n_blank": int(blank.sum()),
                    "blank_real": int((blank & real).sum()), "blank_invented": int((blank & ~real).sum())},
           "results": S}, open(os.path.join(OUT, "bignull_v3_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- figure: per-category leak AUC, three meters ----
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(figsize=(10, 5.6))
xpos = np.arange(len(CATS4)); w = 0.26
colc = {"v2_none": "#c0392b", "v3_global": "#e08214", "v3_percat": "#2e7d32"}
lab = {"v2_none": "v2 (no correction)", "v3_global": "v3 global null", "v3_percat": "v3 per-category null"}
for i, k in enumerate(cols):
    ax.bar(xpos + (i - 1) * w, [S[k]["per_category_leak_auc"].get(c, np.nan) for c in CATS4], width=w,
           color=colc[k], edgecolor="#111", lw=.6, label=lab[k])
ax.axhline(0.5, color="#333", ls="--", lw=1.3)
ax.annotate("0.50 = no leak (goal)", (3.3, 0.51), fontsize=8.5, color="#333", ha="right")
ax.set_xticks(xpos); ax.set_xticklabels(["wars", "Nobel", "painters", "novelists"])
ax.set_ylabel("realness-leak AUC  (0.5 = meter can't tell real from fake among blanks)")
ax.set_ylim(0, 1)
ax.set_title(f"Figure 14.  Realness leak by category on 278 items (layer {best}). The leak is in person-name\n"
             f"categories; a per-category null closes it where a global null cannot. Memory kept (AUC {S['v3_percat']['memory_auc']:.2f}).", fontsize=9.6, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_bignull_v3.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"\nwrote {FIG} and bignull_v3_summary.json")
