# -*- coding: utf-8 -*-
"""
Sweep how HARD we null the realness direction (INLP stop threshold) and read the tradeoff:
closing the in-distribution leak vs distorting the external obscure treaties. Picks the operating point
honestly instead of tuning to one number. GPU-free (cached activations). Run: python src/memory_detector_v3_sweep.py
"""
import os, json, csv, math
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import common
import treaties_battery as B

OUT = os.path.join(common.RESULTS, "superlatives")
CHANCE = 1.0 / len(B.PARTY_OPTIONS)
LAYER = json.load(open(os.path.join(OUT, "memory_detector_summary.json"), encoding="utf-8"))["bestL"]
X = np.load(os.path.join(OUT, "memory_detector.npz"))["X"]
meta = json.load(open(os.path.join(OUT, "memory_detector_meta.json"), encoding="utf-8"))


def label(m):
    if m["kind"] == "invented":
        return "invented", 0
    kd = m["ptrue"] > 0.5
    kp = (m["parties"] is not None) and (m["parties"] > 3 * CHANCE)
    if kd or kp:
        return "real_known", 1
    if m["peak"] < 0.25 and not kp:
        return "real_unknown", 0
    return "ambiguous", -1


grp = np.array([label(m)[0] for m in meta]); mem = np.array([label(m)[1] for m in meta])
keep = mem >= 0
XL = X[keep, LAYER, :].astype(np.float64); mem = mem[keep]; grp = grp[keep]
name = np.array([m["name"] for m in meta])[keep]
real = np.array([m["kind"] == "real" for m in meta])[keep].astype(int)
nomem = mem == 0; knownm = grp == "real_known"; runk = grp == "real_unknown"; inv = grp == "invented"

exp = {}
with open(os.path.join(OUT, "treaty_exposure.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        exp[r["treaty"]] = float(r["exposure_pre1929"])
logx = lambda v: math.log10(v) if v > 0 else -13.0
unk_idx = np.where(runk)[0]; xe = np.array([logx(exp.get(name[i], 0.0)) for i in unk_idx])

ob = np.load(os.path.join(OUT, "obscure_real.npz"))
Xo = ob["X"][:, LAYER, :].astype(np.float64); o_studied = ob["ptrue"] > 0.5; o_unstud = ob["peak"] < 0.25
KUCUK = 5   # index of Küçük Kaynarca (real, unstudied) in the OBSCURE list


def learn_null(Xs, real_tr, nomem_tr, auc_stop, max_iter=40):
    dirs = []; Xc = Xs.copy()
    for _ in range(max_iter):
        sub, ys = Xc[nomem_tr], real_tr[nomem_tr]
        if len(np.unique(ys)) < 2 or np.bincount(ys).min() < 3:
            break
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(sub, ys)
        if roc_auc_score(ys, clf.decision_function(sub)) < auc_stop:
            break
        w = clf.coef_[0].astype(np.float64)
        for u in dirs:
            w -= (w @ u) * u
        nrm = np.linalg.norm(w)
        if nrm < 1e-9:
            break
        w /= nrm; dirs.append(w); Xc = Xc - (Xc @ w)[:, None] * w[None, :]
    return dirs


def project(A, dirs):
    A = A.copy()
    for w in dirs:
        A = A - (A @ w)[:, None] * w[None, :]
    return A


def probe():
    return make_pipeline(PCA(n_components=15, random_state=0), LogisticRegression(C=1.0, max_iter=3000))


def evaluate(auc_stop):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(XL)); ndir = []
    for tr, te in skf.split(XL, grp):
        mu, sd = XL[tr].mean(0), XL[tr].std(0) + 1e-6
        Xtr, Xte = (XL[tr] - mu) / sd, (XL[te] - mu) / sd
        dirs = learn_null(Xtr, real[tr], nomem[tr], auc_stop)
        ndir.append(len(dirs))
        oof[te] = probe().fit(project(Xtr, dirs), mem[tr]).predict_proba(project(Xte, dirs))[:, 1]
    mu, sd = XL.mean(0), XL.std(0) + 1e-6
    Xall = (XL - mu) / sd; dirs = learn_null(Xall, real, nomem, auc_stop)
    clf = probe().fit(project(Xall, dirs), mem)
    obm = clf.predict_proba(project((Xo - mu) / sd, dirs))[:, 1]
    return {"auc_stop": auc_stop, "ndirs": int(np.mean(ndir)),
            "memory_auc": roc_auc_score(mem[knownm | runk], oof[knownm | runk]),
            "leak_auc": roc_auc_score(real[nomem], oof[nomem]),
            "gap": float(oof[runk].mean() - oof[inv].mean()),
            "corr_exp": float(np.corrcoef(xe, oof[unk_idx])[0, 1]),
            "ob_studied": float(obm[o_studied].mean()), "ob_unstud": float(obm[o_unstud].mean()),
            "ob_FP": int((obm[o_unstud] > 0.5).sum()), "kucuk": float(obm[KUCUK])}


print(f"layer {LAYER} | real_known {knownm.sum()} real_unknown {runk.sum()} invented {inv.sum()}\n")
print(f"{'stop':>5}{'ndir':>5} | {'memAUC':>7}{'leakAUC':>8}{'gap':>7}{'corrExp':>8} | "
      f"{'obStud':>7}{'obUnst':>7}{'obFP':>5}{'kucuk':>7}")
print(f"{'(want)':>5}{'':>5} | {'high':>7}{'0.50':>8}{'0.00':>7}{'0.00':>8} | {'high':>7}{'low':>7}{'0':>5}{'low':>7}")
rows = []
for stop in [1.01, 0.80, 0.72, 0.65, 0.58, 0.55]:   # 1.01 -> no null (=v2); lower -> more aggressive
    r = evaluate(stop); rows.append(r)
    tag = " (=v2, no null)" if stop > 1 else ""
    print(f"{stop:>5.2f}{r['ndirs']:>5} | {r['memory_auc']:>7.2f}{r['leak_auc']:>8.2f}{r['gap']:>+7.2f}{r['corr_exp']:>8.2f} | "
          f"{r['ob_studied']:>7.2f}{r['ob_unstud']:>7.2f}{r['ob_FP']:>5}{r['kucuk']:>7.2f}{tag}")
json.dump(rows, open(os.path.join(OUT, "memory_detector_v3_sweep.json"), "w"), indent=1)
print("\nwrote memory_detector_v3_sweep.json")
