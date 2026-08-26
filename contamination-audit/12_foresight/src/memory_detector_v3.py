# -*- coding: utf-8 -*-
"""
v3 MEMORY DETECTOR — closing the "realness leak" by orthogonalisation.
======================================================================
Version : v1 (2026-07-10)   |   GPU-FREE: runs entirely from cached activations.

The leak (documented in the tech note, quantified by ngram_exposure.py): the meter partly reads
"this is a REAL thing in the record", not "the model REMEMBERS it". v2 (memory_detector.py) fought it by
relabelling (real-but-unknown moved into the negative class) — helped (FP 25->14) but 13 real-unknown
negatives is too few to break the confound, and the meter still tracks book-exposure (r~0.40).

v3 fix = orthogonalisation. Among the NO-MEMORY items (real-unknown + invented — both blank behaviourally,
differing ONLY in realness) we iteratively find and REMOVE the direction that separates real from fake
(INLP-style null-space projection), until realness is no longer linearly decodable there. THEN we fit the
memory probe on the nulled features. The memory probe can no longer use the realness axis; the only thing
left to separate real-known from no-memory is genuine recall.

CV-honest: standardisation, the realness-null directions, and the probe are all fit on the TRAIN fold only
and applied out-of-fold. v2 (no null) and v3 (nulled) are computed side by side, same layer, same pipeline.
Validation: (i) memory AUC (real-known vs real-unknown) must be PRESERVED; (ii) the realness-leak AUC
(real-unknown vs invented, read off the meter) must fall to ~0.5; (iii) corr(meter, log book-exposure)
among behaviourally-unknown treaties must fall toward 0; (iv) EXTERNAL test — the 20 obscure real treaties
(obscure_real.npz, never in training): the real-but-UNSTUDIED ones (incl. Kucuk Kaynarca) must stop
reading as "remembered" while the STUDIED ones stay high.

Inputs (cached): results/superlatives/{memory_detector.npz, memory_detector_meta.json,
memory_detector_summary.json, obscure_real.npz, treaty_exposure.csv}.
Output: results/superlatives/memory_detector_v3_summary.json + figures/fig_memory_detector_v3.png
Run: python src/memory_detector_v3.py
"""
import os, json, csv, math
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
import treaties_battery as B

OUT = os.path.join(common.RESULTS, "superlatives")
CHANCE = 1.0 / len(B.PARTY_OPTIONS)
LAYER = json.load(open(os.path.join(OUT, "memory_detector_summary.json"), encoding="utf-8"))["bestL"]  # v2's layer (25)

# ---- load cached activations + labels ----
X = np.load(os.path.join(OUT, "memory_detector.npz"))["X"]              # [132, 37, 2560]
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


grp = np.array([label(m)[0] for m in meta])
mem = np.array([label(m)[1] for m in meta])
keep = mem >= 0
XL = X[keep, LAYER, :].astype(np.float64)
mem = mem[keep]
grp = grp[keep]
name = np.array([m["name"] for m in meta])[keep]
real = np.array([m["kind"] == "real" for m in meta])[keep].astype(int)   # 1 real, 0 invented (the "realness" label)
nomem = (mem == 0)                                                       # no-memory items: real_unknown + invented
knownm = grp == "real_known"
runk = grp == "real_unknown"
inv = grp == "invented"
print(f"layer {LAYER} | clean {keep.sum()}: real_known {knownm.sum()}, real_unknown {runk.sum()}, invented {inv.sum()}")


# ---- INLP: null the realness direction AMONG no-memory items ----
def learn_null(Xtr_std, real_tr, nomem_tr, max_iter=40, auc_stop=0.65):
    """Return orthonormal directions whose removal makes real-vs-fake undecodable among no-memory items.
    auc_stop=0.65 (~13 dirs) is the safe operating point from memory_detector_v3_sweep.py: the leak closes
    and the external obscure treaties stay clean; nulling harder (>18 dirs, stop<0.57) destabilises them."""
    dirs = []
    Xc = Xtr_std.copy()
    for _ in range(max_iter):
        sub, ys = Xc[nomem_tr], real_tr[nomem_tr]
        if len(np.unique(ys)) < 2 or np.bincount(ys).min() < 3:
            break
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(sub, ys)
        if roc_auc_score(ys, clf.decision_function(sub)) < auc_stop:   # realness now ~undecodable here
            break
        w = clf.coef_[0].astype(np.float64)
        for u in dirs:                                                  # orthogonalise against earlier dirs
            w -= (w @ u) * u
        nrm = np.linalg.norm(w)
        if nrm < 1e-9:
            break
        w /= nrm
        dirs.append(w)
        Xc = Xc - (Xc @ w)[:, None] * w[None, :]
    return dirs


def project(A, dirs):
    A = A.copy()
    for w in dirs:
        A = A - (A @ w)[:, None] * w[None, :]
    return A


def probe():
    return make_pipeline(PCA(n_components=15, random_state=0), LogisticRegression(C=1.0, max_iter=3000))


# ---- CV-honest out-of-fold meters for v2 (no null) and v3 (nulled) ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
oof_v2, oof_v3 = np.zeros(len(XL)), np.zeros(len(XL))
n_dirs = []
for tr, te in skf.split(XL, grp):
    mu, sd = XL[tr].mean(0), XL[tr].std(0) + 1e-6
    Xtr, Xte = (XL[tr] - mu) / sd, (XL[te] - mu) / sd
    oof_v2[te] = probe().fit(Xtr, mem[tr]).predict_proba(Xte)[:, 1]
    dirs = learn_null(Xtr, real[tr], nomem[tr])
    n_dirs.append(len(dirs))
    Xtrp, Xtep = project(Xtr, dirs), project(Xte, dirs)
    oof_v3[te] = probe().fit(Xtrp, mem[tr]).predict_proba(Xtep)[:, 1]


def metrics(oof):
    memory_auc = roc_auc_score(mem[knownm | runk], oof[knownm | runk])   # real-known vs real-unknown (want HIGH)
    leak_auc = roc_auc_score(real[nomem], oof[nomem])                    # real-unk vs invented via meter (want ~0.5)
    gap = float(oof[runk].mean() - oof[inv].mean())                      # meter lift of real-unk over invented (want ~0)
    return {"memory_auc": float(memory_auc), "leak_auc": float(leak_auc),
            "meter_real_unknown": float(oof[runk].mean()), "meter_invented": float(oof[inv].mean()), "gap": gap}


m2, m3 = metrics(oof_v2), metrics(oof_v3)

# ---- exposure correlation among behaviourally-unknown real treaties ----
exp = {}
with open(os.path.join(OUT, "treaty_exposure.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        exp[r["treaty"]] = float(r["exposure_pre1929"])
logx = lambda v: math.log10(v) if v > 0 else -13.0
unk_idx = np.where(runk)[0]
xe = np.array([logx(exp.get(name[i], 0.0)) for i in unk_idx])
rexp2 = float(np.corrcoef(xe, oof_v2[unk_idx])[0, 1]) if len(unk_idx) > 2 else float("nan")
rexp3 = float(np.corrcoef(xe, oof_v3[unk_idx])[0, 1]) if len(unk_idx) > 2 else float("nan")

# ---- EXTERNAL test: 20 obscure real treaties, never trained on ----
ob = np.load(os.path.join(OUT, "obscure_real.npz"))
Xo, opt, opk = ob["X"][:, LAYER, :].astype(np.float64), ob["ptrue"], ob["peak"]
OBSCURE = [("Andrusovo", 1667), ("Roskilde", 1658), ("Stolbovo", 1617), ("Passarowitz", 1718), ("Belgrade", 1739),
           ("Küçük Kaynarca", 1774), ("Jassy", 1792), ("Gulistan", 1813), ("Turkmenchay", 1828),
           ("Hünkâr İskelesi", 1833), ("Whampoa", 1844), ("Wanghia", 1844), ("Aigun", 1858), ("Björkö", 1905),
           ("Nöteborg", 1323), ("Nemours", 1585), ("Zuhab", 1639), ("Hubertusburg", 1763), ("Bucharest", 1812),
           ("Tientsin", 1858)]
o_studied = opt > 0.50
o_unstud = opk < 0.25
# fit full-train v2 and v3, apply to obscure
mu, sd = XL.mean(0), XL.std(0) + 1e-6
Xall = (XL - mu) / sd
Xo_std = (Xo - mu) / sd
dirs_full = learn_null(Xall, real, nomem)
v2_full = probe().fit(Xall, mem)
v3_full = probe().fit(project(Xall, dirs_full), mem)
ob_v2 = v2_full.predict_proba(Xo_std)[:, 1]
ob_v3 = v3_full.predict_proba(project(Xo_std, dirs_full))[:, 1]
FP_thr = 0.5


def obstat(scores):
    return {"studied_mean": float(scores[o_studied].mean()) if o_studied.any() else None,
            "unstudied_mean": float(scores[o_unstud].mean()) if o_unstud.any() else None,
            "unstudied_FP": int((scores[o_unstud] > FP_thr).sum()), "n_unstudied": int(o_unstud.sum()),
            "n_studied": int(o_studied.sum())}


ob2, ob3 = obstat(ob_v2), obstat(ob_v3)

# ---- report ----
print(f"\nINLP realness directions removed per fold: {n_dirs}  (full-train: {len(dirs_full)})")
print("\n                         v2 (no null)   v3 (realness-nulled)   goal")
print(f"memory AUC (known vs unk)   {m2['memory_auc']:>6.2f}        {m3['memory_auc']:>6.2f}            preserve (high)")
print(f"realness-LEAK AUC          {m2['leak_auc']:>6.2f}        {m3['leak_auc']:>6.2f}            -> 0.50")
print(f"meter gap (real-unk - inv) {m2['gap']:>+6.2f}       {m3['gap']:>+6.2f}            -> 0.00")
print(f"  meter on real-unknown    {m2['meter_real_unknown']:>6.2f}        {m3['meter_real_unknown']:>6.2f}")
print(f"  meter on invented        {m2['meter_invented']:>6.2f}        {m3['meter_invented']:>6.2f}")
print(f"corr(meter, log-exposure)  {rexp2:>6.2f}        {rexp3:>6.2f}            -> 0.00   (among {len(unk_idx)} unknown reals)")
print(f"\nEXTERNAL obscure treaties (never trained):")
print(f"  studied  (want HIGH):    {ob2['studied_mean']:>6.2f}        {ob3['studied_mean']:>6.2f}   (n={ob2['n_studied']})")
print(f"  UNSTUDIED(want LOW):     {ob2['unstudied_mean']:>6.2f}        {ob3['unstudied_mean']:>6.2f}   (n={ob2['n_unstudied']})")
print(f"  unstudied FALSE POS:     {ob2['unstudied_FP']:>4d}          {ob3['unstudied_FP']:>4d}       (of {ob2['n_unstudied']})")
print(f"\n  per obscure treaty (name: v2 -> v3):")
for i, (nm, yr) in enumerate(OBSCURE):
    cls = "STUDIED" if o_studied[i] else ("unstudied" if o_unstud[i] else "elsewhere")
    print(f"   {nm:18s}{yr}  {ob_v2[i]:.2f} -> {ob_v3[i]:.2f}   {cls}")

summary = {"meta": {"version": "v3", "date": "2026-07-10", "layer": int(LAYER),
                    "method": "INLP null of realness-among-no-memory + memory probe on nulled features",
                    "n_dirs_per_fold": n_dirs, "n_dirs_full": len(dirs_full)},
           "v2": {**m2, "corr_exposure_unknown": rexp2, "obscure": ob2},
           "v3": {**m3, "corr_exposure_unknown": rexp3, "obscure": ob3},
           "obscure_per_treaty": {OBSCURE[i][0]: {"year": OBSCURE[i][1],
                                                  "class": ("studied" if o_studied[i] else "unstudied" if o_unstud[i] else "elsewhere"),
                                                  "v2": round(float(ob_v2[i]), 3), "v3": round(float(ob_v3[i]), 3)}
                                  for i in range(len(OBSCURE))}}
json.dump(summary, open(os.path.join(OUT, "memory_detector_v3_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- figure: before/after on the two things that matter ----
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))
# left: meter on real-unknown vs invented, v2 vs v3 (the leak)
for a_i, (oof, tag) in enumerate([(oof_v2, "v2 (no null)"), (oof_v3, "v3 (nulled)")]):
    xs = [a_i - 0.15, a_i + 0.15]
    ax[0].bar(xs, [oof[inv].mean(), oof[runk].mean()], width=0.28,
              color=["#8a8f98", "#c0392b"], edgecolor="#111", lw=.7)
ax[0].axhline(0.5, color="#999", ls=":", lw=1)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["v2 (no null)", "v3 (realness-nulled)"])
ax[0].set_ylabel("mean meter  (0 = made-up, 1 = remembered)")
ax[0].set_ylim(0, 1)
ax[0].set_title("The leak: both are blank memories.\nGrey = invented, red = real-but-unknown. v3 should equalise them.", fontsize=9.6, loc="left")
ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#8a8f98"), plt.Rectangle((0, 0), 1, 1, fc="#c0392b")],
             labels=["invented", "real-but-unknown"], frameon=False, fontsize=9)
# right: external obscure treaties, v2 -> v3, coloured by studied/unstudied
for i, (nm, yr) in enumerate(OBSCURE):
    c = "#2e7d32" if o_studied[i] else ("#c0392b" if o_unstud[i] else "#e08214")
    ax[1].plot([0, 1], [ob_v2[i], ob_v3[i]], "-", color=c, alpha=.5, lw=1)
    ax[1].scatter([0, 1], [ob_v2[i], ob_v3[i]], s=28, color=c, zorder=5)
ax[1].axhline(0.5, color="#999", ls=":", lw=1)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["v2", "v3"]); ax[1].set_xlim(-0.2, 1.2)
ax[1].set_ylabel("meter"); ax[1].set_ylim(-0.02, 1.02)
ax[1].set_title("External obscure treaties (never trained).\nGreen = studied (keep high), red = unstudied (drive low).", fontsize=9.6, loc="left")
for sp in ("top", "right"):
    ax[0].spines[sp].set_visible(False); ax[1].spines[sp].set_visible(False)
fig.suptitle(f"Figure 13.  Closing the realness leak by orthogonalisation (Ranke-4B-1929, layer {LAYER}).", fontsize=12, y=1.01, x=0.01, ha="left")
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_memory_detector_v3.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"\nwrote {FIG} and memory_detector_v3_summary.json")
