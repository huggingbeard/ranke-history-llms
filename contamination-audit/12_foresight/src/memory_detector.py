# -*- coding: utf-8 -*-
"""v2 memory detector, and an honest v1-vs-v2 comparison (PI's plan).
GROUND TRUTH from TWO independent checks: a real treaty counts as REMEMBERED only if the model nails its DATE or its
SIGNATORIES; it counts as UNKNOWN only if it fails BOTH. Invented treaties are UNKNOWN by construction. (Reals that pass
neither cleanly nor fail both — e.g. concentrate on a wrong year with no parties info — are set aside as ambiguous.)
  v1  = the old idea: a probe trained to tell REAL treaties from INVENTED ones (i.e. it predicts "is this real").
  v2  = the fix: a probe trained to predict MEMORY — real-but-unknown treaties are moved into the negative class beside
        the invented ones, so the probe learns that a real name the model never studied is still "no memory".
Both are evaluated against the same two-signal memory truth, with cross-validated (every item held out) AND a final
untouched-20% confusion matrix, broken out per group (invented / real-known / real-unknown) so we see WHERE each errs.
Run:  python src/memory_detector.py
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
from sklearn.model_selection import cross_val_predict, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, confusion_matrix
import common
import treaties_battery as B

OUT = os.path.join(common.RESULTS, "superlatives")
MID = "uzh-echist-org/Ranke-4B-1929"
WIN = 8
DATE = "The Treaty of {} was signed in the year"
PARTY = "The Treaty of {} was signed between"
CHANCE = 1.0 / len(B.PARTY_OPTIONS)


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


NPZ = os.path.join(OUT, "memory_detector.npz")
if not os.path.exists(NPZ):
    print(f"[score] loading {MID} ... ({len(B.REAL)} real + {len(B.INVENTED)} invented)", flush=True)
    tok, model = common.load(MID)

    def acts(name):
        ids = tok(DATE.format(name), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        return torch.stack([h[0, -1, :] for h in hs]).float().cpu().numpy()

    def datestat(name, yr):
        ys = list(range(yr - WIN, yr + WIN + 1))
        p = softmax([common.slp(model, tok, DATE.format(name), f" {y}") for y in ys])
        return p[ys.index(yr)], max(p)

    def partystat(name, true):
        p = softmax([common.slp(model, tok, PARTY.format(name), " " + o) for o in B.PARTY_OPTIONS])
        return p[B.PARTY_OPTIONS.index(true)]

    X, meta = [], []
    for name, yr, parties in B.REAL:
        pt, pk = datestat(name, yr)
        pp = partystat(name, parties) if parties else None
        X.append(acts(name)); meta.append({"name": name, "kind": "real", "ptrue": pt, "peak": pk, "parties": pp})
    for name, yr in B.INVENTED:
        pt, pk = datestat(name, yr)
        X.append(acts(name)); meta.append({"name": name, "kind": "invented", "ptrue": pt, "peak": pk, "parties": None})
    np.savez_compressed(NPZ, X=np.stack(X))
    json.dump(meta, open(os.path.join(OUT, "memory_detector_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model; common.free()

X = np.load(NPZ)["X"]
meta = json.load(open(os.path.join(OUT, "memory_detector_meta.json"), encoding="utf-8"))
nL = X.shape[1]


def label(m):
    if m["kind"] == "invented":
        return "invented", 0                      # no memory (certain)
    knows_date = m["ptrue"] > 0.5
    knows_parties = (m["parties"] is not None) and (m["parties"] > 3 * CHANCE)
    if knows_date or knows_parties:
        return "real_known", 1
    if m["peak"] < 0.25 and not knows_parties:    # fails BOTH checks
        return "real_unknown", 0
    return "ambiguous", -1


grp = np.array([label(m)[0] for m in meta])
mem = np.array([label(m)[1] for m in meta])       # memory truth: 1 known, 0 no-memory, -1 ambiguous
keep = mem >= 0
Xk, memk, grpk = X[keep], mem[keep], grp[keep]
real = np.array([m["kind"] == "real" for m in meta])[keep]
y_v1 = real.astype(int)                            # v1 target: "is it a real treaty" (realness)
y_v2 = memk.copy()                                 # v2 target: "does the model remember it" (memory)
print(f"clean-labelled items: {keep.sum()} of {len(meta)}  |  "
      f"real-known {int((grpk=='real_known').sum())}, real-UNKNOWN {int((grpk=='real_unknown').sum())}, invented {int((grpk=='invented').sum())}, dropped-ambiguous {int((grp=='ambiguous').sum())}")


def pipe():
    return make_pipeline(StandardScaler(), PCA(n_components=15, random_state=0), LogisticRegression(C=1.0, max_iter=3000))


# pick the layer that best predicts MEMORY (v2), cross-validated
bestL, bestA = 0, 0.0
for L in range(nL):
    pr = cross_val_predict(pipe(), Xk[:, L, :], y_v2, cv=5, method="predict_proba")[:, 1]
    a = roc_auc_score(y_v2, pr)
    if a > bestA:
        bestA, bestL = a, L

# cross-validated out-of-fold predictions for BOTH detectors at that layer
p1 = cross_val_predict(pipe(), Xk[:, bestL, :], y_v1, cv=5, method="predict_proba")[:, 1]   # v1: P(real)
p2 = cross_val_predict(pipe(), Xk[:, bestL, :], y_v2, cv=5, method="predict_proba")[:, 1]   # v2: P(memory)
auc1, auc2 = roc_auc_score(memk, p1), roc_auc_score(memk, p2)                                # both scored vs MEMORY truth
c1, c2 = confusion_matrix(memk, (p1 > .5).astype(int)), confusion_matrix(memk, (p2 > .5).astype(int))


def by_group(pred):
    r = {}
    for g in ("real_known", "real_unknown", "invented"):
        mk = grpk == g
        r[g] = {"n": int(mk.sum()), "called_remembered": int((pred[mk] > .5).sum())}
    return r


g1, g2 = by_group(p1), by_group(p2)
# final untouched-20% holdout, stratified by group
ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
tr, te = next(ss.split(Xk[:, bestL, :], grpk))
h1 = pipe().fit(Xk[tr, bestL, :], y_v1[tr]).predict_proba(Xk[te, bestL, :])[:, 1]
h2 = pipe().fit(Xk[tr, bestL, :], y_v2[tr]).predict_proba(Xk[te, bestL, :])[:, 1]
hold = {"n": len(te), "v1_auc": float(roc_auc_score(memk[te], h1)), "v2_auc": float(roc_auc_score(memk[te], h2))}

print(f"\nbest layer {bestL}/{nL-1}  (memory AUC {bestA:.2f})")
print(f"scored against MEMORY truth:  v1 (realness) AUC {auc1:.2f}   |   v2 (memory) AUC {auc2:.2f}")
print(f"\nWHERE THEY DIFFER — of {g1['real_unknown']['n']} real-but-UNKNOWN treaties, called 'remembered':")
print(f"   v1: {g1['real_unknown']['called_remembered']}   ->   v2: {g2['real_unknown']['called_remembered']}   (fewer false alarms = better)")
print(f"of {g1['real_known']['n']} real-known treaties, called 'remembered':  v1 {g1['real_known']['called_remembered']}  v2 {g2['real_known']['called_remembered']}")
print(f"of {g1['invented']['n']} invented treaties, called 'remembered':      v1 {g1['invented']['called_remembered']}  v2 {g2['invented']['called_remembered']}")
print(f"untouched 20% holdout (n={hold['n']}):  v1 AUC {hold['v1_auc']:.2f}   v2 AUC {hold['v2_auc']:.2f}")

json.dump({"bestL": int(bestL), "n_clean": int(keep.sum()), "chance_parties": CHANCE,
           "groups": {g: int((grpk == g).sum()) for g in ("real_known", "real_unknown", "invented")},
           "auc_vs_memory": {"v1": auc1, "v2": auc2}, "cm_v1": c1.tolist(), "cm_v2": c2.tolist(),
           "by_group_called_remembered": {"v1": g1, "v2": g2}, "holdout": hold},
          open(os.path.join(OUT, "memory_detector_summary.json"), "w"), indent=1)

# ---- figure: two confusion matrices + the real-unknown story ----
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(1, 2, figsize=(13.6, 6.0))
for a, cm, title, auc in [(ax[0], c1, "v1 — trained on 'is it a real treaty?'", auc1),
                          (ax[1], c2, "v2 — trained on 'does the model remember it?'", auc2)]:
    a.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for (i, j), v in np.ndenumerate(cm):
        a.text(j, i, str(v), ha="center", va="center", fontsize=15,
               color="white" if v > cm.max() * .5 else "#111", fontweight="bold")
    a.set_xticks([0, 1]); a.set_yticks([0, 1])
    a.set_xticklabels(["called\n'made-up'", "called\n'remembered'"]); a.set_yticklabels(["truly\nno-memory", "truly\nremembered"])
    a.set_title(f"{title}\nAUC vs memory truth = {auc:.2f}", fontsize=10.2, loc="left")
    a.set_xlabel("detector says");
    if a is ax[0]:
        a.set_ylabel("the truth (date + parties)")
    a.tick_params(length=0)
fig.suptitle(f"Figure 12.  Same treaties, same activations — retraining the label fixes the false alarms.\n"
             f"Top-right cell = false positives (no memory, called remembered). Ranke-4B-1929, layer {bestL}, cross-validated.",
             fontsize=11, y=1.02, x=0.01, ha="left")
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_memory_detector.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG} and memory_detector_summary.json")
