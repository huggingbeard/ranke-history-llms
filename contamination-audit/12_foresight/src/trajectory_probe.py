# -*- coding: utf-8 -*-
"""Test the PI's Q2: is a realness-leak (a fake that the final meter calls 'remembered') a LATE bloomer, while a genuine
memory rises EARLY? If so, the layer TRAJECTORY -- not any single layer -- is the right discriminator, and it should
shrink the realness leak. Uses the all-layer last-token states from deep_probe.npz (no new GPU). For every layer we get an
out-of-fold meter score; then we (a) plot the mean trajectory per group, (b) ask whether an EARLY-layer read separates
true positives (known, final>0.5) from false positives (real-unknown, final>0.5), and (c) compare a trajectory classifier
to the best single layer on AUC and on the realness leak.
Run:  python src/trajectory_probe.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import treaties_battery as B

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "superlatives")
CHANCE = 1.0 / len(B.PARTY_OPTIONS)
last = np.load(os.path.join(OUT, "deep_probe.npz"))["last"]           # [n, L+1, d]
meta = json.load(open(os.path.join(OUT, "memory_detector_meta.json"), encoding="utf-8"))
nL = last.shape[1]


def label(m):
    if m["kind"] == "invented":
        return "invented", 0
    kd = m["ptrue"] > 0.5; kp = (m["parties"] is not None) and (m["parties"] > 3 * CHANCE)
    if kd or kp:
        return "known", 1
    if m["peak"] < 0.25 and not kp:
        return "unknown", 0
    return "ambiguous", -1


grp = np.array([label(m)[0] for m in meta]); mem = np.array([label(m)[1] for m in meta])
clean = mem >= 0
idx = np.where(clean)[0]
gk = grp[clean]; yk = mem[clean]


def pipe():
    return make_pipeline(StandardScaler(), PCA(n_components=15, random_state=0), LogisticRegression(C=1.0, max_iter=3000))


# per-layer out-of-fold meter score for every clean item -> trajectory S[i, layer]
S = np.zeros((clean.sum(), nL))
for L in range(nL):
    S[:, L] = cross_val_predict(pipe(), last[clean, L, :], yk, cv=5, method="predict_proba")[:, 1]

FINAL = 25                                                            # the deployed v2 read-out layer
tp = (gk == "known") & (S[:, FINAL] > 0.5)                            # true positives at the final layer
fp = (gk == "unknown") & (S[:, FINAL] > 0.5)                          # realness-leak false positives
print(f"at layer {FINAL}: {tp.sum()} true positives (known), {fp.sum()} realness-leak false positives (real-unknown)")

# (b) does an EARLY read separate true memories from the realness-leak false positives?
for Lp in [15, 18, 20, 22, 25]:
    if tp.sum() and fp.sum():
        a = roc_auc_score(np.r_[np.ones(tp.sum()), np.zeros(fp.sum())], np.r_[S[tp, Lp], S[fp, Lp]])
        print(f"   layer {Lp}: AUC(true memory vs realness-leak) = {a:.2f}   mean score  TP {S[tp,Lp].mean():.2f}  FP {S[fp,Lp].mean():.2f}")

# (c) trajectory classifier vs best single layer: AUC(known vs no-memory) and realness LEAK
def leak(score):
    return float(score[gk == "unknown"].mean() - score[gk == "invented"].mean())


single_auc = [roc_auc_score(yk, S[:, L]) for L in range(nL)]
bestL = int(np.argmax(single_auc))
traj = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)), S, yk, cv=5, method="predict_proba")[:, 1]
print(f"\nbest single layer {bestL}: AUC {single_auc[bestL]:.2f}  leak {leak(S[:,bestL]):+.2f}")
print(f"deployed layer {FINAL}:     AUC {single_auc[FINAL]:.2f}  leak {leak(S[:,FINAL]):+.2f}")
print(f"TRAJECTORY classifier:      AUC {roc_auc_score(yk, traj):.2f}  leak {leak(traj):+.2f}")

# ---- figure ----
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
for g, c in [("known", "#2e7d32"), ("unknown", "#c0392b"), ("invented", "#8a8f98")]:
    m = gk == g
    ax[0].plot(range(nL), S[m].mean(0), "-", lw=2, color=c, label=f"{g} (n={int(m.sum())})")
ax[0].axvline(FINAL, color="#bbb", ls=":", lw=1); ax[0].axhline(0.5, color="#999", lw=.8)
ax[0].set_xlabel("layer"); ax[0].set_ylabel("meter score P(remembered)")
ax[0].set_title("A. Mean meter trajectory by group", fontsize=10.5, loc="left"); ax[0].legend(frameon=False, fontsize=9)
# true positives vs realness-leak false positives, individual trajectories
for i in np.where(tp)[0]:
    ax[1].plot(range(nL), S[i], "-", color="#2e7d32", lw=.8, alpha=.5)
for i in np.where(fp)[0]:
    ax[1].plot(range(nL), S[i], "-", color="#c0392b", lw=1.2, alpha=.8)
ax[1].plot(range(nL), S[tp].mean(0), "-", color="#155724", lw=3, label="TRUE memories (known)")
if fp.sum():
    ax[1].plot(range(nL), S[fp].mean(0), "-", color="#7b0000", lw=3, label="realness-leak FALSE positives")
ax[1].axvline(FINAL, color="#bbb", ls=":", lw=1); ax[1].axhline(0.5, color="#999", lw=.8)
ax[1].set_xlabel("layer"); ax[1].set_ylabel("meter score")
ax[1].set_title("B. Do the false positives bloom LATE? (each line one fact)", fontsize=10.5, loc="left"); ax[1].legend(frameon=False, fontsize=9)
for a in ax:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.grid(color="#eee", lw=.5); a.set_ylim(-.03, 1.03)
fig.suptitle("Figure 20.  Layer trajectory as the discriminator: true memory (early) vs realness leak (late). Ranke-4B-1929, treaty battery.",
             fontsize=11.5, y=1.01, x=0.01, ha="left")
fig.tight_layout()
FIG = os.path.join(os.path.dirname(OUT), "..", "figures", "fig_trajectory.png")
FIG = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "fig_trajectory.png"))
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG}")
