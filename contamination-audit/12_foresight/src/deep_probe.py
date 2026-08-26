# -*- coding: utf-8 -*-
"""Test the PI's j-lens hypothesis: reading the memory signal FURTHER DOWN (at the SUBJECT token, in MID layers, where
ROME localises factual recall) should beat the single late/last-token probe and shrink the REALNESS LEAK.
For the treaty battery we extract residual states at BOTH the subject token and the last token, at EVERY layer, plus a
logit-lens 'year emergence' trajectory (P(true-year token) decoded from each layer). We then measure, per (position,
layer):  (a) AUC(known vs no-memory) -- detector quality, higher better;  (b) LEAK = mean_probe(real-unknown) -
mean_probe(invented) -- both are behaviourally blank, so any gap is pure realness bias, LOWER better.  Hypothesis: the
subject token in mid-layers is high-AUC and low-LEAK. Head-to-head confusion vs the deployed last-token v2.
Run:  python src/deep_probe.py
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
from sklearn.metrics import roc_auc_score, confusion_matrix
import common
import treaties_battery as B

OUT = os.path.join(common.RESULTS, "superlatives")
MID = "uzh-echist-org/Ranke-4B-1929"
DATE = "The Treaty of {} was signed in the year"
CHANCE = 1.0 / len(B.PARTY_OPTIONS)
ITEMS = [(n, y, p) for n, y, p in B.REAL] + [(n, y, None) for n, y in B.INVENTED]

NPZ = os.path.join(OUT, "deep_probe.npz")
if not os.path.exists(NPZ):
    print(f"[deep] loading {MID} ... extracting subject+last states, all layers, {len(ITEMS)} treaties", flush=True)
    tok, model = common.load(MID)
    subj, last, emg = [], [], []
    for name, yr, _ in ITEMS:
        prompt = DATE.format(name)
        sp = len(tok("The Treaty of " + name, add_special_tokens=False).input_ids) - 1
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states           # tuple (L+1) of [1,n,d]
            H = torch.stack([h[0] for h in hs])                                 # [L+1, n, d]
            yt = tok(f" {yr}", add_special_tokens=False).input_ids[0]
            ll = model.lm_head(model.model.norm(H[:, -1, :]))                   # logit-lens at last token: [L+1, vocab]
            e = torch.softmax(ll.float(), -1)[:, yt]                            # P(true-year token) per layer
        subj.append(H[:, sp, :].float().cpu().numpy())
        last.append(H[:, -1, :].float().cpu().numpy())
        emg.append(e.float().cpu().numpy())
    np.savez_compressed(NPZ, subj=np.stack(subj), last=np.stack(last), emg=np.stack(emg))
    del tok, model; common.free()

d = np.load(NPZ)
subj, last, emg = d["subj"], d["last"], d["emg"]
nL = subj.shape[1]
meta = json.load(open(os.path.join(OUT, "memory_detector_meta.json"), encoding="utf-8"))


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
gk = grp[clean]; yk = mem[clean]


def pipe():
    return make_pipeline(StandardScaler(), PCA(n_components=15, random_state=0), LogisticRegression(C=1.0, max_iter=3000))


def sweep(A):
    auc, leak = [], []
    for L in range(nL):
        pr = cross_val_predict(pipe(), A[clean, L, :], yk, cv=5, method="predict_proba")[:, 1]
        auc.append(roc_auc_score(yk, pr))
        leak.append(float(pr[gk == "unknown"].mean() - pr[gk == "invented"].mean()))
    return np.array(auc), np.array(leak)


auc_last, leak_last = sweep(last)
auc_subj, leak_subj = sweep(subj)
# pick best subject-token layer by (high AUC, low leak): maximise AUC - leak among layers with AUC within .03 of max
score_subj = auc_subj - np.abs(leak_subj)
bs = int(np.argmax(score_subj))
bl = int(np.argmax(auc_last - np.abs(leak_last)))
print(f"deployed last-token v2 (~layer 25):  AUC {auc_last[25]:.2f}  leak {leak_last[25]:+.2f}")
print(f"best LAST-token layer {bl}:            AUC {auc_last[bl]:.2f}  leak {leak_last[bl]:+.2f}")
print(f"best SUBJECT-token layer {bs}:         AUC {auc_subj[bs]:.2f}  leak {leak_subj[bs]:+.2f}   <- j-lens hypothesis")


def confusion(A, L):
    pr = cross_val_predict(pipe(), A[clean, L, :], yk, cv=5, method="predict_proba")[:, 1]
    cm = confusion_matrix(yk, (pr > .5).astype(int), labels=[0, 1])
    fp_unknown = int(((pr > .5) & (gk == "unknown")).sum()); n_unknown = int((gk == "unknown").sum())
    return cm, roc_auc_score(yk, pr), fp_unknown, n_unknown


cm_v2, auc_v2, fpu_v2, nu = confusion(last, 25)
cm_sj, auc_sj, fpu_sj, _ = confusion(subj, bs)
print(f"\nreal-UNKNOWN false positives (the realness leak, of {nu}):  last-token v2 {fpu_v2}   ->   subject-token {fpu_sj}")

summ = {"auc_last": auc_last.tolist(), "leak_last": leak_last.tolist(), "auc_subj": auc_subj.tolist(),
        "leak_subj": leak_subj.tolist(), "best_subj_layer": bs, "best_last_layer": bl,
        "v2": {"auc": auc_v2, "fp_unknown": fpu_v2}, "subj": {"auc": auc_sj, "fp_unknown": fpu_sj, "layer": bs}, "n_unknown": nu}
json.dump(summ, open(os.path.join(OUT, "deep_probe_summary.json"), "w"), indent=1)

# ---- figure ----
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
ax[0].plot(range(nL), auc_last, "-o", ms=3, color="#c0392b", label="last token (answer position)")
ax[0].plot(range(nL), auc_subj, "-o", ms=3, color="#1f5f9e", label="subject token (recall position)")
ax[0].axhline(0.5, color="#999", lw=.8); ax[0].set_ylim(0.45, 1.02)
ax[0].set_xlabel("layer"); ax[0].set_ylabel("AUC: known vs no-memory")
ax[0].set_title("A. Detector quality by depth", fontsize=10.5, loc="left"); ax[0].legend(frameon=False, fontsize=8.5, loc="lower right")
ax[1].plot(range(nL), leak_last, "-o", ms=3, color="#c0392b", label="last token")
ax[1].plot(range(nL), leak_subj, "-o", ms=3, color="#1f5f9e", label="subject token")
ax[1].axhline(0, color="#999", lw=.8)
ax[1].set_xlabel("layer"); ax[1].set_ylabel("realness LEAK: meter(real-unknown) - meter(invented)")
ax[1].set_title("B. Realness leak by depth (lower = better)", fontsize=10.5, loc="left"); ax[1].legend(frameon=False, fontsize=8.5)
for gname, col in [("known", "#2e7d32"), ("unknown", "#c0392b"), ("invented", "#8a8f98")]:
    m = grp == gname
    ax[2].plot(range(nL), emg[m].mean(0), "-", lw=2, color=col, label=f"{gname} (n={int(m.sum())})")
ax[2].set_xlabel("layer"); ax[2].set_ylabel("logit-lens P(true-year token)")
ax[2].set_title("C. When the true year emerges (j-lens spirit)", fontsize=10.5, loc="left"); ax[2].legend(frameon=False, fontsize=8.5)
for a in ax:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.grid(color="#eee", lw=.5)
fig.suptitle("Figure 19.  Reading the memory further down the stack (subject token, mid-layers) vs the last-token answer probe. Ranke-4B-1929.",
             fontsize=12, y=1.02, x=0.01, ha="left")
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_deep_probe.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG} and deep_probe_summary.json")
