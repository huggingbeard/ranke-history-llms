# -*- coding: utf-8 -*-
"""PILOT: is there an internal 'MRI' signature that tells retrieval from confabulation? On Ranke-4B-1929 (base), for
each date-spike event we grab the residual-stream vector at the last prompt token (right before it would emit the year)
at every layer, plus the full-vocab entropy there. Then a linear probe separates KNOWN (real, retrieved) from FICTIONAL
(invented, confabulated), cross-validated, per layer. Key checks: (a) does the probe beat the output signals (P(true),
entropy)? (b) does it GENERALISE to post-cutoff real events it never trained on -> should read as confabulation, unless
LEAKED. Reuses P(true) from date_spike_base_big.json. Local, ~1 forward pass/event.
Run:  python src/probe_mri.py
"""
import os, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import common
import events_dates_big as E

OUT = os.path.join(common.RESULTS, "superlatives")
MID = "uzh-echist-org/Ranke-4B-1929"
NPZ = os.path.join(OUT, "probe_mri_acts.npz")
PT = json.load(open(os.path.join(OUT, "date_spike_base_big.json"), encoding="utf-8"))["1929"]["events"]

# event list with keys matching date_spike_base_big
ITEMS = ([("known::" + l, p, y, "known") for l, p, y in E.KNOWN]
         + [("null::" + l, p, y, "null") for l, p, y in E.NULL]
         + [("test::" + l, p, y, g) for l, p, y, g in E.TEST])

if not os.path.exists(NPZ):
    print(f"[extract] loading {MID} ...", flush=True)
    tok, model = common.load(MID)
    X, ent = [], []
    for key, prompt, ty, kind in ITEMS:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        hs = torch.stack([h[0, -1, :] for h in out.hidden_states]).float().cpu().numpy()   # [n_layers+1, d_model]
        lp = torch.log_softmax(out.logits[0, -1].float(), -1)
        ent.append(float(-(lp.exp() * lp).sum().item()))
        X.append(hs)
    X = np.stack(X)                                        # [n_events, n_layers+1, d_model]
    np.savez_compressed(NPZ, X=X, ent=np.array(ent), keys=np.array([k for k, *_ in ITEMS]))
    del tok, model; common.free()
    print(f"[extract] saved {NPZ}  X={X.shape}", flush=True)

d = np.load(NPZ, allow_pickle=True)
X, ent, keys = d["X"], d["ent"], list(d["keys"])
kind = np.array([k.split("::")[0] for k in keys])
ptrue = np.array([PT[k]["p_true"] * 100 for k in keys])
year = np.array([PT[k]["true"] for k in keys])
CHANCE = 100.0 / (2 * E.WIN + 1)
THR = float(np.percentile(ptrue[kind == "null"], 95))
nL = X.shape[1]

known = kind == "known"
null = kind == "null"
train = known | null
ytr = known[train].astype(int)                            # 1 = retrieval, 0 = confabulation

# ---- per-layer cross-validated probe (KNOWN vs FICTIONAL) ----
acc_by_layer, auc_by_layer = [], []
for L in range(nL):
    XL = X[train, L, :]
    XL = (XL - XL.mean(0)) / (XL.std(0) + 1e-6)
    pred = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), XL, ytr, cv=5, method="predict_proba")[:, 1]
    acc_by_layer.append(((pred > .5).astype(int) == ytr).mean())
    auc_by_layer.append(roc_auc_score(ytr, pred))
bestL = int(np.argmax(auc_by_layer))
# output-signal baselines (how well do P(true) and entropy alone separate known vs fictional)
auc_ptrue = roc_auc_score(ytr, ptrue[train])
auc_ent = roc_auc_score(ytr, -ent[train])

# ---- fit probe at best layer, score every event (incl. held-out test) ----
def z(a, ref):
    return (a - ref.mean(0)) / (ref.std(0) + 1e-6)
Xb = X[:, bestL, :]; ref = Xb[train]
clf = LogisticRegression(max_iter=2000, C=0.5).fit(z(Xb[train], ref), ytr)
score_all = clf.predict_proba(z(Xb, ref))[:, 1]           # P(retrieval) per event
# generalisation: split TEST into unknown (P(true)<=thr) vs leaked (P(true)>thr)
test = kind != "known"; test &= (kind != "null")
t_unknown = test & (ptrue <= THR)
t_leaked = test & (ptrue > THR)

print(f"\nprobe: best layer {bestL}/{nL-1}  AUC {auc_by_layer[bestL]:.2f}  acc {acc_by_layer[bestL]:.2f}")
print(f"output baselines:  P(true) AUC {auc_ptrue:.2f}   entropy AUC {auc_ent:.2f}   (probe beats output = {auc_by_layer[bestL]>max(auc_ptrue,auc_ent)})")
print(f"mean P(retrieval) by group @ layer {bestL}:")
print(f"   KNOWN (real, should be high):     {score_all[known].mean():.2f}")
print(f"   FICTIONAL (should be low):        {score_all[null].mean():.2f}")
print(f"   post-cutoff UNKNOWN (should be low): {score_all[t_unknown].mean():.2f}   (n={t_unknown.sum()})")
print(f"   post-cutoff LEAKED (should be high): {score_all[t_leaked].mean():.2f}   (n={t_leaked.sum()})")
json.dump({"bestL": int(bestL), "auc": float(auc_by_layer[bestL]), "acc": float(acc_by_layer[bestL]),
           "auc_ptrue": float(auc_ptrue), "auc_ent": float(auc_ent),
           "known": float(score_all[known].mean()), "fict": float(score_all[null].mean()),
           "unknown": float(score_all[t_unknown].mean()), "n_unknown": int(t_unknown.sum()),
           "leaked": float(score_all[t_leaked].mean()), "n_leaked": int(t_leaked.sum())},
          open(os.path.join(OUT, "probe_mri_summary.json"), "w"), indent=1)

# ---- figure ----
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.2))
ax[0].plot(range(nL), auc_by_layer, "-o", ms=3, color="#1f5f9e", label="internal probe (residual stream)")
ax[0].axhline(auc_ptrue, color="#c0392b", ls="--", lw=1.2, label=f"output P(true) baseline ({auc_ptrue:.2f})")
ax[0].axhline(auc_ent, color="#e08214", ls=":", lw=1.2, label=f"output entropy baseline ({auc_ent:.2f})")
ax[0].axhline(0.5, color="#999", lw=.8)
ax[0].scatter([bestL], [auc_by_layer[bestL]], s=80, color="#c0392b", zorder=5)
ax[0].set_xlabel("layer (0 = embeddings … final)"); ax[0].set_ylabel("AUC: known (retrieval) vs fictional (confabulation)")
ax[0].set_ylim(0.45, 1.02); ax[0].set_title("Figure 4a.  Where the 'do I know this?' signal lives", fontsize=11.5, loc="left")
ax[0].legend(frameon=False, fontsize=8.5, loc="lower right")
for sp in ("top", "right"):
    ax[0].spines[sp].set_visible(False)
# PCA at best layer, coloured by group
P2 = PCA(2).fit(z(Xb[train], ref)); XY = P2.transform(z(Xb, ref))
for m, c, lab in [(known, "#2e7d32", "known (real, retrieved)"), (null, "#8a8f98", "fictional (confabulated)"),
                  (t_unknown, "#1f5f9e", "post-cutoff unknown"), (t_leaked, "#c0392b", "post-cutoff LEAKED")]:
    ax[1].scatter(XY[m, 0], XY[m, 1], s=42, color=c, edgecolor="white", lw=.5, label=lab, alpha=.9)
ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
ax[1].set_title(f"Figure 4b.  Activation map at layer {bestL} (the 'MRI')", fontsize=11.5, loc="left")
ax[1].legend(frameon=False, fontsize=8.5, loc="best")
for sp in ("top", "right"):
    ax[1].spines[sp].set_visible(False)
fig.suptitle("Internal signature of retrieval vs confabulation — Ranke-4B-1929 base, date-spike events.", fontsize=12.5, y=1.0, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
FIG = os.path.join(common.FIGURES, "fig_probe_mri.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG}")
