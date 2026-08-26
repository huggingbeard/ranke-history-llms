# -*- coding: utf-8 -*-
"""PROOF OF CONCEPT (PI): same events, two models, watch the remembering-meter flip with the cutoff.
Events from 1914-1929 are the FUTURE to Ranke-1913 (it was frozen before they happened, so it cannot remember them) and
the PAST to Ranke-1929 (it lived through them and learned them). Nothing about the events, the words, or the questions
changes between the two runs -- only which model we ask. Point each model's own meter at the identical events:
  - pre-1913 real facts  -> should read REMEMBERING on both (both models learned them)      [control, flat high]
  - invented facts       -> should read MAKING-IT-UP on both (neither can learn them)        [control, flat low]
  - 1914-1929 events      -> should read MAKING-IT-UP on 1913 but REMEMBERING on 1929         [the shift]
If the reading jumps from low to high exactly when the event crosses into the training window, the meter is tracking what
each model actually learned, and nothing else. Each model gets its OWN meter (trained on ITS known-vs-invented split),
because the internal pattern for 'remembering' is model-specific.
Run:  python src/cutoff_shift.py
"""
import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import common
import events_dates_big as E

OUT = os.path.join(common.RESULTS, "superlatives")
MODELS = {"1913": "uzh-echist-org/Ranke-4B-1913", "1929": "uzh-echist-org/Ranke-4B-1929"}
ITEMS = ([("known::" + l, p) for l, p, y in E.KNOWN] + [("null::" + l, p) for l, p, y in E.NULL]
         + [("test::" + l, p) for l, p, y, g in E.TEST])
GRP = {"test::" + l: g for l, p, y, g in E.TEST}


def extract(mid, npz):
    if os.path.exists(npz):
        return
    print(f"[extract] loading {mid} ...", flush=True)
    tok, model = common.load(mid)
    X = []
    for key, prompt in ITEMS:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        X.append(torch.stack([h[0, -1, :] for h in hs]).float().cpu().numpy())
    np.savez_compressed(npz, X=np.stack(X), keys=np.array([k for k, _ in ITEMS]))
    del tok, model; common.free()


NPZ = {"1913": os.path.join(OUT, "probe_mri_acts_1913.npz"), "1929": os.path.join(OUT, "probe_mri_acts.npz")}
extract(MODELS["1913"], NPZ["1913"])   # 1929 already extracted by probe_mri.py (identical ITEMS order)


def meter(npz):
    d = np.load(npz, allow_pickle=True)
    X, keys = d["X"], list(d["keys"])
    kind = np.array([k.split("::")[0] for k in keys])
    known, null = kind == "known", kind == "null"
    train = known | null
    ytr = known[train].astype(int)
    best, bestauc = 0, 0.0
    for L in range(X.shape[1]):
        M = X[train, L]; M = (M - M.mean(0)) / (M.std(0) + 1e-6)
        pr = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), M, ytr, cv=5, method="predict_proba")[:, 1]
        a = roc_auc_score(ytr, pr)
        if a > bestauc:
            bestauc, best = a, L
    ref = X[train, best]; mu, sd = ref.mean(0), ref.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=0.5).fit((X[train, best] - mu) / sd, ytr)
    sc = clf.predict_proba((X[:, best] - mu) / sd)[:, 1]
    return {k: float(s) for k, s in zip(keys, sc)}, best, bestauc


M13, b13, a13 = meter(NPZ["1913"])
M29, b29, a29 = meter(NPZ["1929"])
keys = list(np.load(NPZ["1929"], allow_pickle=True)["keys"])
kind = np.array([k.split("::")[0] for k in keys])
shift = [k for k in keys if GRP.get(k) == "1914-29"]                 # events between the two cutoffs

groups = {"pre-1913 real facts": [k for k in keys if k.startswith("known::")],
          "invented facts": [k for k in keys if k.startswith("null::")],
          "1914–1929 events": shift}
print(f"meter layers: 1913 L{b13} (AUC {a13:.2f}), 1929 L{b29} (AUC {a29:.2f})")
print(f"{'group':22}{'mean meter 1913':>18}{'mean meter 1929':>18}")
summ = {"layer_1913": b13, "layer_1929": b29, "auc_1913": a13, "auc_1929": a29, "groups": {}, "events": {}}
for g, ks in groups.items():
    m13, m29 = np.mean([M13[k] for k in ks]), np.mean([M29[k] for k in ks])
    summ["groups"][g] = {"m1913": float(m13), "m1929": float(m29), "n": len(ks)}
    print(f"{g:22}{m13:>18.2f}{m29:>18.2f}")
for k in shift:
    lab = k.split("::", 1)[1]
    summ["events"][lab] = {"m1913": round(M13[k], 2), "m1929": round(M29[k], 2)}


# ---- honest three-way split of the shift events ----
def cat(k):
    if M29[k] < 0.40:
        return "gap29"       # Ranke-1929 never properly learned it either (its own late-1920s gaps)
    if M13[k] > 0.70:
        return "leak13"      # the 1913 model already leaks / knows it
    return "clean"           # clean shift: low on 1913, high on 1929


CATS = {c: [k for k in shift if cat(k) == c] for c in ("clean", "leak13", "gap29")}
CATNAME = {"clean": "clean shift (proof of concept)", "leak13": "1913 already leaks it", "gap29": "Ranke-1929 has a gap too"}
CATCOL = {"clean": "#c0392b", "leak13": "#e08214", "gap29": "#8a8f98"}
summ["categories"] = {}
for c in ("clean", "leak13", "gap29"):
    ks = CATS[c]
    if not ks:
        continue
    m13, m29 = float(np.mean([M13[k] for k in ks])), float(np.mean([M29[k] for k in ks]))
    summ["categories"][c] = {"name": CATNAME[c], "events": [k.split("::", 1)[1] for k in ks], "m1913": m13, "m1929": m29}
    print(f"  {CATNAME[c]:32} n={len(ks):2}  meter {m13:.2f} -> {m29:.2f}   [{', '.join(k.split('::', 1)[1] for k in ks)}]")
json.dump(summ, open(os.path.join(OUT, "cutoff_shift_summary.json"), "w"), indent=1)

# ---- slopegraph, colour-coded by category ----
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(figsize=(11.2, 7.6))
kn13, kn29 = np.mean([M13[k] for k in groups["pre-1913 real facts"]]), np.mean([M29[k] for k in groups["pre-1913 real facts"]])
nu13, nu29 = np.mean([M13[k] for k in groups["invented facts"]]), np.mean([M29[k] for k in groups["invented facts"]])
ax.axhspan(0.75, 1.02, color="#2e7d32", alpha=.05, lw=0)
ax.axhspan(-0.02, 0.25, color="#8a8f98", alpha=.06, lw=0)
ax.annotate("“remembering” zone", (0.5, .995), fontsize=8, color="#2e7d32", ha="center", va="top")
ax.annotate("“making it up” zone", (0.5, .01), fontsize=8, color="#666", ha="center", va="bottom")
# controls
ax.plot([0, 1], [kn13, kn29], "-", color="#2e7d32", lw=2.6, zorder=4)
ax.plot([0, 1], [nu13, nu29], "-", color="#555", lw=2.6, zorder=4)
ax.annotate("pre-1913 real facts — both models learned them", (1.02, kn29), fontsize=8.3, color="#2e7d32", va="center")
ax.annotate("invented facts — neither can learn them", (1.02, nu29 + .01), fontsize=8.3, color="#555", va="center")
# category individual lines + mean
LAB_RIGHT = {"clean", "leak13", "gap29"}
for c in ("leak13", "gap29", "clean"):
    ks = CATS[c]
    if not ks:
        continue
    for k in ks:
        ax.plot([0, 1], [M13[k], M29[k]], "-", color=CATCOL[c], lw=1.0, alpha=.4, zorder=3)
        ax.scatter([0, 1], [M13[k], M29[k]], s=18, color=CATCOL[c], alpha=.5, zorder=3)
    m13, m29 = np.mean([M13[k] for k in ks]), np.mean([M29[k] for k in ks])
    ax.plot([0, 1], [m13, m29], "-", color=CATCOL[c], lw=3.4, zorder=6)
    ax.scatter([0, 1], [m13, m29], s=95, color=CATCOL[c], edgecolor="#111", lw=1, zorder=7)
# label the clean shifters on the right
for k in CATS["clean"]:
    ax.annotate(k.split("::", 1)[1], (1.0, M29[k]), xytext=(6, 0), textcoords="offset points", fontsize=7.4, color="#a11", va="center")
cm13, cm29 = np.mean([M13[k] for k in CATS["clean"]]), np.mean([M29[k] for k in CATS["clean"]])
ax.annotate(f"clean shift\nmean {cm13:.2f} → {cm29:.2f}", (0.44, (cm13 + cm29) / 2), fontsize=9.5, color="#c0392b", ha="right", fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["asked of Ranke-1913\n(1914–29 is its FUTURE)", "asked of Ranke-1929\n(1914–29 is its PAST)"], fontsize=10)
ax.set_xlim(-0.15, 1.5); ax.set_ylim(-0.02, 1.04)
ax.set_ylabel("remembering-meter   —   0 = making it up,   1 = genuinely remembers")
ax.set_yticks([0, .25, .5, .75, 1.0])
ax.set_title("Figure 9.  Same events, two models — move the cutoff and the meter flips.\n"
             "The events, words and questions are identical; only which model we ask changes.", fontsize=10.8, loc="left")
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color="#2e7d32", lw=2.6, label="pre-1913 real facts (control: stays high)"),
           Line2D([0], [0], color="#555", lw=2.6, label="invented facts (control: stays low)"),
           Line2D([0], [0], color=CATCOL["clean"], lw=3.4, label=f"1914–29, CLEAN shift ({len(CATS['clean'])}) — the proof"),
           Line2D([0], [0], color=CATCOL["leak13"], lw=3.4, label=f"1914–29, already leaked into 1913 ({len(CATS['leak13'])})"),
           Line2D([0], [0], color=CATCOL["gap29"], lw=3.4, label=f"1914–29, gap in 1929 too ({len(CATS['gap29'])})")]
ax.legend(handles=handles, loc="center left", frameon=False, fontsize=8.4)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="#eee", lw=.6)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_cutoff_shift.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG} and cutoff_shift_summary.json")
