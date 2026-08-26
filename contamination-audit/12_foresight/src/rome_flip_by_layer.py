# -*- coding: utf-8 -*-
"""Follow-up to rome_inject.py: the probe flipped only 2/5 injected facts at layer 28, though all 5 output the planted
year confidently. Question: do the NON-flippers look like retrieval at some OTHER layer (the injected signature lives
elsewhere and fades), or NOWHERE (a rank-1 edit never reproduces the natural-memory signature)? Reads the per-layer
probe (real vs fake 'Treaty of X', leave-one-out) across ALL layers for each injected fact's post-edit activation, its
pre-edit (clean fake) activation, and the un-injected control fakes. Uses activations saved by rome_inject.py.
Run:  python src/rome_flip_by_layer.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import common

OUT = os.path.join(common.RESULTS, "superlatives")
mp = np.load(os.path.join(OUT, "minimal_pairs.npz"))
Xmp = mp["X"]; ymp = np.r_[np.ones(20), np.zeros(20)]
FAKE_CITIES = ["Bordeaux", "Rotterdam", "Glasgow", "Aberdeen", "Plymouth", "Cardiff", "Leeds", "Dundee", "Swansea",
               "Cork", "Rennes", "Bristol", "Grenoble", "Marseille", "Toulouse", "Le Havre", "Strasbourg", "Rouen",
               "Nancy", "Dijon"]
a = np.load(os.path.join(OUT, "rome_inject_acts.npz"), allow_pickle=True)
inj_pre, inj_post, ctrl_post = a["inj_pre"], a["inj_post"], a["ctrl_post"]
inj_cities, ctrl_cities = list(a["inj_cities"]), list(a["ctrl_cities"])
nL = Xmp.shape[1]


def probe_at(leave_city, L):
    gi = 20 + FAKE_CITIES.index(leave_city)
    mask = np.ones(40, bool); mask[gi] = False
    pipe = make_pipeline(StandardScaler(), PCA(n_components=12, random_state=0), LogisticRegression(C=1.0, max_iter=2000))
    pipe.fit(Xmp[mask, L, :], ymp[mask]); return pipe


pre_L = np.zeros((len(inj_cities), nL)); post_L = np.zeros((len(inj_cities), nL)); ctrl_L = np.zeros((len(ctrl_cities), nL))
for L in range(nL):
    for i, city in enumerate(inj_cities):
        pipe = probe_at(city, L)
        pre_L[i, L] = pipe.predict_proba(inj_pre[i, L][None])[0, 1]
        post_L[i, L] = pipe.predict_proba(inj_post[i, L][None])[0, 1]
    for j, city in enumerate(ctrl_cities):
        ctrl_L[j, L] = probe_at(city, L).predict_proba(ctrl_post[j, L][None])[0, 1]

flippers = [c for i, c in enumerate(inj_cities) if post_L[i, 28] > .5]
print(f"layer-28 flippers: {flippers}")
print(f"peak post-probe across layers, per injected fact (does it look 'retrieved' ANYWHERE?):")
for i, city in enumerate(inj_cities):
    print(f"   {city:10} max {post_L[i].max():.2f} @L{int(post_L[i].argmax())}   |  L28 {post_L[i,28]:.2f}   pre(clean) L28 {pre_L[i,28]:.2f}")

import json
json.dump({"flippers": [str(c) for c in flippers],
           "per_fact": {str(city): {"L28": round(float(post_L[i, 28]), 2), "max": round(float(post_L[i].max()), 2),
                                     "argmaxL": int(post_L[i].argmax())} for i, city in enumerate(inj_cities)}},
          open(os.path.join(OUT, "rome_flip_summary.json"), "w"), indent=1)

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(figsize=(10.2, 6.0))
COL = {"Bristol": "#2e7d32", "Cardiff": "#1f7d4d", "Rennes": "#c0392b", "Glasgow": "#d1662a", "Leeds": "#a11"}
for i, city in enumerate(inj_cities):
    flipped = post_L[i, 28] > .5
    ax.plot(range(nL), post_L[i], "-", lw=2.0 if flipped else 1.4, color=COL.get(city, "#555"),
            label=f"injected {city} ({'FLIPS' if flipped else 'stays confab'})", alpha=.95)
ax.plot(range(nL), ctrl_L.mean(0), "--", lw=1.6, color="#1f5f9e", label="un-injected control fakes (mean)")
ax.plot(range(nL), pre_L.mean(0), ":", lw=1.6, color="#8a8f98", label="injected facts BEFORE edit (mean, clean fake)")
ax.axvline(28, color="#bbb", lw=.9, ls=":"); ax.annotate("read layer\n(Fig 7)", (28, .04), fontsize=7.5, color="#888", ha="center")
ax.axhline(0.5, color="#999", lw=.8)
ax.set_xlabel("layer the probe reads (0 = embeddings … final)")
ax.set_ylabel("probe P(retrieval)  —  1 = looks like a real memory")
ax.set_ylim(-.03, 1.03)
ax.set_title("Figure 8.  Where does an injected fact look 'retrieved'? Per-layer readout after a rank-1 edit.\n"
             "Two facts acquire the natural-memory signature; three force the answer without it — the probe sees the difference.",
             fontsize=10.2, loc="left")
ax.legend(frameon=False, fontsize=8.3, loc="center left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_rome_flip_by_layer.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG}")
