# -*- coding: utf-8 -*-
"""PI's question: for a leak detector we must not MISS leaked facts. So combine the two signals with OR — call a fact
"clean" only if it shows NEITHER a year-spike NOR a high memory-meter; flag it otherwise. How accurate is that, vs each
signal alone? Confusion matrices on the treaty battery, against the two-signal memory truth (known = nails date OR
signatories). Three rules: year-spike alone, meter alone, and (spike OR meter). The cell that matters for leak-hunting is
MISSED LEAKS (truly-has-memory but called clean) — OR-ing minimises it.
Caveat: the year-spike is part of how "known" is defined, so it is near-perfect by construction; read the meter's real
contribution in (a) the extra memories it catches that the spike misses, and (b) the false alarms it adds on no-memory.
Run:  python src/combined_rule.py
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
from sklearn.metrics import confusion_matrix
import treaties_battery as B

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "superlatives")
CHANCE = 1.0 / len(B.PARTY_OPTIONS)
X = np.load(os.path.join(OUT, "memory_detector.npz"))["X"]
meta = json.load(open(os.path.join(OUT, "memory_detector_meta.json"), encoding="utf-8"))
bestL = json.load(open(os.path.join(OUT, "memory_detector_summary.json"), encoding="utf-8"))["bestL"]


def label(m):
    if m["kind"] == "invented":
        return 0
    kd = m["ptrue"] > 0.5
    kp = (m["parties"] is not None) and (m["parties"] > 3 * CHANCE)
    if kd or kp:
        return 1
    if m["peak"] < 0.25 and not kp:
        return 0
    return -1


mem = np.array([label(m) for m in meta])
clean = mem >= 0
# v2 meter, out-of-fold
pipe = make_pipeline(StandardScaler(), PCA(n_components=15, random_state=0), LogisticRegression(C=1.0, max_iter=3000))
meter = cross_val_predict(pipe, X[clean, bestL, :], mem[clean], cv=5, method="predict_proba")[:, 1]
date = np.array([m["ptrue"] for m in meta])[clean]
truth = mem[clean]                                    # 1 = has memory (a leak, if post-cutoff), 0 = no memory (clean)

RULES = [("year-spike alone", (date > 0.5)),
         ("memory-meter alone", (meter > 0.5)),
         ("EITHER fires (spike OR meter)", (date > 0.5) | (meter > 0.5))]


def stats(flag):
    tp = int((flag & (truth == 1)).sum()); fn = int((~flag & (truth == 1)).sum())
    fp = int((flag & (truth == 0)).sum()); tn = int((~flag & (truth == 0)).sum())
    rec = tp / (tp + fn); prec = tp / (tp + fp) if (tp + fp) else 0.0
    return tp, fn, fp, tn, rec, prec


print(f"{'rule':32}{'recall':>8}{'missed leaks':>14}{'false alarms':>14}{'precision':>11}")
res = {}
for nm, flag in RULES:
    tp, fn, fp, tn, rec, prec = stats(flag)
    res[nm] = dict(tp=tp, fn=fn, fp=fp, tn=tn, rec=rec, prec=prec)
    print(f"{nm:32}{100*rec:>7.0f}%{fn:>14}{fp:>14}{100*prec:>10.0f}%")
n_pos = int((truth == 1).sum()); n_neg = int((truth == 0).sum())
print(f"\n({n_pos} treaties the model truly remembers, {n_neg} it truly doesn't; meter read at layer {bestL})")
print("Combined OR catches the most memories (fewest missed leaks) — at the cost of the meter's false alarms.")
json.dump({nm: res[nm] for nm in res}, open(os.path.join(OUT, "combined_rule_summary.json"), "w"), indent=1)

# ---- three confusion matrices ----
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.4))
for a, (nm, flag) in zip(ax, RULES):
    cm = confusion_matrix(truth, flag.astype(int), labels=[0, 1])   # rows: truth 0/1 ; cols: clean/flagged
    a.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for (i, j), v in np.ndenumerate(cm):
        miss = (i == 1 and j == 0)
        a.text(j, i, str(v), ha="center", va="center", fontsize=16, fontweight="bold",
               color="#b00" if miss and v > 0 else ("white" if v > cm.max() * .5 else "#111"))
    a.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False, edgecolor="#b00", lw=2.2))  # highlight missed-leak cell
    r = res[nm]
    a.set_xticks([0, 1]); a.set_yticks([0, 1])
    a.set_xticklabels(["called\nCLEAN", "FLAGGED"]); a.set_yticklabels(["truly\nno-memory", "truly\nremembered"])
    a.set_title(f"{nm}\ncatches {100*r['rec']:.0f}% of leaks · {r['fn']} missed · {r['fp']} false alarms", fontsize=9.6, loc="left")
    a.set_xlabel("the rule says"); a.tick_params(length=0)
ax[0].set_ylabel("the truth (date + signatories)")
fig.suptitle("Figure 15.  Combining the two signals to not miss leaks. Red box = MISSED LEAKS (the costly cell). "
             "Ranke-4B-1929 treaty battery.", fontsize=11, y=1.02, x=0.01, ha="left")
fig.tight_layout()
FIG = os.path.join(os.path.dirname(OUT), "..", "figures", "fig_combined_rule.png")
FIG = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "fig_combined_rule.png"))
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG}")
