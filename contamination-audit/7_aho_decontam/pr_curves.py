"""
pr_curves.py — precision-recall curves for the two content signals, at REAL prevalence.

From confidence_raw.json (per book: #distinct post-1945 words, #distinct post-1945 years), drawn
on 300 modern + 2000 old books. Real corpus prevalence of modern books is ~1.3%, so we weight the
old (negative) samples up to that ratio — otherwise precision is flattered by the balanced sample.

Two signals compared (sweeping the count threshold gives each curve):
  YEARS  : number of distinct post-1945 years printed in the text   (high recall, low precision)
  WORDS  : number of distinct validated post-1945 tripwire n-grams   (high precision, low recall)
  BOTH   : words + years (the confidence score)
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
PI = 0.013   # measured fraction of LoC-PD-Books that is actually post-1945 (358/28000)

d = json.loads((HERE / "confidence_raw.json").read_text())
mod, old = d["modern"], d["old"]
n_mod, n_old = len(mod), len(old)
w_old = (n_mod / n_old) * ((1 - PI) / PI)     # reweight negatives to real prevalence
y_true = np.r_[np.ones(n_mod), np.zeros(n_old)]
wt = np.r_[np.ones(n_mod), np.full(n_old, w_old)]

def curve(scorer, label):
    s = np.array([scorer(r) for r in mod] + [scorer(r) for r in old], float)
    p, r, _ = precision_recall_curve(y_true, s, sample_weight=wt)
    ap = average_precision_score(y_true, s, sample_weight=wt)
    # precision at integer operating points
    def at(thr):
        tpr = np.mean([scorer(x) >= thr for x in mod])
        fpr = np.mean([scorer(x) >= thr for x in old])
        prec = tpr * PI / (tpr * PI + fpr * (1 - PI)) if (tpr*PI + fpr*(1-PI)) else 1.0
        return tpr, prec
    print(f"\n{label}: average precision (at {PI:.1%} prevalence) = {ap:.3f}")
    for thr in (1, 2, 3, 5):
        tpr, prec = at(thr)
        print(f"   >= {thr}: recall {tpr*100:5.1f}%   precision {prec*100:5.1f}%")
    return r, p, ap

plt.figure(figsize=(7.5, 6))
for scorer, label, col in [
    (lambda r: r[1], "years printed (count)", "#c0392b"),
    (lambda r: r[0], "tripwire words / n-grams (count)", "#2c3e50"),
    (lambda r: r[0] + r[1], "both (confidence score)", "#1f7a47"),
]:
    rec, prec, ap = curve(scorer, label)
    plt.plot(rec * 100, prec * 100, "-", lw=2, color=col, label=f"{label}  (AP={ap:.2f})")
plt.axhline(PI * 100, ls="--", color="#999", lw=1)
plt.text(2, PI*100 + 1.5, f"chance precision = {PI*100:.1f}% (the modern base rate)", fontsize=8, color="#777")
plt.xlabel("recall — % of modern books caught"); plt.ylabel("precision — % of flags that are truly modern")
plt.title(f"Precision-recall at real prevalence (~{PI*100:.1f}% modern)\n"
          "year-count vs word-count signal")
plt.xlim(0, 101); plt.ylim(0, 101); plt.grid(alpha=.3); plt.legend(loc="upper right")
plt.tight_layout(); plt.savefig(HERE / "pr_curves.png", dpi=130); plt.close()
print("\nsaved -> pr_curves.png")
