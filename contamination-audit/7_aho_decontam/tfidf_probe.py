"""
tfidf_probe.py — ONE light classifier over ALL signals, LDA-style, with a jointly-optimised cutoff.

Design (per the "use more information is normally good" principle): don't aim a model at the ~29
books the hand rule misses, and don't hand-OR TF-IDF onto the keyword+year rule. Instead train a
single light classifier on ALL modern vs ALL old books, get a continuous discriminant score, and
find the recall-first cutoff. Let the model learn how to weight year-counts, keyword-counts and
n-grams together. That can BOTH raise the recall ceiling (if the missed books carry faint n-gram
signal) AND cut the 18.8% old-book loss (by learning a lone stray '1962' with no modern n-grams is
OCR noise, not a date).

Three models, compared head-to-head, ALL evaluated out-of-sample (K=5 folds by id-hash; every book
scored by a model trained without it; vectorizer fit on train folds only -> no leakage):
  SIGNALS  : numeric only  [yc_raw, yc_ctx, yc_front, yc_cf, kw_raw, kw_dep]  (learned weighting)
  TEXT     : TF-IDF word 1-2 + char_wb 3-5 grams (OCR-robust)
  COMBINED : TEXT + SIGNALS  (everything)
Plus the current hand rule (year>=1 OR kw>=1) as the baseline to beat.

Reported out-of-sample, for each model: old-loss at recall {96.7, 98, 99, 99.5}%, the max recall
reachable at <=10% old-loss, whether it moves the 96.7% ceiling, and the in-sample-vs-OOS gap
(overfit check — a 40k-feature model on 880 positives CAN overfit, unlike the 2-threshold rule).
"""
import json, sys, hashlib
import numpy as np
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
K = 5
NUM = ["yc_raw", "yc_ctx", "yc_front", "yc_cf"]           # year signals
d = json.loads((HERE / "features_v2.json").read_text(encoding="utf-8"))
MOD, OLD = d["modern"], d["old"]
ALL = [dict(r, lab=1) for r in MOD] + [dict(r, lab=0) for r in OLD]
for r in ALL:
    r["f"] = int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % K
    r["kw_raw"] = len(r["mk_raw"]); r["kw_dep"] = len(r["mk_dep"])
y = np.array([r["lab"] for r in ALL]); fold = np.array([r["f"] for r in ALL])
NUMF = NUM + ["kw_raw", "kw_dep"]
Xnum_all = np.array([[r[k] for k in NUMF] for r in ALL], dtype=float)
lex_flag = ((Xnum_all[:, NUMF.index("yc_raw")] >= 1) | (Xnum_all[:, NUMF.index("kw_raw")] >= 1)).astype(int)
print(f"loaded {len(MOD)} modern + {len(OLD)} old ; hand rule (year>=1 OR kw>=1): "
      f"recall {lex_flag[y==1].mean()*100:.2f}%  old-loss {lex_flag[y==0].mean()*100:.2f}%\n")


def make_vec(txt):
    vw = TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=40000, ngram_range=(1, 2),
                         token_pattern=r"(?u)\b\w+\b").fit(txt)
    vc = TfidfVectorizer(sublinear_tf=True, min_df=5, max_features=40000, analyzer="char_wb",
                         ngram_range=(3, 5)).fit(txt)
    return vw, vc


def oos_scores(kind):
    """out-of-sample discriminant score per book for model 'kind'."""
    oos = np.zeros(len(ALL)); ins = np.zeros(len(ALL))
    for kf in range(K):
        tr = fold != kf; te = fold == kf
        blocks_tr, blocks_te = [], []
        if kind in ("text", "combined"):
            vw, vc = make_vec([ALL[i]["tx"] for i in np.where(tr)[0]])
            txt_tr = [ALL[i]["tx"] for i in np.where(tr)[0]]; txt_te = [ALL[i]["tx"] for i in np.where(te)[0]]
            blocks_tr += [vw.transform(txt_tr), vc.transform(txt_tr)]
            blocks_te += [vw.transform(txt_te), vc.transform(txt_te)]
        if kind in ("signals", "combined"):
            sc = StandardScaler().fit(Xnum_all[tr])
            blocks_tr.append(csr_matrix(sc.transform(Xnum_all[tr])))
            blocks_te.append(csr_matrix(sc.transform(Xnum_all[te])))
        Xtr = hstack(blocks_tr).tocsr() if len(blocks_tr) > 1 else blocks_tr[0]
        Xte = hstack(blocks_te).tocsr() if len(blocks_te) > 1 else blocks_te[0]
        clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=3000, solver="liblinear")
        clf.fit(Xtr, y[tr]); oos[te] = clf.predict_proba(Xte)[:, 1]; ins[tr] += clf.predict_proba(Xtr)[:, 1]
        print(f"  [{kind}] fold {kf} done", file=sys.stderr)
    return oos, ins / (K - 1)


def oldloss_at_recall(prob, target):
    pos = np.sort(prob[y == 1]); k = int(np.ceil((1 - target) * len(pos)))
    thr = pos[min(k, len(pos) - 1)]; fl = prob >= thr
    return thr, fl[y == 1].mean(), fl[y == 0].mean()


def max_recall_at_oldloss(prob, budget):
    neg = np.sort(prob[y == 0]); thr = neg[min(int(np.ceil((1 - budget) * len(neg))), len(neg) - 1)]
    fl = prob > thr; return fl[y == 1].mean(), fl[y == 0].mean()


results = {}
print(f"{'model':<10}{'old-loss @96.7%':>16}{'@99%':>9}{'@99.5%':>9}{'max-rec@10%OL':>16}{'overfit gap@99.5%':>19}")
for kind in ("signals", "text", "combined"):
    oos, ins = oos_scores(kind)
    row = {}
    for t in (0.967, 0.99, 0.995):
        _, rec, ol = oldloss_at_recall(oos, t); row[t] = ol
    mr, _ = max_recall_at_oldloss(oos, 0.10)
    _, _, ol_oos = oldloss_at_recall(oos, 0.995); _, _, ol_ins = oldloss_at_recall(ins, 0.995)
    gap = ol_oos - ol_ins
    results[kind] = {"oldloss_967": row[0.967], "oldloss_99": row[0.99], "oldloss_995": row[0.995],
                     "max_recall_at_10pct_oldloss": mr, "overfit_gap_995": gap, "oos": oos.tolist()}
    print(f"{kind:<10}{row[0.967]*100:>14.2f}%{row[0.99]*100:>8.2f}%{row[0.995]*100:>8.2f}%"
          f"{mr*100:>14.1f}%{gap*100:>17.2f} pts")

# does COMBINED move the ceiling on the 29 lexically-invisible books?
oos_c = np.array(results["combined"]["oos"])
resid = np.where((y == 1) & (lex_flag == 0))[0]
thr, _, _ = oldloss_at_recall(oos_c, 0.995)
print(f"\n=== does COMBINED catch the {len(resid)} lexically-invisible modern books? (out-of-sample) ===")
print(f"  their COMBINED score: min {oos_c[resid].min():.3f} median {np.median(oos_c[resid]):.3f} max {oos_c[resid].max():.3f}")
print(f"  at the 99.5%-recall cutoff, {int((oos_c[resid] >= thr).sum())}/{len(resid)} of them clear it "
      f"(vs 0 for the hand rule, which is blind to them by definition)")

for k in results: results[k].pop("oos", None)
json.dump({"n_modern": len(MOD), "n_old": len(OLD), "residual": int(len(resid)),
           "hand_rule": {"recall": float(lex_flag[y==1].mean()), "old_loss": float(lex_flag[y==0].mean())},
           "models": results}, open(HERE / "tfidf_probe_results.json", "w"), indent=1)
print("\nsaved -> tfidf_probe_results.json")
