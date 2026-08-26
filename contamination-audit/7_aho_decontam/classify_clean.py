"""
classify_clean.py — the combined all-signals classifier, evaluated on CLEAN external labels.

Label = OpenLibrary first_publish_year (old <=1945 / modern >=1946), NOT the dataset metadata.
Matched subset only. One light classifier over ALL signals (your design), continuous score, one
recall-first cutoff, all out-of-sample (nested K=5 CV by id-hash; vectorizer fit on train only).

Models compared head-to-head:
  SIGNALS  : numeric [yc_raw,yc_ctx,yc_front,yc_cf, kw_raw,kw_dep, en_ratio, nonlatin_ratio]
  TEXT     : TF-IDF word 1-2 + char_wb 3-5 grams
  COMBINED : TEXT + SIGNALS
Reported OOS per model: old-loss at recall {99, 99.5}%, max recall at <=10% old-loss, overfit gap.
Because the labels are now trustworthy, these recall/old-loss numbers are the real ones.
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
EN = set(("the of and to in a is that for it as was with his he be not by but this all at which have "
          "or from are on an their we you i had has said him her they were").split())
def is_latin(c):
    o = ord(c); return (65 <= o <= 90) or (97 <= o <= 122) or (0xC0 <= o <= 0x24F)
def ratios(t):
    a = [c for c in t if c.isalpha()]
    if not a: return 0.0, 0.0
    nl = sum(1 for c in a if not is_latin(c)) / len(a)
    tk = [w.strip(".,;:!?()[]\"'").lower() for w in t.split()]
    return nl, sum(1 for w in tk if w in EN) / max(len(tk), 1)

feat = json.loads((HERE / "features_v2.json").read_text(encoding="utf-8"))
lab = json.loads((HERE / "ol_labels.json").read_text(encoding="utf-8"))
BY = {r["id"]: r for r in feat["modern"] + feat["old"]}
ALL = []
for i, v in lab.items():
    if v["ol_year"] is None or i not in BY: continue
    r = BY[i]; nl, en = ratios(r["tx"])
    ALL.append({"id": i, "lab": 1 if v["ol_year"] >= 1946 else 0, "tx": r["tx"],
                "num": [r["yc_raw"], r["yc_ctx"], r["yc_front"], r["yc_cf"],
                        len(r["mk_raw"]), len(r["mk_dep"]), en, nl]})
y = np.array([r["lab"] for r in ALL])
fold = np.array([int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % K for r in ALL])
Xnum = np.array([r["num"] for r in ALL], dtype=float)
print(f"matched & labeled: {len(ALL)}  (modern {int(y.sum())}, old {int((1-y).sum())})\n")
if y.sum() < 20 or (1 - y).sum() < 20:
    print("not enough labels yet — rerun when the labeler has progressed further."); sys.exit(0)


def make_vec(txt):
    vw = TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=40000, ngram_range=(1, 2),
                         token_pattern=r"(?u)\b\w+\b").fit(txt)
    vc = TfidfVectorizer(sublinear_tf=True, min_df=5, max_features=40000, analyzer="char_wb",
                         ngram_range=(3, 5)).fit(txt)
    return vw, vc


def oos_scores(kind):
    oos = np.zeros(len(ALL)); ins = np.zeros(len(ALL)); cnt = np.zeros(len(ALL))
    for kf in range(K):
        tr = np.where(fold != kf)[0]; te = np.where(fold == kf)[0]
        bt, be = [], []
        if kind in ("text", "combined"):
            vw, vc = make_vec([ALL[i]["tx"] for i in tr])
            bt += [vw.transform([ALL[i]["tx"] for i in tr]), vc.transform([ALL[i]["tx"] for i in tr])]
            be += [vw.transform([ALL[i]["tx"] for i in te]), vc.transform([ALL[i]["tx"] for i in te])]
        if kind in ("signals", "combined"):
            sc = StandardScaler().fit(Xnum[tr])
            bt.append(csr_matrix(sc.transform(Xnum[tr]))); be.append(csr_matrix(sc.transform(Xnum[te])))
        Xtr = hstack(bt).tocsr() if len(bt) > 1 else bt[0]
        Xte = hstack(be).tocsr() if len(be) > 1 else be[0]
        clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=3000, solver="liblinear").fit(Xtr, y[tr])
        oos[te] = clf.predict_proba(Xte)[:, 1]; ins[tr] += clf.predict_proba(Xtr)[:, 1]; cnt[tr] += 1
        print(f"  [{kind}] fold {kf} done", file=sys.stderr)
    return oos, ins / np.maximum(cnt, 1)


def oldloss_at_recall(p, t):
    pos = np.sort(p[y == 1]); k = int(np.ceil((1 - t) * len(pos))); thr = pos[min(k, len(pos) - 1)]
    fl = p >= thr; return fl[y == 1].mean(), fl[y == 0].mean()
def max_recall_at_oldloss(p, b):
    neg = np.sort(p[y == 0]); thr = neg[min(int(np.ceil((1 - b) * len(neg))), len(neg) - 1)]
    return (p > thr)[y == 1].mean()


print(f"{'model':<10}{'old-loss@99%':>14}{'old-loss@99.5%':>16}{'max-rec@10%OL':>16}{'overfit gap':>14}")
out = {}
for kind in ("signals", "text", "combined"):
    oos, ins = oos_scores(kind)
    _, ol99 = oldloss_at_recall(oos, 0.99); _, ol995 = oldloss_at_recall(oos, 0.995)
    mr = max_recall_at_oldloss(oos, 0.10)
    _, oi = oldloss_at_recall(ins, 0.995)
    print(f"{kind:<10}{ol99*100:>12.2f}%{ol995*100:>14.2f}%{mr*100:>14.1f}%{(ol995-oi)*100:>11.2f}pt")
    out[kind] = {"oldloss_99": ol99, "oldloss_995": ol995, "max_rec_10": mr, "overfit_gap": ol995 - oi}
json.dump({"n": len(ALL), "n_modern": int(y.sum()), "models": out},
          open(HERE / "classify_clean_results.json", "w"), indent=1)
print("\nsaved -> classify_clean_results.json")
