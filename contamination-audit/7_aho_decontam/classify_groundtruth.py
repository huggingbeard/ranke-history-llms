"""
classify_groundtruth.py — the ground-truth classification experiment.

GROUND TRUTH (guaranteed by OpenLibrary first_publish_year, not the dataset's own date):
  POST = books OpenLibrary dates >= 1946   (genuinely post-1945 content)
  PRE  = books OpenLibrary dates <= 1945    (genuinely pre-1945 content)
Both drawn from the SAME corpus (LoC-PD-Books, same OCR pipeline) so the classifier learns the
era, not the source. Balanced N per class; a fixed-seed split into TRAIN and a disjoint HOLD-OUT.

FEATURES (fit on TRAIN only -> no leakage):
  word    TF-IDF word 1-2 grams
  char    TF-IDF char_wb 3-5 grams (OCR-robust)
  signals [yc_raw, yc_ctx, kw_raw, kw_dep, english_ratio, nonlatin_ratio]  (the cheap tripwires)
MODELS: word-only, char-only, signals-only, and STACKED (word+char+signals) — logistic regression.

REPORTED ON THE HOLD-OUT: AUC-ROC, confusion matrix, accuracy, precision, recall, F1 (at 0.5),
and — the deployment view — the old-book loss when the threshold is set to catch >=99.5% of POST.
"""
import json, sys, random
import numpy as np
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

HERE = Path(__file__).resolve().parent
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
random.seed(42)
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
POST, PRE = [], []
for i, v in lab.items():
    if v["ol_year"] is None or i not in BY: continue
    (POST if v["ol_year"] >= 1946 else PRE).append(BY[i])
random.shuffle(POST); random.shuffle(PRE)
N = min(len(POST), len(PRE))
POST, PRE = POST[:N], PRE[:N]
half = N // 2
train_rows = POST[:half] + PRE[:half]
hold_rows = POST[half:2*half] + PRE[half:2*half]
random.shuffle(train_rows); random.shuffle(hold_rows)
def label(r): return 1 if (lab[r["id"]]["ol_year"] >= 1946) else 0
ytr = np.array([label(r) for r in train_rows]); yte = np.array([label(r) for r in hold_rows])
print(f"GROUND TRUTH (OpenLibrary-confirmed): {len(POST)} POST + {len(PRE)} PRE available")
print(f"TRAIN {len(train_rows)} ({int(ytr.sum())} post / {int((1-ytr).sum())} pre)  |  "
      f"HOLD-OUT {len(hold_rows)} ({int(yte.sum())} post / {int((1-yte).sum())} pre)  [disjoint]\n")

def signals(r):
    nl, en = ratios(r["tx"])
    return [r["yc_raw"], r["yc_ctx"], len(r["mk_raw"]), len(r["mk_dep"]), en, nl]
Str = np.array([signals(r) for r in train_rows], float); Ste = np.array([signals(r) for r in hold_rows], float)
txtr = [r["tx"] for r in train_rows]; txte = [r["tx"] for r in hold_rows]

vw = TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=40000, ngram_range=(1, 2),
                     token_pattern=r"(?u)\b\w+\b").fit(txtr)
vc = TfidfVectorizer(sublinear_tf=True, min_df=5, max_features=40000, analyzer="char_wb",
                     ngram_range=(3, 5)).fit(txtr)
sc = StandardScaler().fit(Str)
BW_tr, BW_te = vw.transform(txtr), vw.transform(txte)
BC_tr, BC_te = vc.transform(txtr), vc.transform(txte)
BS_tr, BS_te = csr_matrix(sc.transform(Str)), csr_matrix(sc.transform(Ste))
BLOCKS = {"word": (BW_tr, BW_te), "char": (BC_tr, BC_te), "signals": (BS_tr, BS_te)}
MODELS = {"word-only": ["word"], "char-only": ["char"], "signals-only": ["signals"],
          "STACKED(word+char+signals)": ["word", "char", "signals"]}


def oldloss_at_recall(score, y, target=0.995):
    pos = np.sort(score[y == 1]); k = int(np.ceil((1 - target) * len(pos)))
    thr = pos[min(k, len(pos) - 1)]; fl = score >= thr
    return fl[y == 0].mean(), fl[y == 1].mean(), thr

print(f"{'model':<28}{'AUC':>7}{'acc':>7}{'prec':>7}{'recall':>8}{'F1':>7}   confusion[tn fp / fn tp]   old-loss@99.5%rec")
out = {}
for name, blocks in MODELS.items():
    Xtr = hstack([BLOCKS[b][0] for b in blocks]).tocsr() if len(blocks) > 1 else BLOCKS[blocks[0]][0]
    Xte = hstack([BLOCKS[b][1] for b in blocks]).tocsr() if len(blocks) > 1 else BLOCKS[blocks[0]][1]
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=3000, solver="liblinear").fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]; pred = (p >= 0.5).astype(int)
    auc = roc_auc_score(yte, p); tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    ol, rc, _ = oldloss_at_recall(p, yte, 0.995)
    print(f"{name:<28}{auc:>7.3f}{accuracy_score(yte,pred):>7.3f}{precision_score(yte,pred):>7.3f}"
          f"{recall_score(yte,pred):>8.3f}{f1_score(yte,pred):>7.3f}   [{tn:>4} {fp:>3} / {fn:>3} {tp:>4}]"
          f"        {ol*100:>6.1f}%")
    out[name] = {"auc": auc, "acc": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
                 "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
                 "confusion": [int(tn), int(fp), int(fn), int(tp)], "oldloss_at_995recall": float(ol)}
json.dump({"n_post": len(POST), "n_pre": len(PRE), "n_train": len(train_rows), "n_holdout": len(hold_rows),
           "models": out}, open(HERE / "classify_groundtruth_results.json", "w"), indent=1)
print("\n(confusion reads: tn=pre kept, fp=pre wrongly flagged, fn=POST MISSED (bad), tp=post caught)")
print("saved -> classify_groundtruth_results.json")
