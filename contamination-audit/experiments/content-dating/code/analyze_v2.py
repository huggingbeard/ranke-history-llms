"""Analyze flotilla judgments -> precision/recall/F1 surfaces + cost projection.

v2 (2026-07-22) — extended rewrite of analyze.py, run on the office 4090 box.
Files used:
  judgments: D:\\...\\content-dating\\outputs\\judg_<model>.parquet  (one per model)
  timings:   D:\\...\\content-dating\\outputs\\timings_<model>.json  (passages/s per model)
  output:    D:\\...\\content-dating\\outputs\\results.csv  (+ copy into the repo)

Changes vs v1:
  - reports each model's unparseable-answer rate (null preds count as "no flag", which depresses
    recall — a model that cannot answer cannot detect; the rate must be visible, not silent).
  - joins measured throughput -> estimated 4090 GPU-days to scan 2M books for every config
    (cost = N passages x length-scaled per-passage time).
  - adds book-level AUC over the k-of-16 vote-threshold sweep (N=16 per config).
  - any-positive aggregation over the first N passages, as in v1 (recall-max by design).
"""
import os, glob, json, re
import pandas as pd

DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
OUT = os.path.join(DROPBOX, "outputs")
REPO = os.path.dirname(os.path.abspath(__file__))
NS = [1, 2, 4, 8, 16]
BOOKS_2M = 2_000_000

def auc_from_votes(g, n=16):
    """Book-level ROC-AUC sweeping the vote threshold k=0..n (votes = # of AFTER preds in first n)."""
    pts = []
    P = int((g.label == 1).sum()); Ng = int((g.label == 0).sum())
    if P == 0 or Ng == 0: return None
    for k in range(0, n + 2):
        flag = g.votes >= k
        tpr = float(((g.label == 1) & flag).sum()) / P
        fpr = float(((g.label == 0) & flag).sum()) / Ng
        pts.append((fpr, tpr))
    pts = sorted(set(pts))
    auc = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        auc += (x2 - x1) * (y1 + y2) / 2
    return round(auc, 3)

def main():
    timings = {}
    for f in glob.glob(os.path.join(OUT, "timings_*.json")):
        t = json.load(open(f))
        timings[t["model"]] = t
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "judg_*.parquet"))):
        model = re.sub(r".*judg_", "", f).replace(".parquet", "").replace("__", "/")
        d = pd.read_parquet(f)
        null_rate = float(d.pred.isna().mean())
        pps = timings.get(model, {}).get("passages_per_second")
        mean_L = d.length.mean()
        for L in sorted(d.length.unique()):
            for strat in sorted(d.strategy.unique()):
                sub = d[(d.length == L) & (d.strategy == strat)]
                votes = (sub[sub.idx < 16].assign(v=lambda x: (x.pred == 1).astype(int))
                         .groupby("id").agg(label=("label", "first"), votes=("v", "sum")).reset_index())
                auc = auc_from_votes(votes)
                for N in NS:
                    g = (sub[sub.idx < N].groupby("id")
                         .agg(label=("label", "first"), flag=("pred", lambda x: int((x == 1).any()))).reset_index())
                    t = g.label.astype(bool).values; p = g.flag.astype(bool).values
                    tp = int((t & p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum())
                    P = tp / (tp + fp) if tp + fp else 1.0
                    R = tp / (tp + fn) if tp + fn else 1.0
                    F = 2 * P * R / (P + R) if P + R else 0.0
                    gpu_days_2m = None
                    if pps:
                        sec_per_passage = (1.0 / pps) * (L / mean_L)   # length-scaled from the mixed-length run
                        gpu_days_2m = round(BOOKS_2M * N * sec_per_passage / 86400, 2)
                    rows.append(dict(model=model, length=L, strategy=strat, N=N,
                                     precision=round(P, 3), recall=round(R, 3), f1=round(F, 3),
                                     auc16=auc, n_books=len(g), null_rate=round(null_rate, 4),
                                     gpu_days_2M_books=gpu_days_2m))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "results.csv"), index=False)
    res.to_csv(os.path.join(REPO, "results.csv"), index=False)
    print(f"{res.model.nunique()} models, {len(res)} configs -> results.csv")
    print("\nnull (unparseable) rate per model:")
    print(res.groupby("model").null_rate.first().round(4).to_string())
    ok = res[res.recall >= 0.95]
    if len(ok):
        ok = ok.sort_values(["gpu_days_2M_books", "N", "length"], na_position="last")
        print("\nconfigs with recall >= 0.95, cheapest (GPU-days for 2M books) first:")
        print(ok.head(20).to_string(index=False))
    else:
        print("\nno config reached recall 0.95 — inspect surfaces")

if __name__ == "__main__":
    main()
