"""Stratified analysis of flotilla judgments -> per-stratum recall surfaces + cost projection.

v3 (2026-07-23) — supersedes analyze_v2 after the PI's stratum decisions:
  headline recall  = contam_modern   (239 gold-audited modern-body contaminants from our own corpus)
  hardest cases    = contam_gotthrough (3 books the production pipeline missed; reported as raw counts)
  false-flag floor = contam_reprint_old_body (80 modern printings of OLD content; content-reading
                     should NOT flag these — their flag rate is reported, lower = better)
  diagnostic       = stratum A (1,000 post-1947 institutional-books positives, gov-doc-heavy;
                     PI: "a gov doc classifier by another name" — kept for easy-vs-hard contrast)
  precision side   = 918 certified-clean negatives.
Files used:
  outputs\\judg_*.parquet (+ delta files judg_*.contamC/.retry — merged by cleaned model name)
  outputs\\timings_*.json  -> measured passages/s -> 4090 GPU-days for a 2M-book scan
  repo manifests: manifest_negatives/positives/contamC.csv (id -> stratum)
Output: outputs\\results.csv + repo results.csv; console headline = cheapest config with
        recall(contam_modern) >= 0.95, ranked by projected GPU-days.
"""
import os, glob, json, re
import pandas as pd

DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
OUT = os.path.join(DROPBOX, "outputs")
REPO = os.path.dirname(os.path.abspath(__file__))
NS = [1, 2, 4, 8, 16]
BOOKS_2M = 2_000_000

NICE = {
    "shieldstar/Qwen3.6-35B-A3B-int4-AutoRound-EC": "Qwen3.6-35B-A3B-int4",
    "LeaderboardModel1/Qwen3.6-35B-A3B-autoround-W4A16": "Qwen3.6-35B-A3B-int4",
    "unsloth/Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "unsloth/Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "Qwen36-35B-A3B-Q3-llamacpp": "Qwen3.6-35B-A3B-Q3 (llama.cpp)",
}

def clean_model(fname):
    m = re.sub(r".*judg_", "", fname).replace(".parquet", "")
    m = re.sub(r"\.(contamC|retry|w4a16|pos3046)$", "", m)  # delta/retry suffixes
    m = m.replace("__", "/")
    return NICE.get(m, m.replace("Qwen/", "").replace("microsoft/", "").replace("openai/", ""))

def strata_map():
    neg = pd.read_csv(os.path.join(REPO, "manifest_negatives.csv"), dtype={"id": str})
    pos = pd.read_csv(os.path.join(REPO, "manifest_positives.csv"), dtype={"id": str})
    c = pd.read_csv(os.path.join(REPO, "manifest_contamC.csv"), dtype={"id": str})
    m = {}
    for i in neg.id: m[i] = "negatives"
    for i in pos.id: m[i] = "A_post1947_govheavy"
    for i, cls in zip(c.id, c.cls): m[i] = cls
    p3046 = os.path.join(REPO, "manifest_pos3046.csv")
    if os.path.exists(p3046):
        for i in pd.read_csv(p3046, dtype={"id": str}).id: m[i] = "B_1930_46_ordinary"
    return m

def main():
    smap = strata_map()
    timings = {}
    for f in glob.glob(os.path.join(OUT, "timings_*.json")):
        t = json.load(open(f))
        timings[clean_model("judg_" + t["model"].replace("/", "__"))] = t

    by_model = {}
    for f in sorted(glob.glob(os.path.join(OUT, "judg_*.parquet"))):
        m = clean_model(os.path.basename(f))
        by_model.setdefault(m, []).append(f)

    rows = []
    for model, files in by_model.items():
        d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        d = d.drop_duplicates(["id", "length", "strategy", "idx"])   # delta overlap guard
        d["stratum"] = d.id.map(smap)
        null_rate = float(d.pred.isna().mean())
        pps = timings.get(model, {}).get("passages_per_second")
        mean_L = d.length.mean()
        n_contamC = d[d.stratum.isin(["contam_modern", "contam_reprint_old_body", "contam_gotthrough"])].id.nunique()
        for L in sorted(d.length.unique()):
            for strat in sorted(d.strategy.unique()):
                sub = d[(d.length == L) & (d.strategy == strat)]
                for N in NS:
                  for k in ([1] if N == 1 else [1, 2]):     # any-positive (k=1) and 2-of-N vote
                    g = (sub[sub.idx < N].groupby("id")
                         .agg(stratum=("stratum", "first"),
                              flag=("pred", lambda x: int((x == 1).sum() >= k))).reset_index())
                    def rate(name):
                        s = g[g.stratum == name]
                        return (float(s.flag.mean()) if len(s) else None), len(s)
                    fp_rate, n_neg = rate("negatives")
                    rec_C, n_C = rate("contam_modern")
                    rec_A, n_A = rate("A_post1947_govheavy")
                    rec_B, n_B = rate("B_1930_46_ordinary")
                    rep_rate, n_rep = rate("contam_reprint_old_body")
                    gt = g[g.stratum == "contam_gotthrough"]
                    prec = None
                    if rec_C is not None and fp_rate is not None:
                        tp = rec_C * n_C; fp = fp_rate * n_neg
                        prec = tp / (tp + fp) if tp + fp else 1.0
                    gpu_days = None
                    if pps:
                        gpu_days = round(BOOKS_2M * N * (1.0 / pps) * (L / mean_L) / 86400, 2)
                    rows.append(dict(model=model, length=L, strategy=strat, N=N, k_of_N=k,
                                     recall_contam_modern=None if rec_C is None else round(rec_C, 3),
                                     precision_vs_negatives=None if prec is None else round(prec, 3),
                                     fp_rate_negatives=None if fp_rate is None else round(fp_rate, 3),
                                     reprint_flag_rate=None if rep_rate is None else round(rep_rate, 3),
                                     gotthrough_flagged=f"{int(gt.flag.sum())}/{len(gt)}" if len(gt) else "0/0",
                                     recall_A_govdocs=None if rec_A is None else round(rec_A, 3),
                                     recall_B_1930s=None if rec_B is None else round(rec_B, 3),
                                     n_contam=n_C, n_neg=n_neg, n_reprint=n_rep, n_A=n_A, n_B=n_B,
                                     null_rate=round(null_rate, 4), gpu_days_2M_books=gpu_days,
                                     books_with_C_coverage=n_contamC))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "results.csv"), index=False)
    res.to_csv(os.path.join(REPO, "results.csv"), index=False)
    print(f"{res.model.nunique()} models x {len(res)} configs -> results.csv")
    print("\nC-coverage (books) and null rate per model:")
    print(res.groupby("model")[["books_with_C_coverage", "null_rate"]].first().to_string())
    cols = ["model", "length", "strategy", "N", "k_of_N", "recall_contam_modern",
            "recall_B_1930s", "fp_rate_negatives", "precision_vs_negatives", "reprint_flag_rate",
            "gotthrough_flagged", "recall_A_govdocs", "gpu_days_2M_books"]
    # an always-flag model trivially "wins" on recall; the headline requires a sane FP bound
    for fp_max in (0.05, 0.10):
        ok = res[(res.recall_contam_modern.notna()) & (res.recall_contam_modern >= 0.95)
                 & (res.fp_rate_negatives <= fp_max)]
        ok = ok.sort_values(["gpu_days_2M_books", "N", "length"], na_position="last")
        print(f"\nconfigs with recall(contam_modern) >= 0.95 AND FP rate on negatives <= {fp_max:.0%}, cheapest first:")
        print(ok[cols].head(15).to_string(index=False) if len(ok) else "  none")

if __name__ == "__main__":
    main()
