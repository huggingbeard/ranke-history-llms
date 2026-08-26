"""Judge-ensemble analysis: combine multiple models' verdicts on the SAME passages (zero GPU cost).

v1 (2026-07-23), office 4090 box, PI request ("experiment with ensembles of judges").
Every flotilla model judged the identical passage set, so ensembles are computed from the stored
per-passage verdicts in outputs\\judg_*.parquet — no new inference.

Two ensemble places:
  passage-level: a passage is AFTER if >= m of the M member models say AFTER; then book k-of-N as usual.
  book-level:    each member flags a book via its own k-of-N; the ensemble flags if >= m members flag.
Members: the four discriminating models (gpt-oss-20b, Qwen2.5-7B, Qwen2.5-14B-AWQ, Phi-3.5-mini).
Cost of a production ensemble = SUM of member GPU-days (all members read every passage).
Output: outputs\\ensemble_results.csv + console frontier under FP bounds.
"""
import os, glob, json, re
from itertools import combinations
import pandas as pd

DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
OUT = os.path.join(DROPBOX, "outputs")
REPO = os.path.dirname(os.path.abspath(__file__))
NS = [1, 2, 4, 8, 16]
BOOKS_2M = 2_000_000
MEMBERS = {
    "gptoss": "judg_openai__gpt-oss-20b",
    "q7b": "judg_Qwen__Qwen2.5-7B-Instruct",
    "q14b": "judg_Qwen__Qwen2.5-14B-Instruct-AWQ",
    "phi": "judg_microsoft__Phi-3.5-mini-instruct",
}

def strata_map():
    m = {}
    for fname, tag in [("manifest_negatives.csv", None), ("manifest_positives.csv", None),
                       ("manifest_contamC.csv", "cls")]:
        df = pd.read_csv(os.path.join(REPO, fname), dtype={"id": str})
        if tag:
            for i, c in zip(df.id, df[tag]): m[i] = c
        else:
            lab = "negatives" if "negatives" in fname else "A_post1947_govheavy"
            for i in df.id: m[i] = lab
    p3046 = os.path.join(REPO, "manifest_pos3046.csv")
    if os.path.exists(p3046):
        for i in pd.read_csv(p3046, dtype={"id": str}).id: m[i] = "B_1930_46_ordinary"
    return m

def load_matrix():
    """Wide per-passage matrix: index (id,length,strategy,idx), one 0/1 column per member."""
    base = None
    for tag, stem in MEMBERS.items():
        files = sorted(glob.glob(os.path.join(OUT, stem + "*.parquet")))
        d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        d = d.drop_duplicates(["id", "length", "strategy", "idx"])
        d = d.rename(columns={"pred": tag})[["id", "length", "strategy", "idx", "label", tag]]
        base = d if base is None else base.merge(d.drop(columns=["label"]),
                                                 on=["id", "length", "strategy", "idx"], how="inner")
    print(f"matrix: {len(base)} passages with all {len(MEMBERS)} members", flush=True)
    return base

def gpu_cost(models_pps, members, N, L, mean_L):
    s = 0.0
    for m in members:
        pps = models_pps.get(m)
        if not pps: return None
        s += BOOKS_2M * N * (1.0 / pps) * (L / mean_L) / 86400
    return round(s, 2)

def main():
    smap = strata_map()
    mat = load_matrix()
    mat["stratum"] = mat.id.map(smap)
    mean_L = mat.length.mean()
    pps = {}
    for tag, stem in MEMBERS.items():
        tf = os.path.join(OUT, stem.replace("judg_", "timings_") + ".json")
        if os.path.exists(tf): pps[tag] = json.load(open(tf))["passages_per_second"]

    member_tags = list(MEMBERS)
    ensembles = [list(c) for r in (2, 3, 4) for c in combinations(member_tags, r)]
    rows = []
    for ens in ensembles:
        M = len(ens)
        votes = mat[ens].sum(axis=1, skipna=True)          # nulls count as BEFORE (no flag)
        for m_req in range(1, M + 1):
            mat["_pflag"] = (votes >= m_req).astype(int)
            for L in sorted(mat.length.unique()):
                for strat in ("random", "grep"):
                    sub = mat[(mat.length == L) & (mat.strategy == strat)]
                    for N in NS:
                        for k in ([1] if N == 1 else [1, 2]):
                            g = (sub[sub.idx < N].groupby("id")
                                 .agg(stratum=("stratum", "first"),
                                      flag=("_pflag", lambda x, kk=k: int(x.sum() >= kk))).reset_index())
                            def rate(name):
                                s = g[g.stratum == name]
                                return (float(s.flag.mean()) if len(s) else None), len(s)
                            fp, n_neg = rate("negatives")
                            rc, n_c = rate("contam_modern")
                            rb, n_b = rate("B_1930_46_ordinary")
                            rows.append(dict(ensemble="+".join(ens), m_of_M=f"{m_req}of{M}",
                                             length=L, strategy=strat, N=N, k_of_N=k,
                                             recall_contam_modern=None if rc is None else round(rc, 3),
                                             fp_rate_negatives=None if fp is None else round(fp, 3),
                                             recall_B=None if rb is None else round(rb, 3),
                                             n_contam=n_c, n_neg=n_neg, n_B=n_b,
                                             gpu_days_2M_books=gpu_cost(pps, ens, N, L, mean_L)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "ensemble_results.csv"), index=False)
    print(f"{len(res)} ensemble configs -> ensemble_results.csv")
    for fpmax in (0.05, 0.10):
        ok = res[(res.recall_contam_modern.notna()) & (res.fp_rate_negatives <= fpmax)]
        ok = ok.sort_values("recall_contam_modern", ascending=False)
        print(f"\nbest ensembles at FP <= {fpmax:.0%} (single-model best: 0.773 @5% / 0.857 @10%):")
        print(ok[["ensemble", "m_of_M", "length", "strategy", "N", "k_of_N",
                  "recall_contam_modern", "fp_rate_negatives", "gpu_days_2M_books"]].head(12).to_string(index=False))

if __name__ == "__main__":
    main()
