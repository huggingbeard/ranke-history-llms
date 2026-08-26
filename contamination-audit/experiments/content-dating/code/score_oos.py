"""OOS full-stat scoring vs PREREGISTRATION_OOS.md (predictions P1-P5).

v1 (2026-07-24). Frozen pipeline: stage1 freeze-v2 (hard floor + empirical s2 model at fooled
rates {0.02, 0.069, 0.12}); stage2 = 35B-BF16 >=3/8 OR gpt-oss >=3/8, 8x2000 random.
Panel: manifest_oos_negatives (365 certified), manifest_oos_moderns (250), manifest_oos_plants (127).
Stats: Clopper-Pearson exact 95% CIs; exact binomial tests vs pre-registered points (two-sided);
2,000-rep book bootstrap for composites. All counts printed raw.
"""
import os, sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from scipy.stats import beta, binomtest

sys.path.insert(0, r"C:\Users\hvoth\Documents\GitHub\ranke-edition-dating")
from freeze_pipeline import imprint_year

REPO = os.path.dirname(os.path.abspath(__file__))
DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
OUT = os.path.join(DROPBOX, "outputs")
CORPUS = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\temp-guohui\decontamination\datasets\history-llms\sample-10gb\data"
CUTOFF = 1929
FOOLED = [0.02, 0.069, 0.12]
RNG = np.random.default_rng(1929)

def cp_ci(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi

def fmt(k, n, pred=None):
    p = k / n
    lo, hi = cp_ci(k, n)
    s = f"{p:.3f} [{lo:.3f}, {hi:.3f}] ({k}/{n})"
    if pred is not None:
        pv = binomtest(k, n, pred).pvalue
        s += f"  vs pred {pred}: p={pv:.3f}"
    return s

def stage2_flags():
    q35 = pd.read_parquet(os.path.join(OUT, "judg_Qwen36-35B-A3B-BF16-A100.oos.parquet"))
    go = pd.read_parquet(os.path.join(OUT, "judg_openai__gpt-oss-20b.oos.parquet"))
    def votes(d):
        s = d[(d.length == 2000) & (d.strategy == "random") & (d.idx < 8)]
        return s.groupby("id").agg(v=("pred", lambda x: int((x == 1).sum()))).v
    v35, vgo = votes(q35), votes(go)
    idx = v35.index.union(vgo.index)
    return ((v35.reindex(idx).fillna(0) >= 3) | (vgo.reindex(idx).fillna(0) >= 3)).astype(int).to_dict()

def main():
    s2 = stage2_flags()
    neg = pd.read_csv(os.path.join(REPO, "manifest_oos_negatives.csv"), dtype={"id": str})
    mod = pd.read_csv(os.path.join(REPO, "manifest_oos_moderns.csv"), dtype={"id": str})
    plants = pd.read_csv(os.path.join(REPO, "manifest_oos_plants.csv"), dtype=str)

    # stage 1 on negatives: meta + s2(corpus) + imprint(text)
    txt = {}
    for sh, g in neg.groupby("shard"):
        t = pq.read_table(os.path.join(CORPUS, sh), columns=["id", "text"])
        t = t.filter(pc.is_in(t.column("id"), value_set=pa.array(list(g.id))))
        for i, tx in zip(t.column("id").to_pylist(), t.column("text").to_pylist()): txt[i] = tx or ""
    neg["st1"] = [int(max(v for v in (float(r.meta_year), float(r.stage2_docyear),
                                      imprint_year(txt.get(r.id, "")) or 0)) > CUTOFF) for r in neg.itertuples()]
    neg["st2"] = neg.id.map(s2)
    neg["removed"] = ((neg.st1 == 1) | (neg.st2 == 1)).astype(int)

    # honest moderns: meta + imprint
    def hf_imprint(bid):
        p = os.path.join(DROPBOX, "positives_text", bid + ".txt")
        return imprint_year(open(p, encoding="utf-8").read()) if os.path.exists(p) else None
    mod["st1"] = [int(max(float(r.meta_year), hf_imprint(r.id) or 0) > CUTOFF) for r in mod.itertuples()]
    mod["st2"] = mod.id.map(s2)
    mod["removed"] = ((mod.st1 == 1) | (mod.st2 == 1)).astype(int)

    # plants: fake meta + plant imprint; s2 model applied analytically
    plants["iy"] = [None if r.imprint_after in ("", "nan", None) else float(r.imprint_after)
                    for r in plants.itertuples()]
    plants["st1_floor"] = [int(max(float(r.fake_meta_year), (r.iy or 0)) > CUTOFF) for r in plants.itertuples()]
    plants["st2"] = plants.plant_id.map(s2)

    print("=" * 70)
    print("OOS VALIDATION vs PREREGISTRATION_OOS.md — all CIs Clopper-Pearson 95%")
    print("=" * 70)
    n = len(mod)
    k = int(mod.removed.sum())
    print(f"\nP1 honest-modern composite recall (pred 0.998, band >=0.98):\n   {fmt(k, n, 0.998)}")
    k1 = int(mod.st1.sum())
    print(f"P5 stage-1-only on honest moderns: {fmt(k1, n)}  | stage-2 marginal: {(k - k1) / n:+.3f} (band <=0.02)")

    pl = plants[plants.st2.notna()]
    floor_pool = pl[pl.st1_floor == 0]
    k2 = int(floor_pool.st2.sum()); n2 = len(floor_pool)
    print(f"\nP2 plant stage-2-only rescue, hard floor (pred 0.93, band [0.86,0.97]):\n   {fmt(k2, n2, 0.93)}")
    for fr in FOOLED:
        comp = (pl.st1_floor + (1 - pl.st1_floor) * ((1 - fr) + fr * pl.st2)).mean()
        tag = "  <- P3 (pred 0.995, band [0.98,1.00])" if abs(fr - 0.069) < 1e-9 else ""
        print(f"P3 plant composite @ fooled={fr:.3f}: {comp:.4f}{tag}")
    boots = []
    arr = pl[["st1_floor", "st2"]].to_numpy()
    for _ in range(2000):
        s = arr[RNG.integers(0, len(arr), len(arr))]
        boots.append((s[:, 0] + (1 - s[:, 0]) * ((1 - 0.069) + 0.069 * s[:, 1])).mean())
    print(f"   P3 bootstrap 95% CI @0.069: [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]")

    k4 = int(neg.removed.sum()); n4 = len(neg)
    print(f"\nP4 corpus loss on fresh certified negatives (pred 0.100, band [0.06,0.14]):\n   {fmt(k4, n4, 0.100)}")
    k4a = int(neg.st1.sum())
    print(f"   stage-1-only loss: {fmt(k4a, n4)} (panel negatives NOT screen-selected this time on imprint/s2? they were — same five screens; stage1 loss ~0 expected by construction)")
    print(f"   stage-2 coverage: neg {neg.st2.notna().mean():.0%}, mod {mod.st2.notna().mean():.0%}, plants {pl.st2.notna().mean() if len(pl) else 0:.0%}")

    d = pd.concat([neg.assign(stratum="oos_negatives"), mod.assign(stratum="oos_honest_moderns")])
    d.to_csv(os.path.join(OUT, "oos_results.csv"), index=False)
    plants.to_csv(os.path.join(OUT, "oos_plants_results.csv"), index=False)
    print("\nrows -> oos_results.csv / oos_plants_results.csv")

if __name__ == "__main__":
    main()
