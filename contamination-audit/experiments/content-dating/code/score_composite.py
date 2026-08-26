"""Composite two-stage pipeline evaluation: front-matter dater -> content reader, sequentially.

v1 (2026-07-23). Panel (labels by construction/certification):
  negatives 918 (certified clean) | C modern 239 | reprints 80 (policy stratum, gpt-oss editorial
  pass pending GPU) | got-throughs 3 | B 600 (1930-46 ordinary, honest metadata) | plants 155
  (realism-gated metadata-corrupted copies of B/C books).
Stage 1 = freeze_pipeline v2: remove iff max(meta_year, stage2_docyear, imprint_year(text)) > 1929.
  plants: fake pre-1929 meta year; stage2_docyear scored under two bounds — FOOLED (= fake year;
  worst case) and CORRECT (= true year; stage 1 catches) — until the real 14B doc-dater can run.
  B books: no corpus stage2_docyear (HF pull) -> stage 1 sees metadata + imprint (both honest).
Stage 2 (locked with PI): flag iff 35B-BF16 >=3 of 8 AFTER votes OR gpt-oss >=3 of 8, on
  8 x 2000-char random passages; applied to stage-1 SURVIVORS only. Plants inherit the parent
  book's passage verdicts (body text unchanged by corruption).
Outputs: composite_results.csv (per-book stages+verdicts) + per-stratum table with recall,
  stage-2 marginal value, corpus loss.
"""
import os, sys
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, r"C:\Users\hvoth\Documents\GitHub\ranke-edition-dating")
from freeze_pipeline import imprint_year

REPO = os.path.dirname(os.path.abspath(__file__))
DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
OUT = os.path.join(DROPBOX, "outputs")
CORPUS = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\temp-guohui\decontamination\datasets\history-llms\sample-10gb\data"
I = os.path.join(DROPBOX, "intermediates")
CUTOFF = 1929
K35, KGO = 3, 3

def corpus_texts(df_ids_shards):
    txt = {}
    for sh, g in df_ids_shards.groupby("shard"):
        t = pq.read_table(os.path.join(CORPUS, sh), columns=["id", "text"])
        t = t.filter(pc.is_in(t.column("id"), value_set=pa.array(list(g.id))))
        for i, tx in zip(t.column("id").to_pylist(), t.column("text").to_pylist()):
            txt[i] = tx or ""
    return txt

def stage2_votes():
    q35 = pd.read_parquet(os.path.join(OUT, "judg_Qwen36-35B-A3B-BF16-A100.parquet"))
    go = pd.concat([pd.read_parquet(os.path.join(OUT, f)) for f in
                    ("judg_openai__gpt-oss-20b.parquet", "judg_openai__gpt-oss-20b.pos3046.parquet")],
                   ignore_index=True).drop_duplicates(["id", "length", "strategy", "idx"])
    def votes(d):
        s = d[(d.length == 2000) & (d.strategy == "random") & (d.idx < 8)]
        return s.groupby("id").agg(v=("pred", lambda x: int((x == 1).sum()))).v
    v35, vgo = votes(q35), votes(go)
    flag = ((v35.reindex(v35.index.union(vgo.index)).fillna(0) >= K35) |
            (vgo.reindex(v35.index.union(vgo.index)).fillna(0) >= KGO)).astype(int)
    return flag.to_dict()

def main():
    cm = pd.read_parquet(os.path.join(I, "corpus_meta.parquet")).set_index("id")
    s2flag = stage2_votes()
    neg = pd.read_csv(os.path.join(REPO, "manifest_negatives.csv"), dtype={"id": str})
    mc = pd.read_csv(os.path.join(REPO, "manifest_contamC.csv"), dtype={"id": str})
    b = pd.read_csv(os.path.join(REPO, "manifest_pos3046.csv"), dtype={"id": str})
    plants = pd.read_csv(os.path.join(REPO, "manifest_plants.csv"), dtype=str)

    rows = []
    corpus_panel = pd.concat([
        neg.assign(stratum="negatives", label=0),
        mc.assign(stratum=mc.cls, label=1),
    ], ignore_index=True)
    txt = corpus_texts(corpus_panel[["id", "shard"]])
    for r in corpus_panel.itertuples():
        tx = txt.get(r.id, "")
        meta = cm.loc[r.id] if r.id in cm.index else None
        my = float(meta.year) if meta is not None else None
        s2y = float(meta.stage2_docyear) if meta is not None and pd.notna(meta.stage2_docyear) else None
        iy = imprint_year(tx) if tx else None
        vals = [v for v in (my, s2y, iy) if v is not None]
        st1 = int(bool(vals) and max(vals) > CUTOFF)
        st2 = s2flag.get(r.id)
        rows.append(dict(id=r.id, stratum=r.stratum, label=r.label, scenario="actual",
                         stage1=st1, stage2=None if st2 is None else int(st2),
                         removed=int(st1 or (st2 == 1))))
    for r in b.itertuples():
        p = os.path.join(DROPBOX, "positives_text", r.id + ".txt")
        tx = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
        iy = imprint_year(tx) if tx else None
        vals = [v for v in (float(r.meta_year), iy) if v is not None]
        st1 = int(max(vals) > CUTOFF)
        st2 = s2flag.get(r.id)
        rows.append(dict(id=r.id, stratum="B_honest_metadata", label=1, scenario="actual",
                         stage1=st1, stage2=None if st2 is None else int(st2),
                         removed=int(st1 or (st2 == 1))))
    for r in plants.itertuples():
        p = os.path.join(DROPBOX, "plants_text", r.plant_id + ".txt")
        tx = open(p, encoding="utf-8").read()
        iy = imprint_year(tx)
        st2 = s2flag.get(r.parent_id)          # body unchanged -> parent verdict
        for scen, s2y in (("s2_fooled", float(r.fake_meta_year)), ("s2_correct", float(r.true_year))):
            vals = [v for v in (float(r.fake_meta_year), s2y, iy) if v is not None]
            st1 = int(max(vals) > CUTOFF)
            rows.append(dict(id=r.plant_id, stratum=f"plants_{r.parent_cls}", label=1, scenario=scen,
                             stage1=st1, stage2=None if st2 is None else int(st2),
                             removed=int(st1 or (st2 == 1))))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "composite_results.csv"), index=False)
    d.to_csv(os.path.join(REPO, "composite_results.csv"), index=False)

    print("=== COMPOSITE PIPELINE (stage1 freeze-v2 -> stage2 35B|gptoss OR 3/3, cutoff 1929) ===")
    for scen in ("actual", "s2_fooled", "s2_correct"):
        sub = d[d.scenario == scen]
        if not len(sub): continue
        print(f"--- scenario: {scen}")
        for st, g in sub.groupby("stratum"):
            n = len(g)
            s1r = g.stage1.mean()
            comp = g.removed.mean()
            s2cov = g.stage2.notna().mean()
            tag = "corpus loss" if (g.label == 0).all() else "recall"
            print(f"  {st:28s} n={n:4d} | stage1 {tag}: {s1r:.3f} | composite {tag}: {comp:.3f} "
                  f"| stage2 marginal: {comp - s1r:+.3f} | s2 coverage {s2cov:.0%}")

if __name__ == "__main__":
    main()
