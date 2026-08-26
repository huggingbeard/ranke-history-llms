"""Build stratum C: ground-truth post-1929 contaminants from the deliberately-contaminated eval corpus.

2026-07-22, office box. PI-approved stratum ("no downside to C"): these are the corpus's own
planted/known contaminants — same shards, same OCR/format as the negatives, so models cannot cheat
on format; labels come from the gold audits, not from marker hits (contaminants_bucket.csv contains
homograph false positives and is NOT used as a label source).
Files used:
  gold audits:  intermediates\\gold_300_verdicts.csv, gold_1000_verdicts.csv, gold_1000b_verdicts.csv
                (per-book front-matter audit: contaminated_1929, confidence, is_reprint, stratum)
  got-throughs: intermediates\\genuine_gotthrough_bucket.csv (3 pipeline-missed contaminants)
  corpus map:   intermediates\\corpus_meta.parquet (id -> shard)
  output:       <repo>\\manifest_contamC.csv  + passages\\passages_contamC.parquet
Rules:
  keep contaminated_1929 == True with confidence >= 0.9 (label purity for a recall benchmark);
  cls = contam_modern (is_reprint False -> body text IS post-1929; the recall headline) or
        contam_reprint_old_body (modern printing of old content -> content-reading should NOT
        flag these; diagnostic floor, excluded from the recall headline);
  got-throughs enter as cls = contam_gotthrough (the hardest real cases).
"""
import os
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
I = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating\intermediates"

gold = []
for name in ["gold_300_verdicts", "gold_1000_verdicts", "gold_1000b_verdicts"]:
    g = pd.read_csv(os.path.join(I, name + ".csv"), dtype={"id": str})
    g["gold_file"] = name
    gold.append(g[["id", "source", "stratum", "meta_year", "contaminated_1929", "confidence",
                   "is_reprint", "edition_year", "gold_file"]])
gold = pd.concat(gold, ignore_index=True)
gold["contaminated_1929"] = gold.contaminated_1929.astype(str).str.lower() == "true"
gold["is_reprint"] = gold.is_reprint.astype(str).str.lower() == "true"
gold = gold.sort_values("confidence", ascending=False).drop_duplicates("id")
c = gold[gold.contaminated_1929 & (gold.confidence >= 0.9)].copy()
print(f"gold union: {len(gold)} unique audited; contaminated & conf>=0.9: {len(c)} "
      f"(dropped {int((gold.contaminated_1929 & (gold.confidence < 0.9)).sum())} low-conf)")

cm = pd.read_parquet(os.path.join(I, "corpus_meta.parquet"))[["id", "shard", "source"]]
c = c.drop(columns=["source"]).merge(cm, on="id", how="left")
missing = c.shard.isna().sum()
if missing: print(f"WARNING: {missing} gold contaminants not in corpus_meta — dropped")
c = c.dropna(subset=["shard"])

gt = pd.read_csv(os.path.join(I, "genuine_gotthrough_bucket.csv"), dtype={"id": str})
gt = gt[["id"]].merge(cm, on="id", how="left")
gt["cls"] = "contam_gotthrough"

neg = set(pd.read_csv(os.path.join(REPO, "manifest_negatives.csv"), dtype={"id": str}).id)
pos = set(pd.read_csv(os.path.join(REPO, "manifest_positives.csv"), dtype={"id": str}).id)
overlap_neg = c.id.isin(neg).sum() + gt.id.isin(neg).sum()
print(f"overlap with negatives manifest: {overlap_neg} (must be 0)")
assert overlap_neg == 0, "gold contaminant appears in negatives manifest — STOP"

c["cls"] = c.is_reprint.map({False: "contam_modern", True: "contam_reprint_old_body"})
out = pd.concat([
    c[["id", "source", "meta_year", "shard", "cls"]],
    gt.assign(meta_year=pd.NA)[["id", "source", "meta_year", "shard", "cls"]],
], ignore_index=True).drop_duplicates("id")
out["label"] = 1
out["origin"] = "local_corpus"
out.to_csv(os.path.join(REPO, "manifest_contamC.csv"), index=False)
print(f"manifest_contamC.csv: {len(out)} books")
print(out.cls.value_counts().to_string())

import extract_passages_v2 as ex
os.makedirs(os.path.join(ex.DROPBOX, "passages"), exist_ok=True)
ex.run("manifest_contamC.csv", "passages_contamC.parquet")
