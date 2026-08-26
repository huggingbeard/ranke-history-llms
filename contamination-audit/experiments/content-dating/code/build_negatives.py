"""Select 1,000 high-confidence CLEAN (pre-1929) institutional_books from our corpus for the content-dating
experiment. Clean = every date signal <=1929, no post-1929 date-context years, no anachronism marker hits."""
import pandas as pd, os
# resolve our working files (kept in the ranke-contamination-audit scratch); adjust if paths move
SCRATCH=r"C:\Users\joach\AppData\Local\Temp\claude\C--Users-joach\2cd7fedb-222f-44e8-8c12-3c366e44d737\scratchpad"
OUT=os.path.dirname(os.path.abspath(__file__))
meta=pd.read_parquet(os.path.join(SCRATCH,"corpus_meta.parquet"))          # id, source, year(meta), stage2_docyear, shard
v2=pd.read_parquet(os.path.join(SCRATCH,"fullrun_v2.parquet"))[["id","imprint_v2"]]
v3=pd.read_parquet(os.path.join(SCRATCH,"fullrun_v3.parquet"))[["id","n_datectx_post"]]
marker_hits=set(pd.read_csv(os.path.join(SCRATCH,"clean_check9_hits.csv"),dtype={"id":str}).id)
d=meta.merge(v2,on="id",how="left").merge(v3,on="id",how="left")
clean=d[(d.source=="institutional_books") & (d.year<=1929) & (d.stage2_docyear<=1929)
        & ((d.imprint_v2.isna())|(d.imprint_v2<=1929)) & (d.n_datectx_post.fillna(0)==0)
        & (~d.id.isin(marker_hits))]
print(f"clean institutional_books candidates: {len(clean)}")
neg=clean.sample(min(1000,len(clean)),random_state=1929).copy()
neg["label"]=0; neg["cls"]="clean_pre1929"; neg["origin"]="local_corpus"
neg[["id","source","year","stage2_docyear","shard","label","cls","origin"]].rename(columns={"year":"meta_year"}).to_csv(os.path.join(OUT,"manifest_negatives.csv"),index=False)
print(f"wrote manifest_negatives.csv: {len(neg)} clean books  (meta year range {int(neg.year.min())}-{int(neg.year.max())})")
