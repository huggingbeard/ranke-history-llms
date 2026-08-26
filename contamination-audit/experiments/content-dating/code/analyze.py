"""Analyze flotilla judgments -> precision/recall/F1 surfaces over model x N-passages x length x sampling.
N is derived by subsetting the per-passage judgments (any-positive aggregation = recall-max)."""
import os, glob, re
import pandas as pd
DROPBOX=r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
OUT=os.path.join(DROPBOX,"outputs")
NS=[1,2,4,8,16]
rows=[]
for f in glob.glob(os.path.join(OUT,"judg_*.parquet")):
    model=re.sub(r".*judg_","",f).replace(".parquet","").replace("__","/")
    d=pd.read_parquet(f)
    for L in sorted(d.length.unique()):
        for strat in sorted(d.strategy.unique()):
            sub=d[(d.length==L)&(d.strategy==strat)]
            for N in NS:
                # any-positive over the first N passages per book
                g=sub[sub.idx<N].groupby("id").agg(label=("label","first"), flag=("pred", lambda x: int((x==1).any()))).reset_index()
                t=g.label.astype(bool).values; p=g.flag.astype(bool).values
                tp=int((t&p).sum()); fp=int((~t&p).sum()); fn=int((t&~p).sum()); tn=int((~t&~p).sum())
                P=tp/(tp+fp) if tp+fp else 1; R=tp/(tp+fn) if tp+fn else 1; F=2*P*R/(P+R) if P+R else 0
                rows.append(dict(model=model,length=L,strategy=strat,N=N,precision=round(P,3),recall=round(R,3),f1=round(F,3),n_books=len(g)))
res=pd.DataFrame(rows); res.to_csv(os.path.join(OUT,"results.csv"),index=False)
print(res.sort_values(["recall","f1"],ascending=False).head(25).to_string(index=False))
# headline: cheapest config (smallest model x fewest passages x shortest snippet) with recall>=0.95
ok=res[res.recall>=0.95].sort_values(["N","length"])
print("\nconfigs with recall >= 0.95 (cheapest first):")
print(ok.head(15).to_string(index=False) if len(ok) else "  none reached 0.95 — widen sampling or snippet length")
