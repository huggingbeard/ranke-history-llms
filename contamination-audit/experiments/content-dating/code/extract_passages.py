"""Extract passages for the experiment. For every book (neg = local corpus, pos = Dropbox HF pull):
 - RANDOM: 16 passages sampled uniformly across the body (skip first/last 5%).
 - GREP-DIRECTED: passages centred on anachronism-marker hits (falls back to random if none).
At each snippet length L in {250,500,1000,2000} chars. Written once to Dropbox; the flotilla derives the
N-passages axis by subsetting these 16. Deterministic (seeded) so runs are reproducible.
"""
import os, re, json, random, glob
import pandas as pd, pyarrow.parquet as pq, ahocorasick
REPO=os.path.dirname(os.path.abspath(__file__))
DROPBOX=r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
CORPUS=r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\temp-guohui\decontamination\datasets\history-llms\sample-10gb\data"
LENGTHS=[250,500,1000,2000]; N=16; SEED=1929
# anachronism markers for grep-direction (from the decontaminator, minus corpus-homographs)
MK=r"C:\Users\joach\Documents\GitHub\anachronism-decontaminator\markers_1945.json"
DROP={"internet","nato","lsd","common market","european community","third world","nasa","opec","laser","motel"}
markers=[m["term"].lower() for m in json.load(open(MK))["markers"] if m["term"].lower() not in DROP]
A=ahocorasick.Automaton()
for m in markers: A.add_word(m,m)
A.make_automaton()

def load_text(row):
    if row.origin=="local_corpus":
        t=pq.read_table(os.path.join(CORPUS,row.shard),columns=["id","text"])
        d=dict(zip(t.column("id").to_pylist(),t.column("text").to_pylist())); return d.get(row.id,"")
    p=os.path.join(DROPBOX,"positives_text",row.id+".txt")
    return open(p,encoding="utf-8").read() if os.path.exists(p) else ""

def rng(bid): return random.Random(f"{bid}-{SEED}")
def random_centers(tx,r):
    lo,hi=int(len(tx)*0.05),int(len(tx)*0.95)
    return [r.randint(lo,hi) for _ in range(N)] if hi>lo else [len(tx)//2]*N
def grep_centers(low):
    cs=[]
    for end,term in A.iter(low):
        cs.append(end-len(term)//2)
        if len(cs)>=N: break
    return cs

def passages(tx,centers,L):
    h=L//2; out=[]
    for c in centers[:N]:
        s=max(0,c-h); out.append(tx[s:s+L].replace("\n"," ").strip())
    return out

def run(manifest, out_name):
    df=pd.read_csv(os.path.join(REPO,manifest),dtype={"id":str})
    # local corpus rows need shard; group reads
    rows=[]
    for _,row in df.iterrows():
        tx=load_text(row)
        if len(tx)<1000: continue
        low=" ".join(tx.lower().split()); r=rng(row.id)
        rc=random_centers(tx,r); gc=grep_centers(low) or rc
        for L in LENGTHS:
            for strat,centers in (("random",rc),("grep",gc)):
                for j,ps in enumerate(passages(tx,centers,L)):
                    rows.append({"id":row.id,"label":int(row.label),"length":L,"strategy":strat,"idx":j,"passage":ps})
    pd.DataFrame(rows).to_parquet(os.path.join(DROPBOX,"passages",out_name))
    print(f"{manifest}: wrote {len(rows)} passages -> {out_name}", flush=True)

if __name__=="__main__":
    os.makedirs(os.path.join(DROPBOX,"passages"),exist_ok=True)
    run("manifest_negatives.csv","passages_neg.parquet")
    if os.path.exists(os.path.join(REPO,"manifest_positives.csv")):
        run("manifest_positives.csv","passages_pos.parquet")
    else:
        print("manifest_positives.csv not found — run pull_positives.py first")
