"""Certify clean negatives with deepseek-v4-flash + Llama-3.3-70B, BOTH via OpenRouter (no Groq daily cap).
Read most of the body in chunks; certified clean iff NEITHER flags post-1929 content."""
import pandas as pd, pyarrow.parquet as pq, os, json, asyncio, httpx, sys
SP=os.path.dirname(os.path.abspath(__file__))
DATA=r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\temp-guohui\decontamination\datasets\history-llms\sample-10gb\data"
OKEY=open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\openrouter.txt").read().strip()
OURL="https://openrouter.ai/api/v1/chat/completions"
N=int(sys.argv[1]) if len(sys.argv)>1 else 20
CHUNKS=12; CHUNKLEN=1500
cand=pd.read_csv(r"C:\Users\joach\Documents\GitHub\ranke-edition-dating\experiments\content-dating\manifest_negatives.csv",dtype={"id":str}).head(N)
txt={}
for sh,g in cand.groupby("shard"):
    t=pq.read_table(os.path.join(DATA,sh),columns=["id","text"]); d=dict(zip(t.column("id").to_pylist(),t.column("text").to_pylist()))
    for i in g.id: txt[i]=d[i]
def chunks(tx):
    lo,hi=int(len(tx)*0.03),int(len(tx)*0.97); span=hi-lo
    if span<CHUNKLEN: return [tx]
    step=max(1,span//CHUNKS); return [" ".join(tx[p:p+CHUNKLEN].split()) for p in range(lo,hi,step)][:CHUNKS]
PROMPT='This passage is from a book catalogued as pre-1929. Does it contain content or knowledge that could only exist AFTER 1929 (a post-1929 event, person, technology, dated reference, or clearly modern perspective)? Old, foreign-language, poetic, or historical text is NOT post-1929. Judge the writing, ignore OCR noise. Passage:\n"""{p}"""\nReturn JSON: {{"post1929": true/false, "why": "<=8 words"}}'
async def ask(c,sem,model,p,extra=None):
    b={"model":model,"messages":[{"role":"user","content":PROMPT.format(p=p)}],"temperature":0,"max_tokens":150,"provider":{"sort":"price"}}
    if extra: b.update(extra)
    async with sem:
        for a in range(4):
            try:
                r=await c.post(OURL,headers={"Authorization":f"Bearer {OKEY}"},json=b,timeout=90)
                if r.status_code>=500 or r.status_code==429: await asyncio.sleep(2*(a+1)); continue
                r.raise_for_status(); ct=r.json()["choices"][0]["message"]["content"]
                return bool(json.loads(ct[ct.find("{"):ct.rfind("}")+1]).get("post1929"))
            except Exception:
                if a==3: return None
                await asyncio.sleep(2*(a+1))
async def main():
    sem=asyncio.Semaphore(10); rows=[]
    async with httpx.AsyncClient() as c:
        async def one(bid):
            chs=chunks(txt.get(bid,""))
            lres=await asyncio.gather(*[ask(c,sem,"meta-llama/llama-3.3-70b-instruct",p) for p in chs])
            dres=await asyncio.gather(*[ask(c,sem,"deepseek/deepseek-v4-flash",p,{"reasoning":{"enabled":False}}) for p in chs])
            lf=any(x for x in lres if x); df=any(x for x in dres if x)
            why=next((chs[i][:60] for i,x in enumerate(lres) if x), next((chs[i][:60] for i,x in enumerate(dres) if x),""))
            return {"id":bid,"llama_flag":lf,"deepseek_flag":df,"either":lf or df,"both":lf and df,"evidence":why}
        for coro in asyncio.as_completed([one(b) for b in cand.id]): rows.append(await coro)
    d=pd.DataFrame(rows); d.to_csv(os.path.join(SP,"certify_or_pilot.csv"),index=False)
    print(f"PILOT (OpenRouter): {len(d)} clean candidates, {CHUNKS} body chunks, Llama-3.3-70B + deepseek-v4-flash")
    print(f"  flagged by llama70b: {int(d.llama_flag.sum())}  by deepseek-flash: {int(d.deepseek_flag.sum())}")
    print(f"  EITHER: {int(d.either.sum())} ({100*d.either.mean():.0f}%)   BOTH: {int(d.both.sum())}")
    for _,r in d[d.either].iterrows():
        print(f"    {r.id[:10]} llama={r.llama_flag} deepseek={r.deepseek_flag} :: {str(r.evidence)[:70]}")
asyncio.run(main())
