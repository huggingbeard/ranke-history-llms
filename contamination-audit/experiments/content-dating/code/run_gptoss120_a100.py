"""Judge frontier + OOS slices with gpt-oss-120b on the A100 pod (option 1 of the PI's upgrade round).
v1 (2026-07-24). Pod m62oi8ecbjnpuq; model openai/gpt-oss-120b (63 GB MXFP4, reasoning_effort=low).
Slices: in-sample frontier (neg+C+B, L {1000,2000}, both strategies, idx<8) + OOS slice = ~63.6k
passages. Output: judg_gptoss-120b-A100.parquet (+ .oos rows included; split at scoring by id).
Teardown after: runpod_a100_verify.py terminate (triple protocol).
"""
import os, re, json, time, asyncio, glob
import pandas as pd
import httpx

DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
PASS = os.path.join(DROPBOX, "passages"); OUT = os.path.join(DROPBOX, "outputs")
URL = "https://m62oi8ecbjnpuq-8000.proxy.runpod.net/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
SAFE = "gptoss-120b-A100"
CKPT = os.path.join(OUT, f"judg_{SAFE}.ckpt.csv")
CONCURRENCY = 48

PROMPT = ('Read this passage from a scanned book and judge: was its text WRITTEN before 1929, or after 1929? '
          'Old, foreign-language, poetic, or historical writing is BEFORE. Modern prose, a post-1929 event/person/'
          'technology, or a modern perspective is AFTER. Ignore OCR noise. Passage:\n"""{p}"""\n'
          'Answer with only one word: BEFORE or AFTER.')

def parse(txt):
    t = txt.strip()
    for sep in ("assistantfinal", "final", "</think>"):
        if sep in t: t = t.rsplit(sep, 1)[1]; break
    hits = re.findall(r"\b(BEFORE|AFTER)\b", t.upper())
    return {"BEFORE": 0, "AFTER": 1}[hits[-1]] if hits else None

async def ask(c, sem, p):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT.format(p=p[:8000])}],
            "temperature": 0, "max_tokens": 768, "chat_template_kwargs": {"reasoning_effort": "low"}}
    async with sem:
        for a in range(5):
            try:
                r = await c.post(URL, json=body, timeout=180)
                r.raise_for_status()
                return parse(r.json()["choices"][0]["message"]["content"] or "")
            except Exception:
                if a == 4: return None
                await asyncio.sleep(4 * (a + 1))

async def main():
    repo = os.path.dirname(os.path.abspath(__file__))
    a_ids = set(pd.read_csv(os.path.join(repo, "manifest_positives.csv"), dtype={"id": str}).id)
    base = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(PASS, "passages_*.parquet")))
                      if "oos_slice" not in f], ignore_index=True)
    base = base[(base.length.isin([1000, 2000])) & (base.idx < 8) & (~base.id.isin(a_ids))]
    oos = pd.read_parquet(os.path.join(PASS, "passages_oos_slice.parquet"))
    sl = pd.concat([base, oos], ignore_index=True).drop_duplicates(["id", "length", "strategy", "idx"]).reset_index(drop=True)
    print(f"slices: {len(sl)} passages / {sl.id.nunique()} books", flush=True)
    done = {}
    if os.path.exists(CKPT) and os.path.getsize(CKPT) > 40:
        ck = pd.read_csv(CKPT, dtype={"id": str}); ck = ck[ck.pred.notna()]
        done = {(r.id, r.length, r.strategy, r.idx): r.pred for r in ck.itertuples()}
        print(f"resume: {len(done)}", flush=True)
    elif os.path.exists(CKPT):
        os.remove(CKPT)
    fresh = not os.path.exists(CKPT)
    fck = open(CKPT, "a", newline="", encoding="utf-8")
    if fresh: fck.write("id,length,strategy,idx,pred\n")
    sem = asyncio.Semaphore(CONCURRENCY); lock = asyncio.Lock()
    t0 = time.time(); n_done = 0
    async with httpx.AsyncClient() as c:
        s1 = await ask(c, sem, "The astronaut checked her email after the television broadcast of the 1969 moon landing.")
        s2 = await ask(c, sem, "His Majesty's ship anchored in the roads of Portsmouth, whence the convoy of 1807 was to sail.")
        print(f"SMOKE 120b: [{s1}, {s2}] (want [1, 0])", flush=True)
        if s1 is None and s2 is None: raise SystemExit("pod not answering")
        async def one(row):
            nonlocal n_done
            if (row.id, row.length, row.strategy, row.idx) in done: return
            v = await ask(c, sem, row.passage)
            async with lock:
                fck.write(f"{row.id},{row.length},{row.strategy},{row.idx},{'' if v is None else v}\n")
                n_done += 1
                if n_done % 5000 == 0:
                    fck.flush(); print(f"  {n_done + len(done)}/{len(sl)} ({n_done / (time.time() - t0):.1f} p/s)", flush=True)
        await asyncio.gather(*[one(r) for r in sl.itertuples()])
    fck.close()
    ck = pd.read_csv(CKPT, dtype={"id": str}).drop_duplicates(["id", "length", "strategy", "idx"])
    full = sl[["id", "label", "length", "strategy", "idx"]].merge(ck, on=["id", "length", "strategy", "idx"], how="left")
    full.to_parquet(os.path.join(OUT, f"judg_{SAFE}.parquet"))
    wall = time.time() - t0
    with open(os.path.join(OUT, f"timings_{SAFE}.json"), "w") as f:
        json.dump({"model": SAFE, "n_passages": len(full), "wall_seconds_this_run": round(wall, 1),
                   "passages_per_second": round(n_done / max(wall, 1), 2),
                   "null_preds": int(full.pred.isna().sum()),
                   "note": "gpt-oss-120b MXFP4 on A100-SXM4-80GB, reasoning_effort=low"}, f, indent=1)
    print(f"DONE: {len(full)} judged, {int(full.pred.isna().sum())} null ({n_done / max(wall, 1):.1f} p/s)", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
