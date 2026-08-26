"""Judge the frontier slice with UNQUANTIZED Qwen3.6-35B-A3B on the RunPod A100 (vLLM, HTTPS proxy).

v1 (2026-07-23). Slice: negatives + stratum C + stratum B (NO stratum A), L in {1000, 2000},
both strategies, idx<8 — the full frontier this time (the A100 has the throughput; the local
Q3 run could only afford a subsample of one config).
Endpoint: https://lv2uorbgg0obqy-8000.proxy.runpod.net/v1/chat/completions
Output: outputs\\judg_Qwen36-35B-A3B-BF16-A100.parquet + timings json. Checkpointed/resumable.
"""
import os, re, glob, json, time, asyncio
import pandas as pd
import httpx

DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
PASS = os.path.join(DROPBOX, "passages"); OUT = os.path.join(DROPBOX, "outputs")
POD = "lv2uorbgg0obqy"
URL = f"https://{POD}-8000.proxy.runpod.net/v1/chat/completions"
MODEL = "Qwen/Qwen3.6-35B-A3B"
SAFE = "Qwen36-35B-A3B-BF16-A100"
CKPT = os.path.join(OUT, f"judg_{SAFE}.ckpt.csv")
CONCURRENCY = 48

PROMPT = ('Read this passage from a scanned book and judge: was its text WRITTEN before 1929, or after 1929? '
          'Old, foreign-language, poetic, or historical writing is BEFORE. Modern prose, a post-1929 event/person/'
          'technology, or a modern perspective is AFTER. Ignore OCR noise. Passage:\n"""{p}"""\n'
          'Answer with only one word: BEFORE or AFTER.')

def parse(txt):
    t = txt.strip()
    if "</think>" in t: t = t.rsplit("</think>", 1)[1]
    hits = re.findall(r"\b(BEFORE|AFTER)\b", t.upper())
    return {"BEFORE": 0, "AFTER": 1}[hits[-1]] if hits else None

async def ask(c, sem, p):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": PROMPT.format(p=p[:8000])}],
            "temperature": 0, "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False}}
    async with sem:
        for a in range(5):
            try:
                r = await c.post(URL, json=body, timeout=180)
                r.raise_for_status()
                return parse(r.json()["choices"][0]["message"]["content"] or "")
            except Exception:
                if a == 4: return None
                await asyncio.sleep(4 * (a + 1))
    return None

async def main():
    passages = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(PASS, "passages_*.parquet")))],
                         ignore_index=True)
    repo = os.path.dirname(os.path.abspath(__file__))
    a_ids = set(pd.read_csv(os.path.join(repo, "manifest_positives.csv"), dtype={"id": str}).id)
    sl = passages[(passages.length.isin([1000, 2000])) & (passages.idx < 8) & (~passages.id.isin(a_ids))]
    sl = sl.drop_duplicates(["id", "length", "strategy", "idx"]).reset_index(drop=True)
    print(f"A100 frontier slice: {len(sl)} passages / {sl.id.nunique()} books", flush=True)

    done = {}
    if os.path.exists(CKPT) and os.path.getsize(CKPT) > 40:
        ck = pd.read_csv(CKPT, dtype={"id": str})
        ck = ck[ck.pred.notna()]
        done = {(r.id, r.length, r.strategy, r.idx): r.pred for r in ck.itertuples()}
        print(f"resume: {len(done)} already judged", flush=True)
    elif os.path.exists(CKPT):
        os.remove(CKPT)
    fresh = not os.path.exists(CKPT)
    fck = open(CKPT, "a", newline="", encoding="utf-8")
    if fresh: fck.write("id,length,strategy,idx,pred\n")

    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    t0 = time.time(); n_done = 0
    async with httpx.AsyncClient() as c:
        s1 = await ask(c, sem, "The astronaut checked her email after the television broadcast of the 1969 moon landing.")
        s2 = await ask(c, sem, "His Majesty's ship anchored in the roads of Portsmouth, whence the convoy of 1807 was to sail.")
        print(f"SMOKE A100 BF16: [{s1}, {s2}] (want [1, 0])", flush=True)
        if s1 is None and s2 is None:
            raise SystemExit("pod not answering")

        async def one(row):
            nonlocal n_done
            if (row.id, row.length, row.strategy, row.idx) in done: return
            v = await ask(c, sem, row.passage)
            async with lock:
                fck.write(f"{row.id},{row.length},{row.strategy},{row.idx},{'' if v is None else v}\n")
                n_done += 1
                if n_done % 4000 == 0:
                    fck.flush()
                    print(f"  {n_done + len(done)}/{len(sl)} ({n_done / (time.time() - t0):.1f} p/s)", flush=True)

        await asyncio.gather(*[one(r) for r in sl.itertuples()])
    fck.close()

    ck = pd.read_csv(CKPT, dtype={"id": str}).drop_duplicates(["id", "length", "strategy", "idx"])
    full = sl[["id", "label", "length", "strategy", "idx"]].merge(ck, on=["id", "length", "strategy", "idx"], how="left")
    full.to_parquet(os.path.join(OUT, f"judg_{SAFE}.parquet"))
    wall = time.time() - t0
    with open(os.path.join(OUT, f"timings_{SAFE}.json"), "w") as f:
        json.dump({"model": SAFE, "n_passages": len(full), "n_new_this_run": n_done,
                   "wall_seconds_this_run": round(wall, 1),
                   "passages_per_second": round(n_done / max(wall, 1), 2),
                   "null_preds": int(full.pred.isna().sum()),
                   "note": "unquantized BF16 on RunPod A100-SXM4-80GB, vllm/vllm-openai, $1.49/hr"}, f, indent=1)
    print(f"DONE: {len(full)} judged, {int(full.pred.isna().sum())} null -> judg_{SAFE}.parquet", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
