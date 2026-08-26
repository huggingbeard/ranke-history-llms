"""Qwen3.6-35B-A3B on the 4090 via llama.cpp (UD-Q3_K_M GGUF) — frontier-slice judging.

v1 (2026-07-23), office box. PI: "the qwen 35 needs more quantization ... but can run well on 4090".
vLLM could not run this model on 24 GB (two ~21 GB 4-bit quants: silent hang / CPU-bound eager;
third quant incompatible). Q3_K_M = 16.6 GB -> ~6 GB headroom on llama.cpp's native CUDA path.
All 3-bit CUDA-capable quants: only GGUF exists (the rest are MLX/Apple).

Frontier slice only (the configs that decide the production question; full 648-config sweep would
be wasteful at llama.cpp throughput): L in {1000, 2000} x {random, grep} x idx<8, over all four
passage sets (negatives, stratum A, stratum C, stratum B) = ~91k passages.

Server (start first, waits for GPU to be free):
  C:\\Users\\hvoth\\llama-cpp\\bin\\llama-server.exe -m C:\\Users\\hvoth\\models\\Qwen3.6-35B-A3B-UD-Q3_K_M.gguf ^
      -ngl 999 -c 16384 -np 8 -b 2048 -ub 1024 --port 8087 --jinja
Output: outputs\\judg_Qwen36-35B-A3B-Q3-llamacpp.parquet + timings json (same schema as vLLM runs;
runtime differs from the vLLM flotilla — flagged in the report).
"""
import os, re, glob, json, time, asyncio
import pandas as pd
import httpx

DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
PASS = os.path.join(DROPBOX, "passages"); OUT = os.path.join(DROPBOX, "outputs")
URL = "http://127.0.0.1:8087/v1/chat/completions"
SAFE = "Qwen36-35B-A3B-Q3-llamacpp"
CKPT = os.path.join(OUT, f"judg_{SAFE}.ckpt.csv")
CONCURRENCY = 8

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
    body = {"messages": [{"role": "user", "content": PROMPT.format(p=p[:8000])}],
            "temperature": 0, "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False}}
    async with sem:
        for a in range(4):
            try:
                r = await c.post(URL, json=body, timeout=300)
                r.raise_for_status()
                return parse(r.json()["choices"][0]["message"]["content"] or "")
            except Exception:
                if a == 3: return None
                await asyncio.sleep(3 * (a + 1))
    return None

async def main():
    passages = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(PASS, "passages_*.parquet")))],
                         ignore_index=True)
    sl = passages[(passages.length.isin([1000, 2000])) & (passages.idx < 8)].copy()
    sl = sl.drop_duplicates(["id", "length", "strategy", "idx"]).reset_index(drop=True)
    # llama.cpp gives only ~2-3 p/s on this MoE quant (slow expert kernels on the win-CUDA path),
    # so the 35B gets a seeded SUBSAMPLE at the headline config, not the full slice: the
    # engineering verdict (25x slower than gpt-oss => not the production pick) is already fixed;
    # this run only estimates its recall (+/- ~4-5 pp on ~320 books/straturm).
    repo = os.path.dirname(os.path.abspath(__file__))
    import random as _r
    a_ids = set(pd.read_csv(os.path.join(repo, "manifest_positives.csv"), dtype={"id": str}).id)
    neg_ids = sorted(set(pd.read_csv(os.path.join(repo, "manifest_negatives.csv"), dtype={"id": str}).id) & set(sl.id))
    b_ids = sorted(set(pd.read_csv(os.path.join(repo, "manifest_pos3046.csv"), dtype={"id": str}).id) & set(sl.id))
    rng = _r.Random(1929)
    keep_ids = set(rng.sample(neg_ids, min(400, len(neg_ids)))) | set(rng.sample(b_ids, min(400, len(b_ids))))
    keep_ids |= set(sl.id) - a_ids - set(neg_ids) - set(b_ids)      # all stratum-C books
    sl = sl[sl.id.isin(keep_ids) & (sl.length == 2000) & (sl.strategy == "random")].reset_index(drop=True)
    print(f"35B subsample: {len(sl)} passages / {sl.id.nunique()} books (L=2000 random, idx<8)", flush=True)

    done = {}
    if os.path.exists(CKPT) and os.path.getsize(CKPT) > 40:   # >header: survive a crash-before-flush file
        ck = pd.read_csv(CKPT, dtype={"id": str})
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
        # smoke first
        s1 = await ask(c, sem, "The astronaut checked her email after the television broadcast of the 1969 moon landing.")
        s2 = await ask(c, sem, "His Majesty's ship anchored in the roads of Portsmouth, whence the convoy of 1807 was to sail.")
        print(f"SMOKE llama.cpp Q3: [{s1}, {s2}] (want [1, 0])", flush=True)
        if s1 is None and s2 is None:
            raise SystemExit("server not answering — is llama-server up on :8087?")

        async def one(row):
            nonlocal n_done
            key = (row.id, row.length, row.strategy, row.idx)
            if key in done: return
            v = await ask(c, sem, row.passage)
            async with lock:
                fck.write(f"{row.id},{row.length},{row.strategy},{row.idx},{'' if v is None else v}\n")
                n_done += 1
                if n_done % 2000 == 0:
                    fck.flush()
                    print(f"  {n_done + len(done)}/{len(sl)} ({n_done / (time.time() - t0):.1f} p/s)", flush=True)

        await asyncio.gather(*[one(r) for r in sl.itertuples()])
    fck.close()

    ck = pd.read_csv(CKPT, dtype={"id": str})
    ck = ck.drop_duplicates(["id", "length", "strategy", "idx"])
    full = sl[["id", "label", "length", "strategy", "idx"]].merge(ck, on=["id", "length", "strategy", "idx"], how="left")
    full.to_parquet(os.path.join(OUT, f"judg_{SAFE}.parquet"))
    wall = time.time() - t0
    with open(os.path.join(OUT, f"timings_{SAFE}.json"), "w") as f:
        json.dump({"model": SAFE, "n_passages": len(full), "n_new_this_run": n_done,
                   "wall_seconds_this_run": round(wall, 1),
                   "passages_per_second": round(n_done / max(wall, 1), 2),
                   "null_preds": int(full.pred.isna().sum()),
                   "note": "llama.cpp b10091 win-cuda, UD-Q3_K_M, frontier slice only (L 1000/2000, idx<8)"}, f, indent=1)
    print(f"DONE: {len(full)} judged, {int(full.pred.isna().sum())} null -> judg_{SAFE}.parquet", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
