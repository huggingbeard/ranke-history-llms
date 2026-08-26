"""Certify all 999 clean negatives with deepseek-v4-flash + Llama-3.3-70B via OpenRouter.

v2 (2026-07-22) — full-run rewrite of certify_negatives.py (pilot), run on the office 4090 box.
Files used:
  manifest:  <this repo dir>\\manifest_negatives.csv  (999 books)
  corpus:    D:\\UZHechist Dropbox\\Joachim Voth\\claudecode\\ranke-contamination-audit\\temp-guohui\\decontamination\\datasets\\history-llms\\sample-10gb\\data  (32 shards)
  key:       D:\\UZHechist Dropbox\\Joachim Voth\\claudecode\\api\\openrouter.txt
  output:    D:\\...\\ranke-contamination-audit\\experiments\\content-dating\\intermediates\\certify_negatives_full.csv

Changes vs pilot (documented for the PI — all PROVISIONAL):
  - fixed manifest path (was a stale C:\\Users\\joach\\... path from the laptop).
  - coverage: pilot read 12x1500 chars (~14% of a median book). RUNBOOK/OFFICE_CLAUDE say "read most of
    the body", so v2 reads up to 24 chunks x 3000 chars targeting ~70%% of the body span (median book
    ~127k chars -> ~57%% actual). Same prompt, temperature 0.
  - Llama-3.3-70B runs on OpenRouter, NOT locally: a 70B at 4-bit (~37 GB) does not fit the 4090's
    24 GB, and CPU-offload would take days for ~40k calls. OpenRouter has no daily cap (the Groq
    concern does not apply). Deviation from OFFICE_CLAUDE's "run Llama locally" note — flagged.
  - checkpointing: one CSV row per book appended as completed; resume skips done ids.
  - error-aware: a chunk call that fails 4 retries counts as an error; a book with zero flags but
    errors > 0 is NOT certified clean (goes to the review queue), so silent API failures cannot
    launder a book into the negative set.
Verdicts: certified_clean = neither model flags any chunk and no errors. Any single-model flag or
error -> review queue. Both-model flag -> drop (definite contamination).
"""
import os, json, csv, sys, asyncio, math, time
import pandas as pd
import pyarrow.parquet as pq
import httpx

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\temp-guohui\decontamination\datasets\history-llms\sample-10gb\data"
OUT_CSV = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating\intermediates\certify_negatives_full.csv"
OKEY = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\openrouter.txt").read().strip()
OURL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = {"llama": "meta-llama/llama-3.3-70b-instruct", "deepseek": "deepseek/deepseek-v4-flash"}
CHUNKLEN = 3000
MAX_CHUNKS = 24
COVERAGE = 0.70
CONCURRENCY = 48
FIELDS = ["id", "n_chunks", "llama_flag", "deepseek_flag", "llama_errors", "deepseek_errors",
          "either", "both", "verdict", "evidence"]
PROMPT = ('This passage is from a book catalogued as pre-1929. Does it contain content or knowledge that '
          'could only exist AFTER 1929 (a post-1929 event, person, technology, dated reference, or clearly '
          'modern perspective)? Old, foreign-language, poetic, or historical text is NOT post-1929. Judge '
          'the writing, ignore OCR noise. Passage:\n"""{p}"""\n'
          'Return JSON: {{"post1929": true/false, "why": "<=8 words"}}')

def chunks(tx):
    lo, hi = int(len(tx) * 0.03), int(len(tx) * 0.97)
    span = hi - lo
    if span <= CHUNKLEN: return [" ".join(tx.split())]
    n = min(MAX_CHUNKS, max(1, math.ceil(span * COVERAGE / CHUNKLEN)))
    step = max(CHUNKLEN, span // n)
    return [" ".join(tx[p:p + CHUNKLEN].split()) for p in range(lo, hi - CHUNKLEN + 1, step)][:MAX_CHUNKS]

async def ask(c, sem, model, p, extra=None):
    """Returns True/False (flag verdict) or None (call failed after retries)."""
    b = {"model": model, "messages": [{"role": "user", "content": PROMPT.format(p=p)}],
         "temperature": 0, "max_tokens": 200, "provider": {"sort": "price"}}
    if extra: b.update(extra)
    async with sem:
        for a in range(4):
            try:
                r = await c.post(OURL, headers={"Authorization": f"Bearer {OKEY}"}, json=b, timeout=120)
                if r.status_code >= 500 or r.status_code == 429:
                    await asyncio.sleep(3 * (a + 1)); continue
                r.raise_for_status()
                ct = r.json()["choices"][0]["message"]["content"]
                if not ct or "{" not in ct:            # empty-response guard
                    await asyncio.sleep(2 * (a + 1)); continue
                return bool(json.loads(ct[ct.find("{"):ct.rfind("}") + 1]).get("post1929"))
            except Exception:
                if a == 3: return None
                await asyncio.sleep(3 * (a + 1))
    return None

async def preflight(c):
    r = await c.get("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {OKEY}"}, timeout=30)
    print("credits:", r.json().get("data"), flush=True)
    sem = asyncio.Semaphore(2)
    test = "The televised moon landing of 1969 was watched by millions on the internet era's precursor networks."
    for tag, m in MODELS.items():
        v = await ask(c, sem, m, test, {"reasoning": {"enabled": False}} if tag == "deepseek" else None)
        print(f"preflight {m}: post1929={v}", flush=True)
        if v is not True:
            raise SystemExit(f"PREFLIGHT FAILED for {m} — expected True, got {v}")

async def main():
    cand = pd.read_csv(os.path.join(REPO, "manifest_negatives.csv"), dtype={"id": str})
    done = set()
    if os.path.exists(OUT_CSV):
        done = set(pd.read_csv(OUT_CSV, dtype={"id": str}).id)
        print(f"resume: {len(done)} books already certified", flush=True)
    todo = cand[~cand.id.isin(done)]
    print(f"loading text for {len(todo)} books from {todo.shard.nunique()} shards", flush=True)
    txt = {}
    for sh, g in todo.groupby("shard"):
        t = pq.read_table(os.path.join(DATA, sh), columns=["id", "text"])
        d = dict(zip(t.column("id").to_pylist(), t.column("text").to_pylist()))
        for i in g.id: txt[i] = d[i]
    lens = sorted(len(v) for v in txt.values())
    if lens:
        med = lens[len(lens) // 2]
        print(f"text lengths: min {lens[0]}, median {med}, max {lens[-1]}; "
              f"median coverage {min(1, MAX_CHUNKS * CHUNKLEN / max(1, med * 0.94)) * 100:.0f}%", flush=True)

    fresh = not os.path.exists(OUT_CSV)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fcsv = open(OUT_CSV, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fcsv, fieldnames=FIELDS)
    if fresh: w.writeheader()
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(CONCURRENCY)
    # bound books in flight: without this, every book's llama batch queues ahead of every book's
    # deepseek batch in the FIFO call semaphore, so no book completes (and no checkpoint row is
    # written) until the very end of the run
    book_sem = asyncio.Semaphore(16)
    n_done = len(done)
    t0 = time.time()

    async with httpx.AsyncClient() as c:
        await preflight(c)

        async def one(bid):
            nonlocal n_done
            async with book_sem:
                return await one_inner(bid)

        async def one_inner(bid):
            nonlocal n_done
            chs = chunks(txt.get(bid, ""))
            lres = await asyncio.gather(*[ask(c, sem, MODELS["llama"], p) for p in chs])
            dres = await asyncio.gather(*[ask(c, sem, MODELS["deepseek"], p, {"reasoning": {"enabled": False}}) for p in chs])
            lf = any(x is True for x in lres); df = any(x is True for x in dres)
            lerr = sum(x is None for x in lres); derr = sum(x is None for x in dres)
            verdict = ("drop_both_flag" if (lf and df) else
                       "review_single_flag" if (lf or df) else
                       "review_errors" if (lerr + derr) > 0 else "certified_clean")
            ev = next((chs[i][:70] for i, x in enumerate(lres) if x is True),
                      next((chs[i][:70] for i, x in enumerate(dres) if x is True), ""))
            row = {"id": bid, "n_chunks": len(chs), "llama_flag": lf, "deepseek_flag": df,
                   "llama_errors": lerr, "deepseek_errors": derr, "either": lf or df,
                   "both": lf and df, "verdict": verdict, "evidence": ev}
            async with lock:
                w.writerow(row); fcsv.flush()
                n_done += 1
                if n_done % 10 == 0:
                    rate = (n_done - len(done)) / max(1, time.time() - t0) * 60
                    print(f"  {n_done}/{len(cand)} certified ({rate:.0f} books/min)", flush=True)

        await asyncio.gather(*[one(b) for b in todo.id])
    fcsv.close()
    d = pd.read_csv(OUT_CSV)
    print(f"\nFULL CERTIFICATION: {len(d)} books, chunks up to {MAX_CHUNKS}x{CHUNKLEN} chars (~{COVERAGE:.0%} target coverage)")
    print(d.verdict.value_counts().to_string())
    print(f"flagged by llama70b: {int(d.llama_flag.sum())}  by deepseek-flash: {int(d.deepseek_flag.sum())}  both: {int(d.both.sum())}")

if __name__ == "__main__":
    asyncio.run(main())
