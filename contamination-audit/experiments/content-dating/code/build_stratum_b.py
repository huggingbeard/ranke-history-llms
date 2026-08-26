"""Build stratum B: 1930-1946 ORDINARY AUTHORED books (the "modern but reads like pre-29" hard positives).

v1 (2026-07-23), office 4090 box. PI: gov-doc positives are "a gov doc classifier by another name";
stratum B = near-boundary modern originals with ordinary prose. Pipeline:
  1. _band_1930_46.csv (full-corpus metadata scan, 6,329 eng non-reprint 1930-46 candidates)
  2. heuristic screen: drop institutional titles + corporate/absent authors
  3. deepseek-v4-flash title/author/year classification -> keep ordinary_authored_book
  4. fetch text_by_page_gen for keepers (grouped by parquet file, checkpointed)
  5. STRICT two-model certification (deepseek + Llama-70B, front matter + 3 body probes):
     keep only books BOTH models judge newly-written post-1929 content (not a reprint/edition of
     an older work). Positive-label purity beats yield: we have ~3.3k candidates for a ~600 target.
  6. manifest_pos3046.csv (repo) + texts -> Dropbox positives_text\\ + passages_pos3046.parquet
Costs: ~$0.25 title screen + ~$2-4 certification (OpenRouter). All steps checkpointed/resumable.
"""
import os, re, csv, json, asyncio, random, time
import pandas as pd
import httpx
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = os.path.dirname(os.path.abspath(__file__))
DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
POS_DIR = os.path.join(DROPBOX, "positives_text")
HF_TOKEN = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip()
OKEY = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\openrouter.txt").read().strip()
OURL = "https://openrouter.ai/api/v1/chat/completions"
TITLE_SCREEN_CSV = os.path.join(REPO, "_b_title_screen.csv")
CERT_CSV = os.path.join(REPO, "_b_certify.csv")
MANIFEST = os.path.join(REPO, "manifest_pos3046.csv")
TARGET = 600
SEED = 1929

GOV = re.compile(r"report|hearing|bulletin|proceedings|annual|biennial|statutes|laws|act of|acts of|commission|committee|board of|department|bureau|census|congress|senate|legislat|federal|municipal|ordinance|regulation|budget|treasury|comptroller|auditor|docket|circular|yearbook|year book|directory|register|minutes|session|statistical|assembly|code of|catalog", re.I)
CORP = re.compile(r"united states|u\.s\.|\(.*\)|county|city of|state of|university|college|society|association|committee|commission|company|corporation|inc\b|co\.|church|club|institute|bureau|board|league|line$|dept|laboratory|museum|library|bank|conference|council|union|federation|academy|school", re.I)

def heuristic_pool():
    d = pd.read_csv(os.path.join(REPO, "_band_1930_46.csv"), dtype=str)
    d = d[d.barcode != "__FILE_DONE__"].drop_duplicates("barcode")
    d = d[~d.title.fillna("").str.contains(GOV)]
    d = d[~d.genre.fillna("").str.lower().str.contains("periodical|serial|government|statistic|directorie|legislat")]
    d = d[d.author.fillna("").str.len() > 3]
    d = d[~d.author.fillna("").str.contains(CORP)]
    print(f"heuristic pool: {len(d)} candidates", flush=True)
    return d.sample(frac=1, random_state=SEED)   # seeded shuffle for unbiased order

async def ask_json(c, sem, model, prompt, extra=None, max_tokens=150):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}],
         "temperature": 0, "max_tokens": max_tokens, "provider": {"sort": "price"}}
    if extra: b.update(extra)
    async with sem:
        for a in range(4):
            try:
                r = await c.post(OURL, headers={"Authorization": f"Bearer {OKEY}"}, json=b, timeout=90)
                if r.status_code >= 500 or r.status_code == 429:
                    await asyncio.sleep(2 * (a + 1)); continue
                r.raise_for_status()
                ct = r.json()["choices"][0]["message"]["content"]
                if not ct or "{" not in ct:
                    await asyncio.sleep(2 * (a + 1)); continue
                return json.loads(ct[ct.find("{"):ct.rfind("}") + 1])
            except Exception:
                if a == 3: return None
                await asyncio.sleep(2 * (a + 1))
    return None

TITLE_PROMPT = ('Catalogue record: title: "{t}" | author: "{a}" | year: {y}. Classify this book. '
                'ordinary = a normally-authored book (fiction, poetry, essays, history, biography, religion, '
                'science, how-to) written by a person. NOT ordinary: government/agency/legislative documents, '
                'institutional or corporate reports, serials, directories, catalogues, proceedings. '
                'Return JSON: {{"ordinary": true/false}}')

async def title_screen(pool):
    done = {}
    if os.path.exists(TITLE_SCREEN_CSV):
        for r in csv.DictReader(open(TITLE_SCREEN_CSV, encoding="utf-8")):
            done[r["barcode"]] = r["ordinary"] == "True"
        print(f"resume title screen: {len(done)} done", flush=True)
    todo = pool[~pool.barcode.isin(done)]
    fresh = not os.path.exists(TITLE_SCREEN_CSV)
    f = open(TITLE_SCREEN_CSV, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["barcode", "ordinary"])
    if fresh: w.writeheader()
    sem = asyncio.Semaphore(32)
    lock = asyncio.Lock()
    async with httpx.AsyncClient() as c:
        async def one(row):
            v = await ask_json(c, sem, "deepseek/deepseek-v4-flash",
                               TITLE_PROMPT.format(t=row.title, a=row.author, y=row.year),
                               {"reasoning": {"enabled": False}})
            ok = bool(v and v.get("ordinary"))
            async with lock:
                w.writerow({"barcode": row.barcode, "ordinary": ok}); f.flush()
            done[row.barcode] = ok
        await asyncio.gather(*[one(r) for r in todo.itertuples()])
    f.close()
    keep = pool[pool.barcode.map(done).fillna(False)]
    print(f"title screen: {len(keep)}/{len(pool)} ordinary", flush=True)
    return keep

def fetch_texts(cands):
    fs = HfFileSystem(token=HF_TOKEN)
    os.makedirs(POS_DIR, exist_ok=True)
    have = {c.barcode for c in cands.itertuples() if os.path.exists(os.path.join(POS_DIR, c.barcode + ".txt"))}
    todo = cands[~cands.barcode.isin(have)]
    print(f"fetch: {len(have)} cached, {len(todo)} to fetch from {todo.file.nunique()} files", flush=True)
    for path, grp in todo.groupby("file"):
        for a in range(4):
            try:
                with fs.open(path) as fh:
                    t = pq.ParquetFile(fh).read(columns=["barcode_src", "text_by_page_gen"])
                break
            except Exception:
                if a == 3: raise
                time.sleep(3 * (a + 1))
        d = t.to_pydict()
        want = set(grp.barcode)
        for bid, pages in zip(d["barcode_src"], d["text_by_page_gen"]):
            if bid in want and pages:
                tx = "\n\n".join(p for p in pages if p and p.strip())
                if len(tx) >= 3000:
                    with open(os.path.join(POS_DIR, bid + ".txt"), "w", encoding="utf-8") as f:
                        f.write(tx)

CERT_PROMPT = ('This book was catalogued with publication year {y}. Below are excerpts: first its opening '
               '(front matter), then samples from the body. Question: is the MAIN TEXT newly written in 1929 '
               'or later (a modern original), or is it a reprint/new edition/translation/anthology of work '
               'written BEFORE 1929? Look for reprint/edition statements, archaic composition dates, or '
               'content that could only be post-1929.\n{ex}\n'
               'Return JSON: {{"modern_original": true/false, "why": "<=10 words"}}')

async def certify(cands):
    done = {}
    if os.path.exists(CERT_CSV):
        for r in csv.DictReader(open(CERT_CSV, encoding="utf-8")):
            done[r["barcode"]] = r
        print(f"resume certify: {len(done)} done", flush=True)
    kept = sum(1 for r in done.values() if r["verdict"] == "keep")
    fresh = not os.path.exists(CERT_CSV)
    f = open(CERT_CSV, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["barcode", "deepseek", "llama", "verdict"])
    if fresh: w.writeheader()
    sem = asyncio.Semaphore(24)
    lock = asyncio.Lock()
    book_sem = asyncio.Semaphore(12)
    async with httpx.AsyncClient() as c:
        async def one(row):
            nonlocal kept
            if kept >= TARGET: return
            p = os.path.join(POS_DIR, row.barcode + ".txt")
            if not os.path.exists(p): return
            tx = open(p, encoding="utf-8").read()
            body = [" ".join(tx[k:k + 3000].split()) for k in
                    (len(tx) // 4, len(tx) // 2, 3 * len(tx) // 4)]
            ex = ("FRONT MATTER:\n" + " ".join(tx[:6000].split()) +
                  "\n\nBODY SAMPLES:\n" + "\n---\n".join(body))
            prompt = CERT_PROMPT.format(y=row.year, ex=ex[:24000])
            async with book_sem:
                ds = await ask_json(c, sem, "deepseek/deepseek-v4-flash", prompt, {"reasoning": {"enabled": False}})
                ll = await ask_json(c, sem, "meta-llama/llama-3.3-70b-instruct", prompt)
            dv = bool(ds and ds.get("modern_original")); lv = bool(ll and ll.get("modern_original"))
            verdict = "keep" if (dv and lv) else "drop"
            async with lock:
                if row.barcode not in done:
                    w.writerow({"barcode": row.barcode, "deepseek": dv, "llama": lv, "verdict": verdict})
                    f.flush(); done[row.barcode] = {"verdict": verdict}
                    if verdict == "keep":
                        kept += 1
                        if kept % 50 == 0: print(f"  certified keep: {kept}/{TARGET}", flush=True)
        todo = [r for r in cands.itertuples() if r.barcode not in done]
        await asyncio.gather(*[one(r) for r in todo])
    f.close()
    print(f"certify done: {kept} kept of {len(done)} examined", flush=True)
    return {b for b, r in done.items() if r["verdict"] == "keep"}

def main():
    pool = heuristic_pool()
    keep = asyncio.run(title_screen(pool))
    # fetch + certify in TARGET-sized waves so we do not fetch 3k books' text for a 600 target
    random.seed(SEED)
    kept_ids = set()
    wave = 0
    while len(kept_ids) < TARGET and wave * 900 < len(keep):
        batch = keep.iloc[wave * 900:(wave + 1) * 900]
        fetch_texts(batch)
        kept_ids |= asyncio.run(certify(batch))
        wave += 1
        print(f"after wave {wave}: {len(kept_ids)} certified positives", flush=True)
    sel = keep[keep.barcode.isin(kept_ids)].head(TARGET)
    rows = [{"id": r.barcode, "source": "institutional_books_hf", "meta_year": r.year,
             "label": 1, "cls": "B_1930_46_ordinary", "origin": "hf_stream"} for r in sel.itertuples()]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "source", "meta_year", "label", "cls", "origin"])
        w.writeheader(); w.writerows(rows)
    print(f"manifest_pos3046.csv: {len(rows)} stratum-B positives", flush=True)
    import extract_passages_v2 as ex
    ex.run("manifest_pos3046.csv", "passages_pos3046.parquet")

if __name__ == "__main__":
    main()
