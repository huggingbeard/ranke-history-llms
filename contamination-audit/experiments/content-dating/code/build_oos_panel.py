"""Build the OOS validation panel per PREREGISTRATION_OOS.md (frozen protocol, unseen docs only).

v1 (2026-07-24). Phases (checkpointed, resumable):
  1. fresh negatives: rebuild the build_negatives clean pool (same five screens), EXCLUDE every
     previously touched id (999 negatives, 2,292 gold-audited, contaminants/gotthrough buckets,
     contamC), sample 400 (seed 2440), certify with the EXACT certify_negatives_v2 protocol
     (deepseek + Llama-70B, 24x3000 chars, error-aware) -> keep certified_clean.
  2. fresh honest moderns: 1930-46 ordinary pool minus the 899 certification-examined; fetch text,
     certify modern-original (EXACT build_stratum_b prompt); target 400 certified ->
     first 250 (seeded) = honest-eval stratum, next 150 = plant parents (disjoint).
  3. fresh corrupted plants from the 150 parents: FROZEN build_plants operators (P1/P2, seed 1929),
     G1 imprint gate enforced.
Outputs (repo): manifest_oos_negatives.csv (certified only), manifest_oos_moderns.csv,
  manifest_oos_plants.csv; texts -> Dropbox positives_text\\ and plants_text\\; certification
  verdicts -> intermediates\\certify_oos_*.csv.
"""
import os, sys, re, csv, json, math, time, random, asyncio
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import httpx

sys.path.insert(0, r"C:\Users\hvoth\Documents\GitHub\ranke-edition-dating")
from freeze_pipeline import imprint_year

REPO = os.path.dirname(os.path.abspath(__file__))
DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
I = os.path.join(DROPBOX, "intermediates")
CORPUS = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\temp-guohui\decontamination\datasets\history-llms\sample-10gb\data"
POS_DIR = os.path.join(DROPBOX, "positives_text")
PLANTS_DIR = os.path.join(DROPBOX, "plants_text")
OKEY = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\openrouter.txt").read().strip()
HF_TOKEN = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip()
OURL = "https://openrouter.ai/api/v1/chat/completions"
SEED_NEG = 2440
N_NEG, N_MOD_EVAL, N_MOD_PLANT = 400, 250, 150

# ---- EXACT certify_negatives_v2 protocol constants (frozen) ----
NEG_PROMPT = ('This passage is from a book catalogued as pre-1929. Does it contain content or knowledge that '
              'could only exist AFTER 1929 (a post-1929 event, person, technology, dated reference, or clearly '
              'modern perspective)? Old, foreign-language, poetic, or historical text is NOT post-1929. Judge '
              'the writing, ignore OCR noise. Passage:\n"""{p}"""\n'
              'Return JSON: {{"post1929": true/false, "why": "<=8 words"}}')
CHUNKLEN, MAX_CHUNKS, COVERAGE = 3000, 24, 0.70
# ---- EXACT build_stratum_b modern-original protocol (frozen) ----
MOD_PROMPT = ('This book was catalogued with publication year {y}. Below are excerpts: first its opening '
              '(front matter), then samples from the body. Question: is the MAIN TEXT newly written in 1929 '
              'or later (a modern original), or is it a reprint/new edition/translation/anthology of work '
              'written BEFORE 1929? Look for reprint/edition statements, archaic composition dates, or '
              'content that could only be post-1929.\n{ex}\n'
              'Return JSON: {{"modern_original": true/false, "why": "<=10 words"}}')

def neg_chunks(tx):
    lo, hi = int(len(tx) * 0.03), int(len(tx) * 0.97)
    span = hi - lo
    if span <= CHUNKLEN: return [" ".join(tx.split())]
    n = min(MAX_CHUNKS, max(1, math.ceil(span * COVERAGE / CHUNKLEN)))
    step = max(CHUNKLEN, span // n)
    return [" ".join(tx[p:p + CHUNKLEN].split()) for p in range(lo, hi - CHUNKLEN + 1, step)][:MAX_CHUNKS]

async def ask_json(c, sem, model, prompt, extra=None):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}],
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
                if not ct or "{" not in ct:
                    await asyncio.sleep(2 * (a + 1)); continue
                return json.loads(ct[ct.find("{"):ct.rfind("}") + 1])
            except Exception:
                if a == 3: return None
                await asyncio.sleep(3 * (a + 1))
    return None

def touched_ids():
    ids = set()
    for f in ("manifest_negatives.csv", "manifest_contamC.csv"):
        ids |= set(pd.read_csv(os.path.join(REPO, f), dtype={"id": str}).id)
    for n in ("gold_300_verdicts", "gold_1000_verdicts", "gold_1000b_verdicts",
              "contaminants_bucket", "genuine_gotthrough_bucket"):
        p = os.path.join(I, n + ".csv")
        if os.path.exists(p): ids |= set(pd.read_csv(p, dtype={"id": str}).id)
    return ids

def corpus_texts(df):
    txt = {}
    for sh, g in df.groupby("shard"):
        t = pq.read_table(os.path.join(CORPUS, sh), columns=["id", "text"])
        t = t.filter(pc.is_in(t.column("id"), value_set=pa.array(list(g.id))))
        for i, tx in zip(t.column("id").to_pylist(), t.column("text").to_pylist()):
            txt[i] = tx or ""
    return txt

# ---------------- Phase 1: fresh negatives ----------------
def sample_fresh_negative_candidates():
    meta = pd.read_parquet(os.path.join(I, "corpus_meta.parquet"))
    v2 = pd.read_parquet(os.path.join(I, "fullrun_v2.parquet"))[["id", "imprint_v2"]]
    v3 = pd.read_parquet(os.path.join(I, "fullrun_v3.parquet"))[["id", "n_datectx_post"]]
    hits = set(pd.read_csv(os.path.join(I, "clean_check9_hits.csv"), dtype={"id": str}).id)
    d = meta.merge(v2, on="id", how="left").merge(v3, on="id", how="left")
    clean = d[(d.source == "institutional_books") & (d.year <= 1929) & (d.stage2_docyear <= 1929)
              & ((d.imprint_v2.isna()) | (d.imprint_v2 <= 1929)) & (d.n_datectx_post.fillna(0) == 0)
              & (~d.id.isin(hits))]
    clean = clean[~clean.id.isin(touched_ids())]
    print(f"phase1: untouched clean pool {len(clean)}", flush=True)
    return clean.sample(N_NEG, random_state=SEED_NEG)[["id", "year", "stage2_docyear", "shard"]]

async def certify_negatives(cand, txt):
    out_csv = os.path.join(I, "certify_oos_negatives.csv")
    done = set()
    if os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv, dtype={"id": str}).id)
        print(f"phase1 resume: {len(done)}", flush=True)
    fresh = not os.path.exists(out_csv)
    f = open(out_csv, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["id", "llama_flag", "deepseek_flag", "errors", "verdict"])
    if fresh: w.writeheader()
    sem = asyncio.Semaphore(48); book_sem = asyncio.Semaphore(16); lock = asyncio.Lock()
    async with httpx.AsyncClient() as c:
        async def one(bid):
            async with book_sem:
                chs = neg_chunks(txt.get(bid, ""))
                lres = await asyncio.gather(*[ask_json(c, sem, "meta-llama/llama-3.3-70b-instruct",
                                                       NEG_PROMPT.format(p=p)) for p in chs])
                dres = await asyncio.gather(*[ask_json(c, sem, "deepseek/deepseek-v4-flash",
                                                       NEG_PROMPT.format(p=p), {"reasoning": {"enabled": False}}) for p in chs])
                lf = any(v and v.get("post1929") for v in lres)
                df_ = any(v and v.get("post1929") for v in dres)
                err = sum(v is None for v in lres) + sum(v is None for v in dres)
                verdict = ("drop" if (lf and df_) else "review" if (lf or df_) else
                           "review_errors" if err else "certified_clean")
                async with lock:
                    w.writerow({"id": bid, "llama_flag": lf, "deepseek_flag": df_, "errors": err, "verdict": verdict})
                    f.flush()
        await asyncio.gather(*[one(b) for b in cand.id if b not in done])
    f.close()
    v = pd.read_csv(out_csv, dtype={"id": str})
    print(f"phase1 verdicts: {v.verdict.value_counts().to_dict()}", flush=True)
    return set(v[v.verdict == "certified_clean"].id)

# ---------------- Phase 2: fresh moderns ----------------
def fresh_modern_candidates():
    band = pd.read_csv(os.path.join(REPO, "_band_1930_46.csv"), dtype=str)
    band = band[band.barcode != "__FILE_DONE__"].drop_duplicates("barcode")
    screen = pd.read_csv(os.path.join(REPO, "_b_title_screen.csv"), dtype=str)
    ordinary = set(screen[screen.ordinary == "True"].barcode)
    examined = set(pd.read_csv(os.path.join(REPO, "_b_certify.csv"), dtype=str).barcode)
    pool = band[band.barcode.isin(ordinary) & ~band.barcode.isin(examined)]
    print(f"phase2: untouched ordinary pool {len(pool)}", flush=True)
    return pool.sample(frac=1, random_state=SEED_NEG)

def fetch_hf_texts(cands):
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=HF_TOKEN)
    todo = cands[~cands.barcode.map(lambda b: os.path.exists(os.path.join(POS_DIR, b + ".txt")))]
    print(f"phase2 fetch: {len(todo)} texts from {todo.file.nunique()} files", flush=True)
    for path, grp in todo.groupby("file"):
        for a in range(4):
            try:
                with fs.open(path) as fh:
                    t = pq.ParquetFile(fh).read(columns=["barcode_src", "text_by_page_gen"])
                break
            except Exception:
                if a == 3: raise
                time.sleep(3 * (a + 1))
        d = t.to_pydict(); want = set(grp.barcode)
        for bid, pages in zip(d["barcode_src"], d["text_by_page_gen"]):
            if bid in want and pages:
                tx = "\n\n".join(p for p in pages if p and p.strip())
                if len(tx) >= 3000:
                    with open(os.path.join(POS_DIR, bid + ".txt"), "w", encoding="utf-8") as f:
                        f.write(tx)

async def certify_moderns(cands, target):
    out_csv = os.path.join(I, "certify_oos_moderns.csv")
    done = {}
    if os.path.exists(out_csv):
        for r in csv.DictReader(open(out_csv, encoding="utf-8")): done[r["barcode"]] = r["verdict"]
        print(f"phase2 resume: {len(done)}", flush=True)
    kept = sum(1 for v in done.values() if v == "keep")
    fresh = not os.path.exists(out_csv)
    f = open(out_csv, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["barcode", "verdict"])
    if fresh: w.writeheader()
    sem = asyncio.Semaphore(32); book_sem = asyncio.Semaphore(12); lock = asyncio.Lock()
    async with httpx.AsyncClient() as c:
        async def one(row):
            nonlocal kept
            if kept >= target or row.barcode in done: return
            p = os.path.join(POS_DIR, row.barcode + ".txt")
            if not os.path.exists(p): return
            tx = open(p, encoding="utf-8").read()
            body = [" ".join(tx[k:k + 3000].split()) for k in (len(tx) // 4, len(tx) // 2, 3 * len(tx) // 4)]
            ex = ("FRONT MATTER:\n" + " ".join(tx[:6000].split()) + "\n\nBODY SAMPLES:\n" + "\n---\n".join(body))
            prompt = MOD_PROMPT.format(y=row.year, ex=ex[:24000])
            async with book_sem:
                ds = await ask_json(c, sem, "deepseek/deepseek-v4-flash", prompt, {"reasoning": {"enabled": False}})
                ll = await ask_json(c, sem, "meta-llama/llama-3.3-70b-instruct", prompt)
            v = "keep" if (ds and ds.get("modern_original") and ll and ll.get("modern_original")) else "drop"
            async with lock:
                if row.barcode not in done:
                    w.writerow({"barcode": row.barcode, "verdict": v}); f.flush()
                    done[row.barcode] = v
                    if v == "keep":
                        kept += 1
                        if kept % 50 == 0: print(f"  phase2 keep {kept}/{target}", flush=True)
        await asyncio.gather(*[one(r) for r in cands.itertuples()])
    f.close()
    return [b for b, v in done.items() if v == "keep"]

# ---------------- Phase 3: plants (frozen operators) ----------------
def build_oos_plants(parents_df):
    os.makedirs(PLANTS_DIR, exist_ok=True)
    rng = random.Random(1929)
    neg = pd.read_csv(os.path.join(REPO, "manifest_negatives.csv"), dtype={"id": str})
    neg_years = pd.to_numeric(neg.meta_year).tolist()
    # genuine provenance blocks: same frozen harvest as build_plants.py (seed 1929, certified negs)
    neg_s = neg.sample(220, random_state=1929)
    neg_txt = corpus_texts(neg_s)
    PROV_WORDS = ("library", "legacy", "bequest", "gift", "estate", "fund", "college", "received")
    prov = []
    for i, tx in neg_txt.items():
        if len(tx) > 5000 and imprint_year(tx) is None:
            first_para = tx[:1200].split("\n\n")[0]
            if 120 < len(first_para) < 900 and any(w in first_para.lower() for w in PROV_WORDS):
                prov.append(first_para.strip())
    print(f"phase3: {len(prov)} genuine provenance blocks harvested", flush=True)
    rows, skipped = [], 0
    for r in parents_df.itertuples():
        p = os.path.join(POS_DIR, r.barcode + ".txt")
        tx = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
        if len(tx) < 20000: continue
        lo, hi = int(len(tx) * 0.02), int(len(tx) * 0.05)
        cut = tx.find("\n\n", rng.randint(lo, hi))
        if cut < 0: cut = hi
        body = tx[cut:].lstrip()
        pattern = "P1_truncate"
        if prov and rng.random() < 0.4:
            body = rng.choice(prov) + "\n\n" + body
            pattern = "P2_provenance"
        iy = imprint_year(body)
        if iy is not None and iy > 1929:
            skipped += 1; continue
        pid = r.barcode + "_OP"
        with open(os.path.join(PLANTS_DIR, pid + ".txt"), "w", encoding="utf-8") as f:
            f.write(body)
        rows.append({"plant_id": pid, "parent_id": r.barcode, "pattern": pattern,
                     "fake_meta_year": int(rng.choice(neg_years)), "true_year": r.year,
                     "imprint_after": iy if iy is not None else ""})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(REPO, "manifest_oos_plants.csv"), index=False)
    print(f"phase3: {len(out)} OOS plants ({skipped} skipped, G1); imprint-free "
          f"{(out.imprint_after == '').mean():.0%}", flush=True)
    return out

def main():
    os.makedirs(POS_DIR, exist_ok=True)
    cand = sample_fresh_negative_candidates()
    txt = corpus_texts(cand)
    certified = asyncio.run(certify_negatives(cand, txt))
    cand[cand.id.isin(certified)].rename(columns={"year": "meta_year"}).assign(label=0).to_csv(
        os.path.join(REPO, "manifest_oos_negatives.csv"), index=False)
    print(f"phase1 DONE: {len(certified)} certified fresh negatives", flush=True)

    pool = fresh_modern_candidates()
    need = int((N_MOD_EVAL + N_MOD_PLANT) / 0.65) + 60
    batch = pool.head(need)
    fetch_hf_texts(batch)
    keep = asyncio.run(certify_moderns(batch, N_MOD_EVAL + N_MOD_PLANT))
    keep_df = batch[batch.barcode.isin(keep)].reset_index(drop=True)
    ev = keep_df.head(N_MOD_EVAL)
    par = keep_df.iloc[N_MOD_EVAL:N_MOD_EVAL + N_MOD_PLANT]
    ev.assign(label=1).rename(columns={"barcode": "id", "year": "meta_year"})[
        ["id", "meta_year", "label"]].to_csv(os.path.join(REPO, "manifest_oos_moderns.csv"), index=False)
    print(f"phase2 DONE: {len(ev)} honest moderns, {len(par)} plant parents", flush=True)
    build_oos_plants(par)
    print("PANEL COMPLETE", flush=True)

if __name__ == "__main__":
    main()
