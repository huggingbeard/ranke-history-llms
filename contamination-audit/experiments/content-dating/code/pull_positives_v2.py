"""Pull 1,000 DEFINITELY-post-1929 modern-original positives from institutional/institutional-books-1.0.

v2 (2026-07-22) — schema-verified rewrite of pull_positives.py, run on the office 4090 box.
Files used:
  HF dataset:  institutional/institutional-books-1.0 (983,004 rows, 9,831 parquet files, 100 rows/file)
  token:       D:\\UZHechist Dropbox\\Joachim Voth\\claudecode\\api\\HF.txt
  output text: D:\\UZHechist Dropbox\\Joachim Voth\\claudecode\\ranke-contamination-audit\\experiments\\content-dating\\positives_text\\
  manifest:    <this repo dir>\\manifest_positives.csv
  scan cache:  <this repo dir>\\_scan_candidates.csv  (checkpoint; safe to delete to force rescan)

Why v2 (verified against the live schema on 2026-07-22):
  - text field is `text_by_page_gen`: list<large_string> of pages (NOT a flat string as v1 assumed;
    v1's auto-detect would have picked the JSON-ish repr and written garbage).
  - year field is `date1_src` (string); `date2_src` + `date_types_src` guard reprint/reissue dates.
  - id field is `barcode_src`.
  - added language_gen == 'eng' filter: the 999 negatives are all English (metadata language 'en');
    without this, language would separate the classes.
  - gen (line-merged OCR) matches the negatives' text style; pages joined with '\n\n'
    (newlines are flattened at passage extraction anyway).
Strategy (dataset is ~1 TB — cannot stream blindly):
  Phase A: read ONLY metadata columns from seeded-shuffled files until >=1,500 candidates.
  Phase B: read text_by_page_gen only from contributing files, apply reprint-regex + min-length,
           write .txt per book until N_TARGET kept. Both phases checkpointed/resumable.
"""
import os, re, csv, sys, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, HfApi

HF_TOKEN = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip()
DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
REPO = os.path.dirname(os.path.abspath(__file__))
DATASET = "institutional/institutional-books-1.0"
N_TARGET = 1000
N_CANDIDATES = 1500          # margin over N_TARGET for reprint/length drops
MIN_YEAR = 1947              # definitely post-1929, past WWII boundary noise
MAX_YEAR = 2026
MIN_TOKENS = 1500            # metadata pre-filter; real min-length check is on chars in phase B
MIN_CHARS = 3000
SEED = 1929
META_COLS = ["barcode_src", "title_src", "author_src", "date1_src", "date2_src",
             "date_types_src", "language_gen", "token_count_o200k_base_gen"]
REPRINT = re.compile(r"reprint(ed)? of|facsimile|first published 1[5-9]\d\d|originally published|new impression|photolith", re.I)
YEAR_RE = re.compile(r"(1[5-9]\d\d|20[0-2]\d)")
SCAN_CSV = os.path.join(REPO, "_scan_candidates.csv")
MANIFEST = os.path.join(REPO, "manifest_positives.csv")
POS_DIR = os.path.join(DROPBOX, "positives_text")
os.makedirs(POS_DIR, exist_ok=True)

fs = HfFileSystem(token=HF_TOKEN)

def year_of(s):
    m = YEAR_RE.search(str(s or ""))
    return int(m.group(1)) if m else None

def scan_file(path):
    """Read metadata columns of one parquet file; return (path, n_rows, candidate rows)."""
    for attempt in range(4):
        try:
            with fs.open(path) as fh:
                t = pq.ParquetFile(fh).read(columns=META_COLS)
            break
        except Exception:
            if attempt == 3: raise
            time.sleep(3 * (attempt + 1))
    d = t.to_pydict()
    out = []
    for i in range(t.num_rows):
        y1 = year_of(d["date1_src"][i]); y2 = year_of(d["date2_src"][i])
        dtypes = str(d["date_types_src"][i] or "").lower()
        toks = d["token_count_o200k_base_gen"][i] or 0
        if y1 is None or not (MIN_YEAR <= y1 <= MAX_YEAR): continue
        if y2 is not None and y2 < MIN_YEAR: continue          # reissue of an older original
        if "reprint" in dtypes or "reissue" in dtypes: continue
        if d["language_gen"][i] != "eng": continue
        if toks < MIN_TOKENS: continue
        out.append({"file": path, "barcode": d["barcode_src"][i], "year": y1,
                    "title": (d["title_src"][i] or "")[:120], "tokens": toks})
    return path, t.num_rows, out

def phase_a():
    files = sorted(f for f in HfApi(token=HF_TOKEN).list_repo_files(DATASET, repo_type="dataset")
                   if f.endswith(".parquet"))
    files = [f"datasets/{DATASET}/{f}" for f in files]
    random.Random(SEED).shuffle(files)
    done_files, cands, rows_scanned = set(), [], 0
    if os.path.exists(SCAN_CSV):
        with open(SCAN_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["barcode"] == "__FILE_DONE__":
                    done_files.add(r["file"]); rows_scanned += int(r["year"])
                else:
                    cands.append(r)
        print(f"resume: {len(done_files)} files scanned, {len(cands)} candidates", flush=True)
    fresh = not os.path.exists(SCAN_CSV)
    fcsv = open(SCAN_CSV, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fcsv, fieldnames=["file", "barcode", "year", "title", "tokens"])
    if fresh: w.writeheader()
    todo = [f for f in files if f not in done_files]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {}
        it = iter(todo)
        for _ in range(16):
            p = next(it, None)
            if p: futs[ex.submit(scan_file, p)] = p
        while futs:
            fut = next(as_completed(futs)); futs.pop(fut)
            try:
                path, n, out = fut.result()
            except Exception as e:
                print(f"  SCAN FAIL {fut}: {str(e)[:100]}", flush=True)
                path, n, out = None, 0, []
            if path:
                for c in out: w.writerow(c)
                # sentinel row records the file as fully scanned (year column reuses n_rows)
                w.writerow({"file": path, "barcode": "__FILE_DONE__", "year": n, "title": "", "tokens": 0})
                fcsv.flush()
                cands.extend(out); rows_scanned += n
            if len(cands) >= N_CANDIDATES:
                break
            p = next(it, None)
            if p: futs[ex.submit(scan_file, p)] = p
    fcsv.close()
    nf = len({c["file"] for c in cands})
    print(f"PHASE A done: {len(cands)} candidates from {nf} files; "
          f"hit rate {len(cands)}/{rows_scanned} = {100*len(cands)/max(1,rows_scanned):.1f}%", flush=True)
    return cands

def fetch_file_texts(path, barcodes):
    """Read text_by_page_gen for the given barcodes from one file."""
    for attempt in range(4):
        try:
            with fs.open(path) as fh:
                t = pq.ParquetFile(fh).read(columns=["barcode_src", "text_by_page_gen"])
            break
        except Exception:
            if attempt == 3: raise
            time.sleep(3 * (attempt + 1))
    d = t.to_pydict()
    want = set(barcodes)
    out = {}
    for bid, pages in zip(d["barcode_src"], d["text_by_page_gen"]):
        if bid in want and pages:
            out[bid] = "\n\n".join(p for p in pages if p and p.strip())
    return out

def phase_b(cands):
    kept = []
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            kept = list(csv.DictReader(f))
        print(f"resume: {len(kept)} positives already in manifest", flush=True)
    kept_ids = {r["id"] for r in kept}
    by_file = {}
    for c in cands:
        by_file.setdefault(c["file"], []).append(c)
    dropped_reprint = dropped_short = 0
    for path, cs in by_file.items():
        need = [c for c in cs if c["barcode"] not in kept_ids]
        if not need: continue
        if len(kept) >= N_TARGET: break
        try:
            texts = fetch_file_texts(path, [c["barcode"] for c in need])
        except Exception as e:
            print(f"  FETCH FAIL {path}: {str(e)[:100]}", flush=True)
            continue
        for c in need:
            if len(kept) >= N_TARGET: break
            tx = texts.get(c["barcode"], "")
            if len(tx) < MIN_CHARS:
                dropped_short += 1; continue
            if REPRINT.search(tx[:6000]):
                dropped_reprint += 1; continue
            with open(os.path.join(POS_DIR, c["barcode"] + ".txt"), "w", encoding="utf-8") as f:
                f.write(tx)
            kept.append({"id": c["barcode"], "source": "institutional_books_hf",
                         "meta_year": c["year"], "label": 1, "cls": "modern_post1929", "origin": "hf_stream"})
            kept_ids.add(c["barcode"])
        with open(MANIFEST, "w", newline="", encoding="utf-8") as f:  # checkpoint after every file
            w = csv.DictWriter(f, fieldnames=["id", "source", "meta_year", "label", "cls", "origin"])
            w.writeheader(); w.writerows(kept)
        if len(kept) % 50 < len(need):
            print(f"  kept {len(kept)}/{N_TARGET} (reprint-dropped {dropped_reprint}, short-dropped {dropped_short})", flush=True)
    print(f"PHASE B done: {len(kept)} positives; dropped {dropped_reprint} reprint-regex, {dropped_short} too-short", flush=True)
    if kept:
        years = sorted(int(r["meta_year"]) for r in kept)
        print(f"year range: {years[0]}–{years[-1]}, median {years[len(years)//2]}", flush=True)

if __name__ == "__main__":
    cands = phase_a()
    phase_b(cands)
    print("DONE", flush=True)
