"""Full metadata scan of institutional-books-1.0 for the 1930-1946 English band.
2026-07-22, office box. Read-only: metadata columns of all 9,831 parquet files, checkpointed.
Output: _band_1930_46.csv (barcode, year, title, author, genre, date_types, rights hint, tokens, file).
Purpose: measure and enumerate the near-boundary positive pool (PI question: positives currently
gov-doc-heavy post-1947; the hard cases are 1930-46 'ordinary' books that read like pre-29 prose).
"""
import os, re, csv, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, HfApi

HF_TOKEN = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip()
REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, "_band_1930_46.csv")
COLS = ["barcode_src", "title_src", "author_src", "date1_src", "date2_src", "date_types_src",
        "language_gen", "genre_or_form_src", "hathitrust_data_ext", "token_count_o200k_base_gen"]
YR = re.compile(r"(1[5-9]\d\d|20[0-2]\d)")
fs = HfFileSystem(token=HF_TOKEN)

def scan(path):
    for a in range(4):
        try:
            with fs.open(path) as fh:
                t = pq.ParquetFile(fh).read(columns=COLS)
            break
        except Exception:
            if a == 3: raise
            time.sleep(3 * (a + 1))
    d = t.to_pydict(); out = []
    for i in range(t.num_rows):
        m = YR.search(str(d["date1_src"][i] or ""))
        if not m: continue
        y = int(m.group(1))
        if not (1930 <= y <= 1946): continue
        if d["language_gen"][i] != "eng": continue
        if (d["token_count_o200k_base_gen"][i] or 0) < 1500: continue
        m2 = YR.search(str(d["date2_src"][i] or ""))
        if m2 and int(m2.group(1)) < 1930: continue
        dt = str(d["date_types_src"][i] or "")
        if "reprint" in dt.lower(): continue
        ht = str(d["hathitrust_data_ext"][i] or "")
        mrc = re.search(r'rights_code[":\s]+(\w+)', ht)
        out.append({"file": path, "barcode": d["barcode_src"][i], "year": y,
                    "title": str(d["title_src"][i] or "")[:150],
                    "author": str(d["author_src"][i] or "")[:80],
                    "genre": str(d["genre_or_form_src"][i] or "")[:80],
                    "date_types": dt[:60], "rights": mrc.group(1) if mrc else "",
                    "tokens": d["token_count_o200k_base_gen"][i]})
    return path, t.num_rows, out

FIELDS = ["file", "barcode", "year", "title", "author", "genre", "date_types", "rights", "tokens"]
files = sorted(f for f in HfApi(token=HF_TOKEN).list_repo_files(
    "institutional/institutional-books-1.0", repo_type="dataset") if f.endswith(".parquet"))
files = [f"datasets/institutional/institutional-books-1.0/{f}" for f in files]
done = set()
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["barcode"] == "__FILE_DONE__": done.add(r["file"])
    print(f"resume: {len(done)} files scanned", flush=True)
fresh = not os.path.exists(OUT)
fcsv = open(OUT, "a", newline="", encoding="utf-8")
w = csv.DictWriter(fcsv, fieldnames=FIELDS)
if fresh: w.writeheader()
todo = [f for f in files if f not in done]
n_band = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(scan, p): p for p in todo}
    ndone = len(done)
    for fut in as_completed(futs):
        try:
            path, n, out = fut.result()
        except Exception as e:
            print(f"FAIL {futs[fut]}: {str(e)[:80]}", flush=True); continue
        for c in out: w.writerow(c)
        w.writerow({"file": path, "barcode": "__FILE_DONE__", "year": n, "title": "", "author": "",
                    "genre": "", "date_types": "", "rights": "", "tokens": 0})
        fcsv.flush()
        n_band += len(out); ndone += 1
        if ndone % 500 == 0:
            print(f"{ndone}/{len(files)} files, {n_band} band candidates so far", flush=True)
fcsv.close()
print(f"DONE: {n_band} candidates in band from this run", flush=True)
