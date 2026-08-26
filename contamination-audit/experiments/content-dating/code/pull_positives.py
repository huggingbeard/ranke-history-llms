"""Stream institutional_books from HuggingFace, keep 1,000 DEFINITELY-post-1929 MODERN-ORIGINAL books.
Filter: catalogue year >= 1947 (well past the cutoff, no boundary ambiguity) AND front matter shows no
reprint/facsimile language (so we don't admit a modern reprint of an old work into the positive class).
Writes text to Dropbox and a manifest to the repo. Run here or on the office PC (needs HF token).

VERIFY BEFORE RUN: dataset id + field names below — institutional-books schema may differ; the script
prints the first record's keys so you can adjust YEAR_FIELD / TEXT_FIELD.
"""
import os, re, json, csv
from datasets import load_dataset
HF_TOKEN=open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip()
DROPBOX=r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
REPO=os.path.dirname(os.path.abspath(__file__))
DATASET="institutional/institutional-books-1.0"   # verify
YEAR_FIELD=None   # auto-detect from first record (looks for a field with a 4-digit year)
TEXT_FIELD=None
N_TARGET=1000
MIN_YEAR=1947     # definitely post-1929, past WWII boundary noise
REPRINT=re.compile(r"reprint(ed)? of|facsimile|first published 1[5-9]\d\d|originally published|new impression|photolith", re.I)

def detect_fields(rec):
    tf=next((k for k in rec if isinstance(rec[k],str) and len(rec[k])>500), None)
    yf=None
    for k,v in rec.items():
        s=str(v)
        if re.fullmatch(r"(1[5-9]\d\d|20[0-2]\d)", s) or (k.lower() in ("year","date","publication_year","pubdate","imprint_year")):
            yf=k; break
    return tf,yf

def year_of(rec, yf):
    m=re.search(r"(1[5-9]\d\d|20[0-2]\d)", str(rec.get(yf,"")))
    return int(m.group(1)) if m else None

ds=load_dataset(DATASET, split="train", streaming=True, token=HF_TOKEN)
kept=[]; scanned=0
os.makedirs(os.path.join(DROPBOX,"positives_text"), exist_ok=True)
for rec in ds:
    scanned+=1
    if TEXT_FIELD is None:
        TEXT_FIELD,YEAR_FIELD=detect_fields(rec); print("fields:",list(rec.keys())); print("TEXT_FIELD=",TEXT_FIELD,"YEAR_FIELD=",YEAR_FIELD, flush=True)
    y=year_of(rec, YEAR_FIELD); tx=rec.get(TEXT_FIELD) or ""
    if y is None or y<MIN_YEAR or len(tx)<3000: continue
    if REPRINT.search(tx[:6000]): continue          # drop modern reprints of old works
    bid=str(rec.get("id") or rec.get("barcode") or f"pos_{scanned}")
    open(os.path.join(DROPBOX,"positives_text",bid+".txt"),"w",encoding="utf-8").write(tx)
    kept.append({"id":bid,"source":"institutional_books_hf","meta_year":y,"label":1,"cls":"modern_post1929","origin":"hf_stream"})
    if len(kept)%50==0: print(f"  kept {len(kept)} (scanned {scanned})", flush=True)
    if len(kept)>=N_TARGET: break
with open(os.path.join(REPO,"manifest_positives.csv"),"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["id","source","meta_year","label","cls","origin"]); w.writeheader(); w.writerows(kept)
print(f"DONE: {len(kept)} positives (scanned {scanned}). text -> Dropbox/positives_text, manifest -> repo")
