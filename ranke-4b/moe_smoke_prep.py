"""OLMoE smoke test, step 1 (2026-10-20, v1): build a fixed token snapshot from whole pre-1900 books.

Source: ONE file of the heuristic-filtered English PD books on the UZH cluster share, copied once (the live folder
was being rewritten on 2026-10-20 13:06, so we never read it live):
  /shares/voth.econ.uzh/ranke_ingest_data/data/heuristic_filter/english_pd/000_00000.parquet  (1,696 books, all catalogue
  year < 1900; 'year' and 'date' fields agree on all). Quarantine folder not used. Catalogue year only: the corpus is not
  yet decontaminated - fine for an engineering test, not a quality claim.
Tokenizer: Ranke's own (Qwen3, 151,936 tokens): tokenizer.json from /scratch/ukrenk/hf-models/Ranke-4B-1913@99fd739...
Split by WHOLE BOOKS (books in file order): validation = first books until >= 1.0M tokens; training = next books until
>= 128 x 32 x 4096 = 16.8M tokens (enough for 128 steps of 131,072 tokens). Each book ends with <|endoftext|>.
Outputs (uint32 token arrays + manifest with book ids, token counts, SHA-256):
  <out>/train_tokens.npy, val_tokens.npy, manifest.json
Usage: python moe_smoke_prep.py <parquet> <tokenizer.json> <out_dir>
"""
import hashlib, json, os, sys
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

src, tok_path, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
tok = Tokenizer.from_file(tok_path)
eot = tok.token_to_id("<|endoftext|>")
assert eot is not None
t = pq.read_table(src, columns=["id", "text", "metadata"])
ids, texts, meta = t.column("id").to_pylist(), t.column("text").to_pylist(), t.column("metadata").to_pylist()
VAL_TARGET, TRAIN_TARGET = 1_000_000, 128 * 32 * 4096   # 16.8M tokens = 128 steps of 131,072
val, train, val_books, train_books = [], [], [], []
for bid, text, m in zip(ids, texts, meta):
    assert m["year"] < 1900, (bid, m["year"])
    if sum(map(len, train)) >= TRAIN_TARGET:
        break
    toks = tok.encode(text).ids + [eot]
    if sum(map(len, val)) < VAL_TARGET:
        val.append(toks); val_books.append(dict(id=bid, year=m["year"], title=m.get("title"), tokens=len(toks)))
    else:
        train.append(toks); train_books.append(dict(id=bid, year=m["year"], title=m.get("title"), tokens=len(toks)))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
res = {}
for name, chunks, books in [("train", train, train_books), ("val", val, val_books)]:
    arr = np.fromiter((x for c in chunks for x in c), dtype=np.uint32)
    p = os.path.join(out, f"{name}_tokens.npy")
    np.save(p, arr)
    res[name] = dict(tokens=int(arr.size), books=len(books), sha256=sha(p), book_list=books)
res["source"] = dict(path=src, sha256=sha(src))
res["tokenizer"] = dict(path=tok_path, sha256=sha(tok_path), vocab=tok.get_vocab_size(), eot=eot)
json.dump(res, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "book_list"} for k, v in res.items()}))
