#!/usr/bin/env python3
"""Pack the project's outputs into a few Parquet files for Dropbox (JV 2026-09-30: no floods of small files).

Source: G:\\detritus\\ranke1929_talkie_sft_20260928 (everything except model weights, scratch, staging, zips,
caches and scripts/ - code is on GitHub, repo ranke-history-llms/posttrain).
Output folder (default <root>\\export_parquet\\), four tables + a manifest:
  grades.parquet     one row per grading file under judge_* folders (per call x judge x attempt):
                     run, judge, item_id, file_name, text (request / raw reply / judgment / meta / error)
  records.parquet    one row per JSON line of every .jsonl outside judge_* folders and outside sources_v3/cpt:
                     path, line_no, record (JSON string) + common fields pulled out (condition, prompt_id,
                     candidate_id, set, question, answer)
  cpt_corpus.parquet one row per JSON line of sources_v3/cpt/*.jsonl (continued-pretraining text), path, line_no, record
  files.parquet      every other file, whole: path, bytes, sha256 (md, json, csv, log, txt, raw source texts, png ...)
  manifest.csv       every source file: path, size, sha256, table it went to  (the replay key)
Rows are exact copies (text or bytes) of the source; nothing is parsed away. Restore = write rows back to paths.
"""
import csv, hashlib, json, sys
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "export_parquet"
SKIP_DIRS = {"adapter", "final", "scratch", "staging", "__pycache__", ".venv", "export_parquet", "scripts"}
SKIP_EXT = {".safetensors", ".pt", ".bin", ".zip", ".gz", ".pyc"}
PULL = ["condition", "prompt_id", "candidate_id", "set", "question", "answer"]
OUT.mkdir(parents=True, exist_ok=True)


def files():
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_file() and not (set(rel.parts[:-1]) & SKIP_DIRS) and p.suffix.lower() not in SKIP_EXT:
            yield p, rel.as_posix()


class Writer:
    def __init__(self, name, schema, batch=20000):
        self.path, self.schema, self.buf, self.batch, self.n = OUT / name, schema, [], batch, 0
        self.w = pq.ParquetWriter(self.path, schema, compression="zstd")

    def add(self, row):
        self.buf.append(row); self.n += 1
        if len(self.buf) >= self.batch:
            self.flush()

    def flush(self):
        if self.buf:
            self.w.write_table(pa.Table.from_pylist(self.buf, schema=self.schema)); self.buf = []

    def close(self):
        self.flush(); self.w.close()


S = pa.string()
grades = Writer("grades.parquet", pa.schema([("path", S), ("run", S), ("judge", S), ("item_id", S), ("file_name", S), ("text", S)]))
records = Writer("records.parquet", pa.schema([("path", S), ("line_no", pa.int32()), ("record", S)] + [(k, S) for k in PULL]))
cpt = Writer("cpt_corpus.parquet", pa.schema([("path", S), ("line_no", pa.int32()), ("record", S)]), batch=5000)
other = Writer("files.parquet", pa.schema([("path", S), ("size", pa.int64()), ("sha256", S), ("bytes", pa.binary())]), batch=200)
man = []
for p, rel in files():
    data = p.read_bytes(); h = hashlib.sha256(data).hexdigest()
    parts = rel.split("/")
    jdx = next((i for i, x in enumerate(parts) if x.startswith("judge_")), None)
    if jdx is not None:
        item = parts[jdx + 1] if len(parts) > jdx + 2 else Path(parts[-1]).stem
        grades.add({"path": rel, "run": "/".join(parts[:jdx]), "judge": parts[jdx], "item_id": item,
                    "file_name": parts[-1], "text": data.decode("utf-8", errors="replace")}); table = "grades"
    elif p.suffix == ".jsonl":
        w = cpt if rel.startswith("sources_v3/cpt/") else records; table = w.path.stem
        for i, line in enumerate(data.decode("utf-8", errors="replace").splitlines()):
            if not line.strip():
                continue
            row = {"path": rel, "line_no": i, "record": line}
            if w is records:
                try:
                    o = json.loads(line)
                    for k in PULL:
                        v = o.get(k) if isinstance(o, dict) else None
                        row[k] = None if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                except json.JSONDecodeError:
                    pass
            w.add(row)
    else:
        other.add({"path": rel, "size": len(data), "sha256": h, "bytes": data}); table = "files"
    man.append({"path": rel, "size": len(data), "sha256": h, "table": table})
for w in (grades, records, cpt, other):
    w.close()
with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=["path", "size", "sha256", "table"]); wr.writeheader(); wr.writerows(man)
summary = {w.path.name: {"rows": w.n, "mb": round(w.path.stat().st_size / 1e6, 1)} for w in (grades, records, cpt, other)}
summary["source_files"] = len(man); summary["source_mb"] = round(sum(m["size"] for m in man) / 1e6, 1)
(OUT / "export_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps(summary))
