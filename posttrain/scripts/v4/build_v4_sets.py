#!/usr/bin/env python3
"""Plan v4 (ranke-gap-0930, JV GATE-APPROVE 2026-09-30 09:37) training sets. Deterministic, no model.

Source: runs/night_20260929/data/train_everything_plus.jsonl (41,272 pairs; duplicates and eval overlaps already
removed by build_everything_plus.py). Sets:
  talkie_13330 : all Talkie direct answers (talkie_book_question + talkie_encyclopaedia + talkie_book_question_v3)
  talkie_6370  : job-1 Talkie answers = the night arm's train_talkie_all.jsonl (same rows; checked)
  talkie_<n>   : random n of talkie_6370 (seed 1929), n = size of the long book set -> nested n < 6,370 < 13,330
  books_long_3760 : JV 09:37 request - book pairs (books22_v2, books11_native, books47_v3), English, answer length in
                 the top quarter of all book answers (>= 75th percentile words), question stands alone (same rule (f)
                 as the Talkie filter: >= 3 words, capital first, none of the context words). All that pass (3,753);
                 talkie_<n> is size-matched to it.
Output: runs/v4_20260930/data/<set>.jsonl + sets_report.json (counts, sha256, length stats).
"""
import hashlib, json, random, re, statistics as st
from pathlib import Path

ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
SRC = ROOT / "runs" / "night_20260929" / "data" / "train_everything_plus.jsonl"
T1 = ROOT / "runs" / "night_20260929" / "data" / "train_talkie_all.jsonl"
OUT = ROOT / "runs" / "v4_20260930" / "data"
TALKIE = {"talkie_book_question", "talkie_encyclopaedia", "talkie_book_question_v3"}
BOOKS = {"books22_v2", "books11_native", "books47_v3"}
CTX = re.compile(r"\b(it|its|he|him|his|she|her|they|them|their|this|these|those|former|latter|same|such|above|then|thus|so)\b", re.I)

rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
words = lambda r: len(r["answer"].split())
talkie = [r for r in rows if r["set"] in TALKIE]
t1_ids = {json.loads(l)["candidate_id"] for l in open(T1, encoding="utf-8")}
talkie6 = [r for r in talkie if r["candidate_id"] in t1_ids]
assert len(talkie) == 13330 and len(talkie6) == 6370, (len(talkie), len(talkie6))
rng = random.Random(1929)
books = [r for r in rows if r["set"] in BOOKS]
L = sorted(words(r) for r in books); cut = L[int(.75 * len(L))]
long_books = [r for r in books if r.get("language") == "en" and words(r) >= cut
              and len(r["question"].split()) >= 3 and r["question"][:1].isupper() and not CTX.search(r["question"])]
talkie3 = rng.sample(talkie6, len(long_books))   # size-matched to the long book set (3,753 on 2026-09-30)
sets = {"talkie_13330": talkie, "talkie_6370": talkie6, f"talkie_{len(long_books)}": talkie3, f"books_long_{len(long_books)}": long_books}
OUT.mkdir(parents=True, exist_ok=True)
rep = {"source": [str(SRC), hashlib.sha256(SRC.read_bytes()).hexdigest()], "book_length_cut_words": cut, "sets": {}}
for name, rs in sets.items():
    p = OUT / f"{name}.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
    ws = [words(r) for r in rs]
    rep["sets"][name] = {"rows": len(rs), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "median_answer_words": st.median(ws),
                         "mean_answer_words": round(st.mean(ws), 1), "by_category": {c: sum(r.get("category") == c for r in rs) for c in sorted({r.get("category") for r in rs})}}
(OUT / "sets_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print(json.dumps({k: {m: v[m] for m in ("rows", "median_answer_words")} for k, v in rep["sets"].items()} | {"cut": cut}))
