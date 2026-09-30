#!/usr/bin/env python3
"""Night run 2026-09-29, arm 6 ("everything plus", best candidate): the largest training set, built once BOTH Talkie
cluster jobs are home. No model writes or edits anything here.
  = bookqa_all.jsonl (11,737 book pairs, as arms 1/2/4/5)
  + sources_v3/qa_v3/qa_v3_pairs.jsonl (16,251 pairs from 47 newly found books; same treatment as the book pairs:
    whitespace only; questions already in bookqa_all dropped; eval-overlap Jaccard > 0.5 dropped)
  + kept Talkie answers from job 1 (talkie_direct_kept.jsonl: book + encyclopaedia questions, pre-registered filters)
  + Talkie answers from job 2 (talkie_direct_v3/talkie_direct_answers_v3.jsonl) through the SAME filters
Output: runs/night_20260929/data/train_everything_plus.jsonl + everything_plus_report.json (+ kept/rejected v3 rows)."""
import hashlib, json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
sys.path.insert(0, str(ROOT / "scripts" / "v3")); sys.path.insert(0, str(ROOT / "scripts" / "v2"))
import build_talkie_direct_set as F  # noqa: E402  (filters; module does nothing on import)
from ranke_judge import wset  # noqa: E402

N = ROOT / "runs" / "night_20260929"
D = N / "data"
ev = [wset(json.loads(l)["prompt"]) for l in open(ROOT / "eval" / "frozen_eval_84.jsonl", encoding="utf-8")]


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


book = rd(D / "bookqa_all.jsonl")
seen = {" ".join(sorted(wset(x["question"]))) for x in book}
v3, why = [], Counter()
for x in rd(ROOT / "sources_v3" / "qa_v3" / "qa_v3_pairs.jsonl"):
    q, a = re.sub(r"\s+", " ", x["question"]).strip(), re.sub(r"\s+", " ", x["answer"]).strip()
    s = wset(q); k = " ".join(sorted(s))
    if not q or not a:
        why["empty"] += 1; continue
    if k in seen:
        why["duplicate_question"] += 1; continue
    if s and max(len(s & e) / len(s | e) for e in ev) > 0.5:
        why["eval_overlap_question"] += 1; continue
    seen.add(k)
    v3.append({"candidate_id": "books47_v3::" + x["pair_id"], "set": "books47_v3", "source_id": x["source_id"],
               "title": x["title"], "year": x["year"], "language": x["language"], "category": x["category"],
               "question": q, "answer": a})
t1 = rd(D / "talkie_direct_kept.jsonl")
t2_rows = rd(N / "talkie_direct_v3" / "talkie_direct_answers_v3.jsonl")
t2, t2_rej = F.filter_answers(t2_rows)
for x in t2:
    x["set"] = "talkie_book_question_v3"
allrows = book + v3 + t1 + t2


def write(name, items):
    f = D / name
    f.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
    return {"file": str(f), "rows": len(items), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}


rep = {"parts": {"book_pairs_11_22_books": len(book), "book_pairs_47_new_books": len(v3),
                 "talkie_job1_kept": len(t1), "talkie_job2_in": len(t2_rows), "talkie_job2_kept": len(t2)},
       "v3_dropped": why, "talkie_job2_rejection_reasons": Counter(w for x in t2_rej for w in x["reasons"]),
       "files": {"train": write("train_everything_plus.jsonl", allrows),
                 "talkie_v3_kept": write("talkie_direct_v3_kept.jsonl", t2),
                 "talkie_v3_rejected": write("talkie_direct_v3_rejected.jsonl", t2_rej)}}
(D / "everything_plus_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print(json.dumps(rep, indent=1))
