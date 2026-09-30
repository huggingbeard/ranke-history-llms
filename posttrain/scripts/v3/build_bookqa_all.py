#!/usr/bin/env python3
"""Night run 2026-09-29 (JV: "find things for the 4090 to do for ranke"): ONE training set of all question-answer
pairs printed as such in period books. No model writes, edits or selects anything here (deterministic code only).

Inputs
  A = sources_v3/qa/qa_pairs_v2_uncorrected.jsonl  (9,266 pairs, 22 books; JV 16:16 filter rules already applied:
      only specific events + 1914-1939 years dropped; text as scanned, JV 17:05 "forget the ocr correction")
  B = runs/v3_20260929/data/native/native_qa_books.jsonl (2,644 pairs, 11 books; used by the round-3 "books" arms)
Rules (same as build_native_set.py, minus the old broad late-fact/protected filters that JV narrowed at 16:16)
  - drop empty question or answer
  - drop if question word-set Jaccard > 0.5 with any of the 84 frozen eval prompts
  - drop if answer word-set Jaccard > 0.5 with any eval prompt + its criteria
  - drop duplicate questions (same word set); A is read first, so A wins over B
Output: runs/night_20260929/data/bookqa_all.jsonl (+ bookqa_all_report.json with every rejection and its reason)."""
import hashlib, json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
sys.path.insert(0, str(ROOT / "scripts" / "v2"))
from ranke_judge import wset  # noqa: E402

A = ROOT / "sources_v3" / "qa" / "qa_pairs_v2_uncorrected.jsonl"
B = ROOT / "runs" / "v3_20260929" / "data" / "native" / "native_qa_books.jsonl"
OUT = ROOT / "runs" / "night_20260929" / "data"
OUT.mkdir(parents=True, exist_ok=True)
ev = [json.loads(l) for l in open(ROOT / "eval" / "frozen_eval_84.jsonl", encoding="utf-8")]
evq = [wset(e["prompt"]) for e in ev]
eva = [wset(e["prompt"] + " " + " ".join(e.get("criteria", []))) for e in ev]


def jac(s, pool):
    return max((len(s & e) / len(s | e) if s | e else 0.0) for e in pool)


rows, report, seen = [], [], set()
for src, path in (("books22_v2", A), ("books11_native", B)):
    for r in map(json.loads, open(path, encoding="utf-8")):
        q = re.sub(r"\s+", " ", r["question"]).strip()
        a = re.sub(r"\s+", " ", r["answer"]).strip()
        rid = r.get("pair_id") or r.get("candidate_id")
        reasons = []
        if not q or not a:
            reasons.append("empty")
        qs, as_ = wset(q), wset(a)
        if qs and jac(qs, evq) > 0.5:
            reasons.append("eval_overlap_question")
        if as_ and jac(as_, eva) > 0.5:
            reasons.append("eval_overlap_answer")
        key = " ".join(sorted(qs))
        if key in seen:
            reasons.append("duplicate_question")
        seen.add(key)
        report.append({"id": rid, "set": src, "reasons": reasons})
        if not reasons:
            rows.append({"candidate_id": f"{src}::{rid}", "set": src, "source_id": r.get("source_id"),
                         "title": r.get("title"), "year": r.get("year"), "language": r.get("language", "en"),
                         "category": r.get("category"), "question": q, "answer": a})
f = OUT / "bookqa_all.jsonl"
f.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
summary = {"inputs": [str(A), str(B)], "input_rows": len(report), "kept": len(rows),
           "rejections": Counter(x for r in report for x in r["reasons"]),
           "kept_by_set": Counter(r["set"] for r in rows), "kept_by_language": Counter(r["language"] for r in rows),
           "kept_by_category": Counter(r["category"] for r in rows),
           "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
(OUT / "bookqa_all_report.json").write_text(json.dumps({**summary, "records": report}, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
