#!/usr/bin/env python3
"""Plan v3 (ranke-nextround-0929), data arm N: question-answer pairs printed as such in period books.

Input: G:\\detritus\\ranke_period_qa_run1_20260927\\data\\luna_accepted_literal.jsonl (4,023 literal pairs from
11 books, 1813-1909; a modern model only SELECTED pairs, never wrote or edited words).
Deterministic cleanup only (no model): join OCR line-break hyphens between lowercase letters
("the- idea" -> "the idea" only when a hyphen is followed by space+lowercase and preceded by a lowercase letter
 is NOT done: we keep hyphens; we only collapse whitespace, strip a trailing lone footnote number, drop stray
 quote-mark glitches like "consenting' minds"). Raw text kept in *_raw fields.
Filters (same rules as the Talkie data): protected eval terms (validate_talkie_pairs.PROTECTED), late-fact word list,
question word-set Jaccard > 0.5 with any eval prompt, answer word-set Jaccard > 0.5 with any eval prompt+criteria,
duplicate questions, empty fields.
Output: out/native/native_all.jsonl (+ native_report.json with every rejection reason).
"""
import hashlib, json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "v2"))
from validate_talkie_pairs import contains_protected_term  # noqa: E402
from ranke_judge import late_hits, wset  # noqa: E402

SRC = Path(r"G:\detritus\ranke_period_qa_run1_20260927\data\luna_accepted_literal.jsonl")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ev = [json.loads(l) for l in open(ROOT / "eval" / "frozen_eval_84.jsonl", encoding="utf-8")]
evq = [wset(e["prompt"]) for e in ev]
eva = [wset(e["prompt"] + " " + " ".join(e.get("criteria", []))) for e in ev]


def clean(t):
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(?<=[a-z])' (?=[a-z])", " ", t)          # OCR stray apostrophe: "consenting' minds"
    t = re.sub(r"\s+\d{1,2}$", "", t)                       # trailing footnote/page number
    return t.strip()


rows, report, seen = [], [], set()
for r in map(json.loads, open(SRC, encoding="utf-8")):
    q, a = clean(r["question_display"]), clean(r["answer_display"])
    reasons = []
    if not q or not a:
        reasons.append("empty")
    if contains_protected_term(q + " " + a):
        reasons.append("protected_term")
    if late_hits(q + " " + a):
        reasons.append("late_fact_terms")
    qs, as_ = wset(q), wset(a)
    if max((len(qs & e) / len(qs | e) if qs | e else 0) for e in evq) > 0.5:
        reasons.append("eval_overlap_question")
    if max((len(as_ & e) / len(as_ | e) if as_ | e else 0) for e in eva) > 0.5:
        reasons.append("eval_overlap_answer")
    key = " ".join(sorted(qs))
    if key in seen:
        reasons.append("duplicate_question")
    seen.add(key)
    report.append({"id": r["id"], "category": r["category"], "reasons": reasons})
    if not reasons:
        rows.append({"candidate_id": "native::" + r["id"], "passage_id": r["id"], "source_id": r["source_id"],
                     "title": r["title"], "year": r["year"], "category": r["category"], "form": r["category"],
                     "question": q, "answer": a, "question_raw": r["question_display"], "answer_raw": r["answer_display"],
                     "source_sha256": r["source_sha256"], "selector_model": r["selector_model"]})
(OUT / "native_all.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
summary = {"input": str(SRC), "input_rows": len(report), "kept": len(rows),
           "rejections": Counter(x for r in report for x in r["reasons"]),
           "kept_by_category": Counter(r["category"] for r in rows),
           "sha256": hashlib.sha256((OUT / "native_all.jsonl").read_bytes()).hexdigest()}
(OUT / "native_report.json").write_text(json.dumps({**summary, "records": report}, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
