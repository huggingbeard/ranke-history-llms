#!/usr/bin/env python3
"""Night run 2026-09-29: filter Talkie-13B's direct answers (cluster job 6697045) with the PRE-REGISTERED mechanical
rules (LOG.md 19:15; no model judges or edits anything) and build the three training files for the 4090 arms.

Every answer, book question or encyclopaedia question:
  (a) >= 20 words (JV option B), (b) Talkie stopped cleanly (termination == "eos"),
  (c) JV event rule (qa_v2_restore.blocked: EVENT_TERMS + years 1914-1939 in digits or words),
  (e) no 4-digit year after 1913 anywhere in the answer (covers 1940+ as well).
Encyclopaedia answers additionally:
  (d) agree with the EB11 entry: share a year (4 digits) with the entry, or >= 2 capitalised words other than the
      words of the asked name / heading.
Outputs (runs/night_20260929/data/):
  talkie_direct_kept.jsonl, talkie_direct_rejected.jsonl (every rejected row + reasons), talkie_direct_report.json
  train_talkie_all.jsonl       = kept Talkie answers (book + encyclopaedia questions)           -> arm 3
  train_everything.jsonl       = bookqa_all.jsonl + kept Talkie answers (book + encyclopaedia)  -> arm 4
  train_no_encyclopedia.jsonl  = bookqa_all.jsonl + kept Talkie answers to BOOK questions only  -> arm 5"""
import hashlib, json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
sys.path.insert(0, str(ROOT / "scripts" / "v2"))
# JV 16:16 event rule, copied verbatim from sources_v3/qa/qa_v2_restore.py (importing that file re-runs it and
# rewrites qa_pairs_v2_uncorrected.jsonl, so it is not imported).
from ranke_judge import LATE_YEAR, SPELLED_YEARS  # noqa: E402
EVENT_TERMS = ["russo-japanese", "russo japanese", "port arthur", "tsushima", "mukden", "liaoyang", "shaho",
               "treaty of portsmouth", "norman angell", "great illusion"]


def blocked(text):
    low = text.lower()
    ev = [t for t in EVENT_TERMS if t in low]
    yrs = [m.group(0) for m in LATE_YEAR.finditer(text)] +           [w for w in SPELLED_YEARS if w in low.replace("nineteen hundred and ", "nineteen ")]
    return ev + yrs


D = ROOT / "runs" / "night_20260929" / "data"
ANS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "night_20260929" / "talkie_direct" / "talkie_direct_answers.jsonl"
enc = {json.loads(l)["candidate_id"]: json.loads(l) for l in open(D / "encyclopedia_questions.jsonl", encoding="utf-8")}
YEAR = re.compile(r"(?<!\d)(1\d{3}|20\d{2})(?!\d)")
CAP = re.compile(r"\b[A-Z][a-z]{2,}\b")
STOP = {"The", "This", "That", "These", "Those", "His", "Her", "Its", "Their", "Who", "What", "When", "Where", "Which",
        "And", "But", "For", "With", "From", "After", "Before", "During", "Under", "Upon", "There", "Some", "Many",
        "Being", "Having", "Although", "Though", "While", "Since", "Chiefly", "Remembered"}


# (f) AMENDMENT 2026-09-29 19:52, logged before any filtered result was looked at: many book questions only make
# sense inside the book ("What was it?", "How far did he extend his dominion?"); Talkie then answers nonsense
# (first rows: "What was it?" -> "a kind of musical entertainment"). Book questions must stand alone: after stripping
# numbering ("441 Q. -", "1.", "-"), >= 3 words, start with a capital, and no word pointing back to earlier text.
DEICTIC = {"it", "its", "he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs", "this", "these",
           "those", "former", "latter", "same", "such", "above", "then", "thus", "so"}


def standalone(q):
    q2 = re.sub(r"^[\s\-—\d\.]*(?:Q\.)?[\s\-—]*", "", q).strip()
    w = re.findall(r"[A-Za-z’']+", q2)
    if len(w) < 3 or not q2[:1].isupper():
        return False
    return not any(x.lower() in DEICTIC for x in w)


def reasons_for(r):
    a = r["answer"]
    why = []
    if not r["candidate_id"].startswith("eb11::") and not standalone(r["question"]):
        why.append("question_needs_book_context")
    if len(a.split()) < 20:
        why.append("under_20_words")
    if r.get("termination") != "eos":
        why.append("not_stopped_cleanly")
    if blocked(r["question"] + " " + a):
        why.append("jv_event_or_1914_1939_rule")
    if any(int(y) > 1913 for y in YEAR.findall(a)):
        why.append("year_after_1913")
    if r["candidate_id"].startswith("eb11::"):
        e = enc[r["candidate_id"]]
        asked = set(CAP.findall(r["question"])) | set(CAP.findall(e["headword"].title()))
        shared_years = set(YEAR.findall(a)) & set(YEAR.findall(e["entry_text"]))
        shared_caps = (set(CAP.findall(a)) - asked - STOP) & set(CAP.findall(e["entry_text"]))
        if not shared_years and len(shared_caps) < 2:
            why.append("does_not_agree_with_encyclopaedia_entry")
    return why


def filter_answers(rows):
    kept, rej = [], []
    for r in rows:
        why = reasons_for(r)
        src = "encyclopaedia" if r["candidate_id"].startswith("eb11::") else "book_question"
        rec = {"candidate_id": "talkie::" + r["candidate_id"], "set": "talkie_" + src, "question": r["question"],
               "answer": r["answer"], "source_id": r.get("source_id"), "title": r.get("title"), "year": r.get("year"),
               "category": r.get("category"), "language": "en", "teacher": r.get("model"),
               "teacher_revision": r.get("model_revision"), "new_tokens": r.get("new_tokens")}
        (rej if why else kept).append({**rec, "reasons": why} if why else rec)
    return kept, rej


def write(name, items):
    f = D / name
    f.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
    return {"file": str(f), "rows": len(items), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}


def main():
    rows = [json.loads(l) for l in open(ANS, encoding="utf-8") if l.strip()]
    kept, rej = filter_answers(rows)
    book = [json.loads(l) for l in open(D / "bookqa_all.jsonl", encoding="utf-8")]
    report(rows, kept, rej, book)


def report(rows, kept, rej, book):
  out = {"answers_in": len(rows), "kept": len(kept), "rejected": len(rej),
       "kept_by_set": Counter(x["set"] for x in kept), "in_by_set": Counter(
           "talkie_encyclopaedia" if r["candidate_id"].startswith("eb11::") else "talkie_book_question" for r in rows),
       "rejection_reasons": Counter(w for x in rej for w in x["reasons"]),
       "files": {"kept": write("talkie_direct_kept.jsonl", kept), "rejected": write("talkie_direct_rejected.jsonl", rej),
                 "arm3": write("train_talkie_all.jsonl", kept),
                 "arm4": write("train_everything.jsonl", book + kept),
                 "arm5": write("train_no_encyclopedia.jsonl", book + [x for x in kept if x["set"] == "talkie_book_question"])}}
  (D / "talkie_direct_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
