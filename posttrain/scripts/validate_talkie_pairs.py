#!/usr/bin/env python3
"""Apply reproducible mechanical checks to Talkie question-answer records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PROTECTED = (
    "norman angell",
    "great illusion",
    "angell",
    "russo-japanese",
    "russo japanese",
    "port arthur",
    "tsushima",
    "mukden",
    "portsmouth",
    "manchuria",
    "japan",
    "japanese",
    "russia",
    "russian",
)
MODEL_REVISION = "8033675be6360ae0127fa75f941c12d52064f1dc"


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def longest_source_run(answer: str, source: str) -> int:
    aw, sw = words(answer), words(source)
    if not aw or not sw:
        return 0
    positions: dict[str, list[int]] = {}
    for i, token in enumerate(sw):
        positions.setdefault(token, []).append(i)
    best = 0
    for i, token in enumerate(aw):
        for j in positions.get(token, []):
            k = 0
            while i + k < len(aw) and j + k < len(sw) and aw[i + k] == sw[j + k]:
                k += 1
            best = max(best, k)
    return best


def contains_protected_term(text: str) -> bool:
    lowered = text.lower()
    return any(
        re.search(r"(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z])", lowered)
        for term in PROTECTED
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--accepted", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--min-answer-words", type=int, default=80)  # plan v2.11 (JV 07:36): 20
    args = ap.parse_args()

    sources = {}
    with open(args.sources, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                sources[row["passage_id"]] = row["text"]

    generated = []
    with open(args.generated, encoding="utf-8") as handle:
        generated = [json.loads(line) for line in handle if line.strip()]

    seen_questions: set[str] = set()
    reason_counts: Counter[str] = Counter()
    accepted = []
    audited = []
    for row in generated:
        reasons = []
        question = row.get("question") or ""
        answer = row.get("answer") or ""
        source = sources.get(row.get("passage_id"), "")
        answer_words = words(answer)
        qnorm = " ".join(words(question))
        if row.get("generation_status") != "ok" or row.get("parse_status") != "parsed":
            reasons.append("generation_or_parse_failure")
        if row.get("model_revision") != MODEL_REVISION:
            reasons.append("wrong_model_revision")
        if not source:
            reasons.append("source_missing")
        if not question.endswith("?"):
            reasons.append("question_not_interrogative")
        if len(words(question)) < 7 or len(words(question)) > 45:
            reasons.append("question_length")
        if len(answer_words) < args.min_answer_words or len(answer_words) > 240:
            reasons.append("answer_length")
        if qnorm in seen_questions:
            reasons.append("duplicate_question")
        seen_questions.add(qnorm)
        if contains_protected_term(question + " " + answer):
            reasons.append("protected_term")
        copied_run = longest_source_run(answer, source)
        if copied_run >= 24:
            reasons.append("long_verbatim_run")
        for reason in reasons:
            reason_counts[reason] += 1
        checked = dict(row)
        checked["mechanical_accept"] = not reasons
        checked["mechanical_reasons"] = reasons
        checked["answer_word_count"] = len(answer_words)
        checked["longest_source_word_run"] = copied_run
        audited.append(checked)
        if not reasons:
            accepted.append(checked)

    accepted_path = Path(args.accepted)
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    with accepted_path.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "generated": len(generated),
        "mechanically_accepted": len(accepted),
        "accept_rate": len(accepted) / len(generated) if generated else 0.0,
        "rejection_reasons": dict(reason_counts),
        "generated_sha256": hashlib.sha256(Path(args.generated).read_bytes()).hexdigest(),
        "accepted_sha256": hashlib.sha256(accepted_path.read_bytes()).hexdigest(),
        "records": audited,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
