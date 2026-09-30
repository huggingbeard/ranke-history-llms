#!/usr/bin/env python3
"""Measure deterministic surface failures in frozen evaluation outputs.

These measures do not judge historical correctness.  They expose mechanical
failures that SFT should reduce: failure to stop, repeated sentences or
n-grams, and generation of a new Question/Answer exchange inside an answer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized_words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def repetition_metrics(text: str) -> dict[str, float | int | bool]:
    words = normalized_words(text)
    fourgrams = [tuple(words[i : i + 4]) for i in range(max(0, len(words) - 3))]
    eightgrams = [tuple(words[i : i + 8]) for i in range(max(0, len(words) - 7))]
    sentences = [
        " ".join(normalized_words(s))
        for s in SENTENCE_RE.split(text)
        if len(normalized_words(s)) >= 4
    ]
    fourgram_repeat = 0.0 if not fourgrams else 1.0 - len(set(fourgrams)) / len(fourgrams)
    sentence_repeat = 0.0 if not sentences else 1.0 - len(set(sentences)) / len(sentences)
    max_eightgram_count = max(Counter(eightgrams).values(), default=0)
    return {
        "word_count": len(words),
        "distinct_word_ratio": 0.0 if not words else len(set(words)) / len(words),
        "fourgram_repeat_fraction": fourgram_repeat,
        "sentence_repeat_fraction": sentence_repeat,
        "max_eightgram_count": max_eightgram_count,
        "qa_continuation": bool(re.search(r"(?im)^\s*(question|answer)\s*:", text)),
        "loop_flag": sentence_repeat >= 0.20 or max_eightgram_count >= 3,
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"rows": 0}
    return {
        "rows": len(rows),
        "length_terminated_n": sum(r["termination"] == "length" for r in rows),
        "length_terminated_rate": mean(r["termination"] == "length" for r in rows),
        "loop_flag_n": sum(bool(r["loop_flag"]) for r in rows),
        "loop_flag_rate": mean(bool(r["loop_flag"]) for r in rows),
        "qa_continuation_n": sum(bool(r["qa_continuation"]) for r in rows),
        "qa_continuation_rate": mean(bool(r["qa_continuation"]) for r in rows),
        "median_word_count": median(r["word_count"] for r in rows),
        "mean_fourgram_repeat_fraction": mean(r["fourgram_repeat_fraction"] for r in rows),
        "mean_sentence_repeat_fraction": mean(r["sentence_repeat_fraction"] for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, type=Path)
    ap.add_argument("--battery", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    battery = {}
    with args.battery.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            battery[row["id"]] = row

    measured = []
    with args.matrix.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompt_id = row["prompt_id"]
            meta = battery[prompt_id]
            measured.append(
                {
                    "prompt_id": prompt_id,
                    "condition": row["condition"],
                    "suite": meta["suite"],
                    "category": meta["category"],
                    "termination": row["termination"],
                    "new_tokens": row["new_tokens"],
                    **repetition_metrics(row["answer"]),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "surface_metrics.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(measured[0]))
        writer.writeheader()
        writer.writerows(measured)

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in measured:
        groups[f"condition={row['condition']}"] .append(row)
        groups[f"condition={row['condition']}|suite={row['suite']}"] .append(row)

    summary = {
        "matrix": str(args.matrix),
        "matrix_sha256": sha256(args.matrix),
        "battery": str(args.battery),
        "battery_sha256": sha256(args.battery),
        "metrics_scope": "deterministic surface behavior only; not historical correctness",
        "groups": {key: summarize(value) for key, value in sorted(groups.items())},
        "row_metrics": str(rows_path),
    }
    summary_path = args.output_dir / "surface_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
