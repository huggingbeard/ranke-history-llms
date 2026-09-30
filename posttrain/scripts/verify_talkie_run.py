#!/usr/bin/env python3
"""Fail closed when a Talkie generation run lacks required evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_REPO_ID = "talkie-lm/talkie-1930-13b-it"
EXPECTED_REVISION = "8033675be6360ae0127fa75f941c12d52064f1dc"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("verification failed: " + message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-code-sha256", required=True)
    parser.add_argument("--expected-max-tokens", type=int, required=True)
    parser.add_argument("--min-parsed", type=int, default=0)
    parser.add_argument("--min-accepted", type=int, default=0)
    args = parser.parse_args()

    generated_path = Path(args.generated)
    accepted_path = Path(args.accepted)
    report_path = Path(args.report)
    receipt_path = generated_path.with_name(generated_path.name + ".receipt.json")
    request_path = generated_path.with_name(generated_path.name + ".request.json")
    capture_dir = generated_path.with_name(generated_path.name + ".captures")

    for path in (generated_path, accepted_path, report_path, receipt_path, request_path):
        if not path.is_file():
            raise SystemExit(f"missing required artifact: {path}")
    if not capture_dir.is_dir():
        raise SystemExit(f"missing capture directory: {capture_dir}")

    rows = [
        json.loads(line)
        for line in generated_path.open(encoding="utf-8")
        if line.strip()
    ]
    accepted = [
        json.loads(line)
        for line in accepted_path.open(encoding="utf-8")
        if line.strip()
    ]
    report = read_json(report_path)
    receipt = read_json(receipt_path)
    request = read_json(request_path)

    require(len(rows) == args.expected_count, f"generated rows {len(rows)} != {args.expected_count}")
    require(all(row.get("generation_status") == "ok" for row in rows), "generation error present")
    require(all(row.get("model_revision") == EXPECTED_REVISION for row in rows), "wrong model revision in row")
    require(all(row.get("max_tokens") == args.expected_max_tokens for row in rows), "wrong max_tokens in row")
    require(
        all(row.get("code_archive_sha256") == args.expected_code_sha256 for row in rows),
        "wrong source archive hash in row",
    )
    candidate_ids = [row.get("candidate_id") for row in rows]
    require(all(candidate_ids), "missing candidate_id")
    require(len(set(candidate_ids)) == args.expected_count, "duplicate candidate_id")
    parsed = sum(row.get("parse_status") == "parsed" for row in rows)
    require(parsed >= args.min_parsed, f"parsed rows {parsed} < {args.min_parsed}")
    require(len(accepted) >= args.min_accepted, f"accepted rows {len(accepted)} < {args.min_accepted}")
    require(
        {row.get("candidate_id") for row in accepted}.issubset(set(candidate_ids)),
        "accepted output contains an unknown candidate",
    )

    require(request.get("input_sha256") == args.expected_input_sha256, "wrong input hash")
    require(request.get("model_revision") == EXPECTED_REVISION, "wrong request model revision")
    require(request.get("expected_repo_id") == EXPECTED_REPO_ID, "wrong request repository")
    require(request.get("max_tokens") == args.expected_max_tokens, "wrong request max_tokens")
    require(request.get("code_archive_sha256") == args.expected_code_sha256, "wrong request archive hash")
    require(request.get("candidate_ids") == candidate_ids, "request and output candidate order differ")

    require(receipt.get("status") == "complete", "receipt is not complete")
    require(receipt.get("new_rows") == args.expected_count, "wrong receipt row count")
    require(receipt.get("generation_ok") == args.expected_count, "wrong receipt success count")
    require(receipt.get("generation_errors") == 0, "receipt reports generation errors")
    require(receipt.get("parsed") == parsed, "wrong receipt parsed count")
    require(receipt.get("code_archive_sha256") == args.expected_code_sha256, "wrong receipt archive hash")
    require(receipt.get("complete_raw_captures") == args.expected_count, "wrong raw capture count")
    require(receipt.get("partial_raw_captures") == 0, "partial raw capture remains")
    require(receipt.get("request_captures") == args.expected_count, "wrong candidate request count")
    require(receipt.get("record_captures") == args.expected_count, "wrong record capture count")
    require(receipt.get("model_lock", {}).get("repo_id") == EXPECTED_REPO_ID, "wrong locked repository")
    require(receipt.get("model_lock", {}).get("revision") == EXPECTED_REVISION, "wrong locked revision")
    output_sha = hashlib.sha256(generated_path.read_bytes()).hexdigest()
    require(receipt.get("output_sha256") == output_sha, "generated output hash mismatch")

    require(report.get("generated") == args.expected_count, "wrong validator generated count")
    require(report.get("mechanically_accepted") == len(accepted), "wrong validator accepted count")
    require(report.get("generated_sha256") == output_sha, "validator input hash mismatch")
    require(
        report.get("accepted_sha256") == hashlib.sha256(accepted_path.read_bytes()).hexdigest(),
        "accepted output hash mismatch",
    )

    complete_raw = list(capture_dir.glob("*.completion.txt"))
    partial_raw = list(capture_dir.glob("*.completion.txt.partial"))
    request_captures = list(capture_dir.glob("*.request.json"))
    record_captures = list(capture_dir.glob("*.record.json"))
    require(len(complete_raw) == args.expected_count, "wrong complete raw capture count")
    require(not partial_raw, "partial raw captures remain")
    require(all(path.stat().st_size > 0 for path in complete_raw), "empty raw capture")
    require(len(request_captures) == args.expected_count, "wrong request file count")
    require(len(record_captures) == args.expected_count, "wrong record file count")
    for row in rows:
        raw_capture = Path(row["raw_capture"])
        request_capture = Path(row["request_capture"])
        record_capture = Path(row["record_capture"])
        require(raw_capture.is_file(), f"missing row raw capture: {raw_capture}")
        require(request_capture.is_file(), f"missing row request capture: {request_capture}")
        require(record_capture.is_file(), f"missing row record capture: {record_capture}")
        require(raw_capture.read_text(encoding="utf-8") == row.get("raw_completion"), "raw capture differs from output row")
        require(read_json(request_capture).get("candidate_id") == row.get("candidate_id"), "candidate request differs from output row")
        require(read_json(record_capture) == row, "record capture differs from output row")

    print(
        json.dumps(
            {
                "status": "verified",
                "generated": len(rows),
                "parsed": parsed,
                "accepted": len(accepted),
                "code_archive_sha256": args.expected_code_sha256,
                "input_sha256": args.expected_input_sha256,
                "model_revision": EXPECTED_REVISION,
                "max_tokens": args.expected_max_tokens,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
