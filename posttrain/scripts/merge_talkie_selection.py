#!/usr/bin/env python3
"""Join complete Luna decisions to unchanged Talkie records."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--accepted", required=True)
    ap.add_argument("--receipt", required=True)
    args = ap.parse_args()
    rows = read_jsonl(Path(args.candidates))
    found: dict[str, dict] = {}
    for path in sorted(Path(args.decisions).glob("batch_*/decision.json")):
        for decision in json.loads(path.read_text(encoding="utf-8"))["decisions"]:
            if decision["id"] in found:
                raise ValueError(f"duplicate decision {decision['id']}")
            found[decision["id"]] = decision
    ids = [row["candidate_id"] for row in rows]
    missing = sorted(set(ids) - found.keys())
    extra = sorted(found.keys() - set(ids))
    if missing or extra:
        raise ValueError(f"decision mismatch missing={len(missing)} extra={len(extra)}")
    merged = []
    accepted = []
    for row in rows:
        item = dict(row)
        item["luna_selection"] = found[row["candidate_id"]]
        merged.append(item)
        if item["luna_selection"]["accept"]:
            accepted.append(item)
    out_path, accepted_path = Path(args.output), Path(args.accepted)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged), encoding="utf-8")
    accepted_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted), encoding="utf-8")
    receipt = {
        "candidates": len(rows), "decisions": len(found), "accepted": len(accepted),
        "accept_rate": len(accepted) / len(rows) if rows else 0,
        "quality": dict(Counter(d["quality"] for d in found.values())),
        "rejection_reason": dict(Counter(d["rejection_reason"] for d in found.values())),
        "merged_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "accepted_sha256": hashlib.sha256(accepted_path.read_bytes()).hexdigest(),
    }
    Path(args.receipt).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
