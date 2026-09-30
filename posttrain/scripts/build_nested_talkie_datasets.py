#!/usr/bin/env python3
"""Build fixed nested 300, 1000, and 3000 Talkie SFT sets."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path


def h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def interleave(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], deque] = defaultdict(deque)
    for row in sorted(rows, key=lambda r: h(r["candidate_id"] + "::1929")):
        groups[(row["source_id"], row["form"])].append(row)
    keys = sorted(groups, key=lambda key: h("::".join(key) + "::order"))
    ordered = []
    while groups:
        for key in list(keys):
            if key not in groups:
                continue
            ordered.append(groups[key].popleft())
            if not groups[key]:
                del groups[key]
        keys = [key for key in keys if key in groups]
    return ordered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=[300, 1000, 3000])
    args = ap.parse_args()
    with Path(args.accepted).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate candidate IDs")
    need = max(args.sizes)
    if len(rows) < need:
        raise ValueError(f"need {need} accepted rows, found {len(rows)}")

    # Prefer strong records, and use at most one record per passage before second variants.
    strong = [r for r in rows if r["luna_selection"]["quality"] == "strong"]
    usable = [r for r in rows if r["luna_selection"]["quality"] == "usable"]
    quality_order = interleave(strong) + interleave(usable)
    first, repeats, seen_passages = [], [], set()
    for row in quality_order:
        if row["passage_id"] in seen_passages:
            repeats.append(row)
        else:
            first.append(row)
            seen_passages.add(row["passage_id"])
    master = first + repeats
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"input": args.accepted, "input_rows": len(rows), "ordering": "strong_then_usable; balanced source-form hash round-robin; unique passages first", "sets": {}}
    previous_ids: set[str] = set()
    for size in sorted(args.sizes):
        selected = master[:size]
        ids = {row["candidate_id"] for row in selected}
        if not previous_ids.issubset(ids):
            raise AssertionError("sets are not nested")
        previous_ids = ids
        path = out_dir / f"talkie_qa_{size}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
        manifest["sets"][str(size)] = {
            "path": str(path), "rows": len(selected),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "unique_passages": len({r["passage_id"] for r in selected}),
            "sources": dict(Counter(r["source_id"] for r in selected)),
            "forms": dict(Counter(r["form"] for r in selected)),
            "quality": dict(Counter(r["luna_selection"]["quality"] for r in selected)),
            "reasoning_present": dict(Counter(r["luna_selection"]["reasoning_present"] for r in selected)),
        }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
