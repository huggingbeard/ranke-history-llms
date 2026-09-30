from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-source", type=int, default=2)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input).open(encoding="utf-8")
        if line.strip()
    ]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source_id"]].append(row)

    selected: list[dict] = []
    for source_id in sorted(groups):
        group = groups[source_id]
        for ordinal in range(1, args.per_source + 1):
            index = (ordinal * len(group)) // (args.per_source + 1)
            selected.append(group[min(index, len(group) - 1)])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "rows": len(selected), "sources": len(groups)}))


if __name__ == "__main__":
    main()
