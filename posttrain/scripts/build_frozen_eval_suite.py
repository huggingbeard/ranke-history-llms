#!/usr/bin/env python3
"""Combine two already-frozen batteries without changing their prompts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--period-chat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--receipt", required=True)
    args = ap.parse_args()

    history_path = Path(args.history)
    period_path = Path(args.period_chat)
    out_path = Path(args.out)
    receipt_path = Path(args.receipt)
    if out_path.exists() or receipt_path.exists():
        raise FileExistsError("frozen suite or receipt already exists")

    merged: list[dict] = []
    for suite, prefix, path in (
        ("history_reasoning_36", "H36", history_path),
        ("period_chat_48", "P48", period_path),
    ):
        for source_item in load(path):
            item = dict(source_item)
            original_id = str(item["id"])
            item["source_battery_id"] = original_id
            item["id"] = f"{prefix}_{original_id}"
            item["suite"] = suite
            if "date_limit" not in item and "date" in item:
                item["date_limit"] = item.pop("date")
            item.setdefault("task_count", 1)
            merged.append(item)

    ids = [item["id"] for item in merged]
    prompts = [item["prompt"] for item in merged]
    if len(ids) != 84 or len(set(ids)) != 84 or len(set(prompts)) != 84:
        raise ValueError("expected 84 unique IDs and prompts")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in merged)
    out_path.write_text(data, encoding="utf-8", newline="\n")
    receipt = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "frozen_before_any_Ranke-1929_SFT",
        "transformation": "namespace IDs and concatenate only; prompt and criteria text unchanged",
        "sources": [
            {"suite": "history_reasoning_36", "path": str(history_path), "rows": 36, "sha256": sha(history_path)},
            {"suite": "period_chat_48", "path": str(period_path), "rows": 48, "sha256": sha(period_path)},
        ],
        "output": {"path": str(out_path), "rows": len(merged), "sha256": sha(out_path)},
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
