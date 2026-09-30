import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


CONDITIONS = ["base", "qa_300", "qa_1000", "qa_3000"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True)
    ap.add_argument("--battery-sha256", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--matrix-sha256", required=True)
    ap.add_argument("--seed", type=int, default=1929)
    args = ap.parse_args()

    if sha256(args.battery) != args.battery_sha256.lower():
        raise ValueError("frozen battery hash mismatch")
    if sha256(args.matrix) != args.matrix_sha256.lower():
        raise ValueError("matrix hash mismatch")
    battery = load_jsonl(args.battery)
    rows = load_jsonl(args.matrix)
    by_id = defaultdict(dict)
    for row in rows:
        condition = row["condition"]
        if condition not in CONDITIONS:
            raise ValueError(f"unexpected condition: {condition}")
        if condition in by_id[row["prompt_id"]]:
            raise ValueError(f"duplicate: {row['prompt_id']} {condition}")
        by_id[row["prompt_id"]][condition] = row
    if len(rows) != len(battery) * len(CONDITIONS):
        raise ValueError(f"wrong matrix row count: {len(rows)}")

    rng = random.Random(args.seed)
    blinded, secret = [], []
    for item in battery:
        item_id = item["id"]
        if set(by_id[item_id]) != set(CONDITIONS):
            raise ValueError(f"incomplete item: {item_id} {sorted(by_id[item_id])}")
        shuffled = CONDITIONS.copy()
        rng.shuffle(shuffled)
        labels = [f"M{i + 1}" for i in range(len(shuffled))]
        mapping = dict(zip(labels, shuffled))
        blinded.append({
            "item_id": item_id,
            "prompt": item["prompt"],
            "date_limit": item["date_limit"],
            "checks": item["criteria"],
            "replies": [{"label": label, "reply": by_id[item_id][condition]["answer"]}
                        for label, condition in mapping.items()],
        })
        secret.append({"item_id": item_id, "labels": mapping})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blind_path = out / "blinded_items.jsonl"
    secret_path = out / "secret_label_map.jsonl"
    blind_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in blinded), encoding="utf-8")
    secret_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in secret), encoding="utf-8")
    receipt = {
        "seed": args.seed,
        "items": len(blinded),
        "conditions": CONDITIONS,
        "battery": str(Path(args.battery)),
        "battery_sha256": sha256(args.battery),
        "matrix": str(Path(args.matrix)),
        "matrix_sha256": sha256(args.matrix),
        "blinded_items_sha256": sha256(blind_path),
        "secret_label_map_sha256": sha256(secret_path),
    }
    (out / "blinding_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
