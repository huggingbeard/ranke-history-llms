import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate(result, expected_item, expected_labels):
    if result.get("item_id") != expected_item:
        raise ValueError(f"wrong item ID {result.get('item_id')} != {expected_item}")
    grades = result.get("grades", [])
    labels = [grade.get("label") for grade in grades]
    expected_count = len(expected_labels)
    if len(grades) != expected_count or set(labels) != set(expected_labels) or len(set(labels)) != expected_count:
        raise ValueError(f"grade labels invalid for {expected_item}: {labels}")
    for grade in grades:
        for axis in ("directness", "substance", "coherence", "naturalness", "period_fit"):
            value = grade[axis]
            if not isinstance(value, int) or not 0 <= value <= 4:
                raise ValueError(f"bad {axis}: {value}")
        computed = sum(grade[a] for a in ("directness", "substance", "coherence", "naturalness", "period_fit")) * 5
        if grade["fatal_failure"]:
            computed = min(computed, 25)
        grade["judge_reported_overall_0_100"] = grade["overall_0_100"]
        grade["overall_0_100"] = computed
    ranked = [label for tier in result.get("ranking", []) for label in tier]
    if len(ranked) != expected_count or set(ranked) != set(expected_labels):
        raise ValueError(f"ranking labels invalid for {expected_item}: {ranked}")
    return result


def run_one(item, out_root, schema, judge_prompt, node, codex_js, model, effort):
    tag = item["item_id"]
    out = out_root / tag
    out.mkdir(parents=True, exist_ok=True)
    final_path = out / "judgment.json"
    receipt_path = out / "receipt.json"
    if final_path.exists() and receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    request = judge_prompt.rstrip() + "\n\nITEM AND ANONYMIZED REPLIES:\n" + json.dumps(item, ensure_ascii=False)
    request_bytes = request.encode("utf-8")
    (out / "request.txt").write_bytes(request_bytes)
    began = time.time()
    cmd = [node, codex_js, "exec", "-", "--model", model, "--ephemeral", "--ignore-user-config",
           "--ignore-rules", "--sandbox", "read-only", "--skip-git-repo-check",
           "--output-schema", str(schema), "--output-last-message", str(final_path),
           "--json", "-c", f'model_reasoning_effort="{effort}"']
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    with (out / "events.jsonl").open("w", encoding="utf-8") as stdout, \
         (out / "stderr.txt").open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, input=request, text=True, encoding="utf-8", stdout=stdout,
                              stderr=stderr, cwd=out, env=env, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"{tag}: codex exit {proc.returncode}")
    result = json.loads(final_path.read_text(encoding="utf-8"))
    result = validate(result, tag, [reply["label"] for reply in item["replies"]])
    final_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt = {
        "item_id": tag,
        "seconds": round(time.time() - began, 2),
        "provider": "OpenAI via Codex CLI",
        "model": model,
        "reasoning_effort": effort,
        "temperature": None,
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "request_sha256": sha256_bytes(request_bytes),
        "response_sha256": sha256_bytes(final_path.read_bytes()),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--judge-prompt", required=True)
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-new-items", type=int)
    ap.add_argument("--node", default=r"C:\Program Files\nodejs\node.exe")
    ap.add_argument("--codex-js", default=r"C:\Users\hvoth\AppData\Roaming\npm\node_modules\@openai\codex\bin\codex.js")
    args = ap.parse_args()
    items = load_jsonl(args.items)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pending = [item for item in items if not ((out / item["item_id"] / "judgment.json").exists() and
                                               (out / item["item_id"] / "receipt.json").exists())]
    if args.max_new_items is not None:
        pending = pending[:args.max_new_items]
    prompt = Path(args.judge_prompt).read_text(encoding="utf-8-sig")
    receipts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, item, out, Path(args.schema), prompt, args.node,
                               args.codex_js, args.model, args.effort) for item in pending]
        for future in concurrent.futures.as_completed(futures):
            receipt = future.result()
            receipts.append(receipt)
            print(json.dumps(receipt), flush=True)
    all_receipts = []
    for item in items:
        path = out / item["item_id"] / "receipt.json"
        if path.exists():
            all_receipts.append(json.loads(path.read_text(encoding="utf-8")))
    (out / "receipts.json").write_text(json.dumps(all_receipts, indent=2), encoding="utf-8")
    print(json.dumps({"new": len(receipts), "complete": len(all_receipts), "expected": len(items),
                      "model": args.model, "effort": args.effort}), flush=True)


if __name__ == "__main__":
    main()
