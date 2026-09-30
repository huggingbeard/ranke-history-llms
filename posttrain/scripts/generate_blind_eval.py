import argparse
import datetime
import hashlib
import json
import os
import time
from pathlib import Path

LOCKED_REVISION = "fb4cede16ed59e1249ebeba2b4e5b9ec2a560642"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_battery(path, default_cap):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                prompt_key = next((key for key in ("question", "prompt", "input") if key in row), None)
                id_key = next((key for key in ("id", "prompt_id", "item_id") if key in row), None)
                if not prompt_key or not id_key:
                    raise KeyError("battery needs an id and question/prompt/input field")
                cap = int(row.get("max_new_tokens", default_cap))
                if cap < 1:
                    raise ValueError(f"invalid max_new_tokens for {row[id_key]}: {cap}")
                rows.append((str(row[id_key]), row[prompt_key], cap))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-revision", default=LOCKED_REVISION)
    ap.add_argument("--expected-prompts", type=int, default=36)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.set_num_threads(1)
    torch.manual_seed(1929)
    battery = read_battery(args.battery, args.max_new_tokens)
    if len(battery) != args.expected_prompts:
        raise ValueError(f"expected {args.expected_prompts} prompts, found {len(battery)}")
    base_path = Path(args.base)
    if args.base_revision not in base_path.parts:
        raise ValueError(f"base path is not locked to {args.base_revision}: {base_path}")
    root = Path(args.root)
    out_dir = root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    conditions = [("base", None)]
    for name in ("qa_300", "qa_1000", "qa_3000"):
        adapter = root / "adapters" / name / "adapter_final"
        if adapter.exists():
            conditions.append((name, adapter))

    manifest = {"battery": str(Path(args.battery)), "battery_sha256": sha256(args.battery),
                "base": str(base_path), "base_revision": args.base_revision,
                "conditions": [], "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    expected = len(battery) * len(conditions)
    matrix_path = out_dir / "blind_output_matrix.jsonl"
    with matrix_path.open("w", encoding="utf-8", newline="\n") as sink:
        for condition, adapter in conditions:
            base = AutoModelForCausalLM.from_pretrained(
                base_path, local_files_only=True, dtype=torch.bfloat16,
                quantization_config=quantization_config,
                device_map={"": 0}, attn_implementation="sdpa").eval()
            model = PeftModel.from_pretrained(base, adapter).eval() if adapter else base
            began_condition = time.monotonic()
            for batch_start in range(0, len(battery), args.batch_size):
                batch = battery[batch_start:batch_start + args.batch_size]
                rendered = [f"Question: {question.strip()}\nAnswer:" for _, question, _ in batch]
                inputs = tokenizer(rendered, return_tensors="pt", padding=True,
                                   add_special_tokens=False).to("cuda")
                began = time.monotonic()
                with torch.no_grad():
                    batch_cap = max(cap for _, _, cap in batch)
                    generated = model.generate(**inputs, do_sample=False,
                                               max_new_tokens=batch_cap,
                                               pad_token_id=tokenizer.eos_token_id,
                                               eos_token_id=tokenizer.eos_token_id)
                batch_seconds = time.monotonic() - began
                input_width = inputs["input_ids"].shape[1]
                for offset, ((prompt_id, question, item_cap), rendered_prompt) in enumerate(zip(batch, rendered)):
                    continuation = generated[offset, input_width:][:item_cap]
                    eos_positions = (continuation == tokenizer.eos_token_id).nonzero(as_tuple=False)
                    if eos_positions.numel():
                        continuation = continuation[:int(eos_positions[0].item()) + 1]
                        termination = "eos"
                    else:
                        termination = "length"
                    row = {
                        "prompt_id": prompt_id, "condition": condition,
                        "question": question, "rendered_prompt": rendered_prompt,
                        "answer": tokenizer.decode(continuation, skip_special_tokens=True).strip(),
                        "continuation_token_ids": continuation.cpu().tolist(),
                        "new_tokens": int(continuation.numel()),
                        "termination": termination,
                        "max_new_tokens": item_cap,
                        "seconds": batch_seconds / len(batch),
                        "batch_seconds": batch_seconds,
                        "batch_size": len(batch),
                        "base_revision": args.base_revision,
                        "base_load": "nf4-double-quant-bfloat16-compute",
                        "adapter": str(adapter) if adapter else None,
                    }
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                os.fsync(sink.fileno())
                print(json.dumps({"stage": "generated", "condition": condition,
                                  "completed": batch_start + len(batch), "total": len(battery),
                                  "batch_seconds": batch_seconds, "batch_cap": batch_cap}), flush=True)
            manifest["conditions"].append({"name": condition, "adapter": str(adapter) if adapter else None,
                                            "seconds": time.monotonic() - began_condition})
            del model, base
            torch.cuda.empty_cache()
    count = sum(1 for line in matrix_path.open(encoding="utf-8") if line.strip())
    assert count == expected, (count, expected)
    manifest.update({"rows": count, "expected_rows": expected,
                     "matrix": str(matrix_path), "matrix_sha256": sha256(matrix_path)})
    manifest_path = out_dir / "blind_output_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "matrix_hashed", "rows": count,
                      "sha256": manifest["matrix_sha256"], "manifest": str(manifest_path)}), flush=True)


if __name__ == "__main__":
    main()
