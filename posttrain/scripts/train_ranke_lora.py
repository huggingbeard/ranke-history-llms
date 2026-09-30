import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

DEFAULT_BASE_REVISION = "fb4cede16ed59e1249ebeba2b4e5b9ec2a560642"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(log, stage, **fields):
    row = {"time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "pid": os.getpid(), "stage": stage, **fields}
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=1929)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mode", choices=["qa", "prose"], default="qa")
    ap.add_argument("--verify-reload", action="store_true")
    args = ap.parse_args()

    base_path = Path(args.base)
    if not base_path.is_dir():
        raise FileNotFoundError(base_path)
    if args.base_revision not in base_path.parts:
        raise RuntimeError(
            f"base revision lock failed: expected {args.base_revision} in {base_path}"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "progress.jsonl"
    record(log, "started", data=args.data, data_sha256=sha256(args.data),
           base=str(base_path), base_revision=args.base_revision)
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.set_num_threads(1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    assert torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rows = []
    with Path(args.data).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]
    assert rows

    examples = []
    clipped = 0
    for row in rows:
        if args.mode == "qa":
            prefix = f"Question: {row['question'].strip()}\nAnswer:"
            prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
            answer_ids = tokenizer.encode(" " + row["answer"].strip(), add_special_tokens=False)
            answer_ids.append(tokenizer.eos_token_id)
            if len(prompt_ids) + len(answer_ids) > args.max_length:
                answer_ids = answer_ids[:max(1, args.max_length - len(prompt_ids))]
                if answer_ids[-1] != tokenizer.eos_token_id:
                    answer_ids[-1] = tokenizer.eos_token_id
                clipped += 1
            ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
        else:
            ids = list(row["token_ids"])
            if len(ids) > args.max_length:
                ids = ids[:args.max_length]
                clipped += 1
            labels = list(ids)
        assert len(ids) == len(labels) and any(x != -100 for x in labels)
        examples.append((torch.tensor([ids], dtype=torch.long),
                         torch.tensor([labels], dtype=torch.long)))

    versions = {p: importlib.metadata.version(p) for p in
                ["torch", "transformers", "peft", "accelerate", "safetensors"]}
    record(log, "data_ready", rows=len(rows), mode=args.mode, clipped=clipped,
           target_tokens=sum((lab != -100).sum().item() for _, lab in examples),
           versions=versions, gpu=torch.cuda.get_device_name())

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_path, local_files_only=True, dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map={"": 0}, attn_implementation="sdpa")
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    base.config.use_cache = False
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                        target_modules=["q_proj", "v_proj"], bias="none",
                        task_type="CAUSAL_LM")
    model = get_peft_model(base, config)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(p.dtype in (torch.bfloat16, torch.float32) for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    record(log, "model_ready", trainable_parameters=sum(p.numel() for p in trainable),
           total_parameters=sum(p.numel() for p in model.parameters()),
           allocated_gb=round(torch.cuda.memory_allocated() / 2**30, 3))

    updates = 0
    loss_history = []
    began = time.monotonic()
    model.train()
    for epoch in range(1, args.epochs + 1):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(examples), generator=generator).tolist()
        epoch_losses = []
        for start in range(0, len(order), args.grad_accum):
            group = order[start:start + args.grad_accum]
            optimizer.zero_grad(set_to_none=True)
            group_losses = []
            for idx in group:
                input_ids, labels = examples[idx]
                input_ids = input_ids.to("cuda")
                labels = labels.to("cuda")
                attention_mask = torch.ones_like(input_ids)
                loss = model(input_ids=input_ids, attention_mask=attention_mask,
                             labels=labels, use_cache=False).loss
                assert torch.isfinite(loss)
                (loss / len(group)).backward()
                group_losses.append(loss.detach().float().item())
            grad = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            assert torch.isfinite(grad)
            optimizer.step()
            updates += 1
            mean_loss = sum(group_losses) / len(group_losses)
            epoch_losses.append(mean_loss)
            loss_history.append(mean_loss)
            if updates == 1 or updates % 20 == 0:
                record(log, "update", epoch=epoch, update=updates,
                       loss=round(mean_loss, 6), grad_norm=round(float(grad), 6),
                       peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3))
        checkpoint = out / f"epoch_{epoch}"
        model.save_pretrained(checkpoint)
        record(log, "epoch_saved", epoch=epoch, updates=updates,
               mean_loss=sum(epoch_losses) / len(epoch_losses), checkpoint=str(checkpoint))

    final_dir = out / "adapter_final"
    model.save_pretrained(final_dir)
    reload_check = None
    if args.verify_reload:
        if args.mode != "qa":
            raise ValueError("reload generation check requires QA data")
        del model, base, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        check_base = AutoModelForCausalLM.from_pretrained(
            base_path, local_files_only=True, dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map={"": 0}, attn_implementation="sdpa")
        check_model = PeftModel.from_pretrained(check_base, final_dir).eval()
        check_prompt = f"Question: {rows[0]['question'].strip()}\nAnswer:"
        check_ids = tokenizer(check_prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.inference_mode():
            generated = check_model.generate(
                input_ids=check_ids, max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
        new_ids = generated[0, check_ids.shape[1]:].tolist()
        check_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if not check_text:
            raise RuntimeError("reloaded adapter generated an empty answer")
        reload_check = {
            "status": "passed", "new_tokens": len(new_ids),
            "answer_sha256": hashlib.sha256(check_text.encode("utf-8")).hexdigest()
        }
        (out / "reload_check.json").write_text(
            json.dumps({**reload_check, "answer": check_text}, indent=2), encoding="utf-8")
        record(log, "reload_check_passed", **reload_check)
    receipt = {
        "status": "complete", "data": args.data, "data_sha256": sha256(args.data),
        "base": str(base_path), "base_revision": args.base_revision,
        "rows": len(rows), "mode": args.mode, "epochs": args.epochs, "updates": updates,
        "max_length": args.max_length, "grad_accum": args.grad_accum,
        "lr": args.lr, "seed": args.seed, "clipped": clipped,
        "first_20_mean_loss": sum(loss_history[:20]) / min(20, len(loss_history)),
        "last_20_mean_loss": sum(loss_history[-20:]) / min(20, len(loss_history)),
        "adapter": str(final_dir), "seconds": time.monotonic() - began,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
        "versions": versions, "base_load": "nf4-double-quant-bfloat16-compute",
        "reload_check": reload_check,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    record(log, "complete", **receipt)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise
