#!/usr/bin/env python3
"""Plan v3: answer the 84 frozen eval prompts with one or more saved LoRA adapters.
Decoding identical to round 2 (scripts/generate_blind_eval.py): 4-bit NF4 base, "Question: ...\\nAnswer:",
greedy, batch 8 with left padding, per-item max_new_tokens, stop at EOS.
Usage: --adapter NAME=PATH (repeatable). One JSONL row per prompt x adapter, written as each batch finishes;
finished adapters are skipped on restart."""
import argparse, json, os, time
from pathlib import Path

BASE_REV = "fb4cede16ed59e1249ebeba2b4e5b9ec2a560642"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--base-revision", default=BASE_REV)   # night 2026-09-29: the continued-pretraining model has its own id
    a = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    torch.manual_seed(1929)
    assert a.base_revision in Path(a.base).parts
    battery = [json.loads(l) for l in open(a.battery, encoding="utf-8") if l.strip()]
    assert len(battery) == 84
    tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True)
    tok.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tok.padding_side = "left"
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out, encoding="utf-8"):
            r = json.loads(l); done.add(r["condition"])
    for spec in a.adapter:
        name, path = spec.split("=", 1)
        if name in done:
            continue
        base = AutoModelForCausalLM.from_pretrained(a.base, local_files_only=True, dtype=torch.bfloat16,
                                                    quantization_config=q, device_map={"": 0},
                                                    attn_implementation="sdpa").eval()
        model = base if path == "NONE" else PeftModel.from_pretrained(base, path).eval()   # NAME=NONE: model without adapter
        rows = []
        for s in range(0, 84, a.batch_size):
            b = battery[s:s + a.batch_size]
            rend = [f"Question: {x['prompt'].strip()}\nAnswer:" for x in b]
            enc = tok(rend, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            t0 = time.time()
            with torch.no_grad():
                g = model.generate(**enc, do_sample=False, max_new_tokens=max(int(x["max_new_tokens"]) for x in b),
                                   pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
            w = enc["input_ids"].shape[1]
            for k, (x, rp) in enumerate(zip(b, rend)):
                c = g[k, w:][:int(x["max_new_tokens"])]
                e = (c == tok.eos_token_id).nonzero(as_tuple=False)
                term = "length"
                if e.numel():
                    c = c[:int(e[0].item()) + 1]; term = "eos"
                rows.append({"prompt_id": x["id"], "condition": name, "question": x["prompt"], "rendered_prompt": rp,
                             "answer": tok.decode(c, skip_special_tokens=True).strip(), "new_tokens": int(c.numel()),
                             "termination": term, "max_new_tokens": int(x["max_new_tokens"]),
                             "batch_seconds": round(time.time() - t0, 2), "adapter": path, "base_revision": a.base_revision})
        with out.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps({"stage": "eval_done", "condition": name, "rows": len(rows)}), flush=True)
        del model, base
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
