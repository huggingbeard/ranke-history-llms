#!/usr/bin/env python3
"""Ceiling check (JV 2026-09-29 14:09, priority 1): Talkie-1930-13B-IT answers the same 84 frozen
eval prompts. Same decoding rule as the Ranke eval (greedy, per-item max_new_tokens from the frozen
file), Talkie's own chat template, stop at its end token. One JSONL row per prompt, same fields as
generate_blind_eval.py rows, condition = "talkie13b". Uses the exact-checked KV-cache decoder
(float32 test: identical to Talkie's own code)."""
import argparse, hashlib, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import generate_talkie_pairs as v1  # noqa: E402
from talkie_kv import KVRunner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    from talkie.chat import format_prompt, truncate_at_stop
    tk, receipt = v1.load_pinned_talkie("talkie-1930-13b-it", v1.MODEL_REVISION, "cuda", a.cache_dir)
    runner = KVRunner(tk.model)
    items = [json.loads(l) for l in open(a.battery, encoding="utf-8") if l.strip()]
    assert len(items) == 84
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as sink:
        for s in range(0, len(items), a.batch_size):
            chunk = items[s:s + a.batch_size]
            ids = [tk.tokenizer.encode(format_prompt(it["prompt"].strip()), allowed_special="all") for it in chunk]
            cap = max(int(it["max_new_tokens"]) for it in chunk)
            t0 = time.time()
            toks, why = runner.generate(ids, cap, 1.0, None, None, [0] * len(chunk), stop_ids=tk._stop_ids,
                                        greedy=True)
            secs = time.time() - t0
            for it, tt, w in zip(chunk, toks, why):
                tt = tt[:int(it["max_new_tokens"])]
                text = truncate_at_stop(tk.tokenizer.decode(tt))[0].strip()
                sink.write(json.dumps({"prompt_id": it["id"], "condition": "talkie13b", "question": it["prompt"],
                                       "rendered_prompt": format_prompt(it["prompt"].strip()), "answer": text,
                                       "new_tokens": len(tt), "termination": "eos" if w == "stop" else "length",
                                       "max_new_tokens": int(it["max_new_tokens"]), "batch_seconds": round(secs, 2),
                                       "model": "talkie-lm/talkie-1930-13b-it", "model_revision": v1.MODEL_REVISION,
                                       "decoding": "greedy, kv_cache_batched_v2"}, ensure_ascii=False) + "\n")
            sink.flush()
            print(json.dumps({"stage": "batch", "done": s + len(chunk), "seconds": round(secs, 1)}), flush=True)
    print(json.dumps({"stage": "done", "rows": 84, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                      "talkie": receipt}), flush=True)


if __name__ == "__main__":
    main()
