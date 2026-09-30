#!/usr/bin/env python3
"""Stage A (plan v2.3): Talkie writes question/answer records, batched with a KV cache.

Locked prompt, forms, parser and loader are imported unchanged from
scripts/generate_talkie_pairs.py. Every raw completion is saved (one JSONL row per
candidate, appended and flushed per batch; resume skips done candidate_ids).

Modes:
  validate-kv : plan test V1 (cached vs original uncached forward, 8 prompts)
  generate    : write candidates for --variants (0,1[,2])
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import generate_talkie_pairs as v1  # noqa: E402
from talkie_kv import KVRunner, uncached_greedy, cached_first_logits  # noqa: E402

FORMS = v1.FORMS
ANSWER_BRIDGE = "</question>\n<answer>"  # plan v2.9: fixed non-semantic delimiter inserted by code


def form_for_v2(passage_id: str, variant: int):
    """Variants 0,1 = v1 rule exactly. Variant 2 (yield top-up) = next form after variant 0,
    because the v1 rule (start + 3*variant) % 6 would repeat variant 0's form."""
    if variant in (0, 1):
        return v1.form_for(passage_id, variant)
    digest = hashlib.sha256(passage_id.encode("utf-8")).digest()
    start = int.from_bytes(digest[:2], "big") % len(FORMS)
    return FORMS[(start + 1) % len(FORMS)]


def load(args):
    t0 = time.time()
    tk, receipt = v1.load_pinned_talkie("talkie-1930-13b-it", v1.MODEL_REVISION, "cuda", args.cache_dir)
    print(json.dumps({"stage": "talkie_loaded", "seconds": round(time.time() - t0, 1), **receipt}), flush=True)
    return tk, receipt


def encode_prompt(tk, prompt):
    from talkie.chat import format_prompt
    return tk.tokenizer.encode(format_prompt(prompt) + v1.ASSISTANT_PREFIX, allowed_special="all")


def read_passages(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def validate_kv(args):
    import torch
    import talkie_kv
    tk, receipt = load(args)
    if args.fp32:
        talkie_kv.AUTOCAST = False
        tk.model.float()
        tk.model.cos, tk.model.sin = tk.model.cos.float(), tk.model.sin.float()
        print("float32 diagnostic mode", flush=True)
    runner = KVRunner(tk.model)
    passages = read_passages(args.input)
    passages.sort(key=lambda r: len(r["text"]))
    idx = [0, len(passages) // 6, len(passages) // 3, len(passages) // 2,
           2 * len(passages) // 3, 5 * len(passages) // 6, len(passages) - 1]
    prompts = []
    for i in idx:
        p = passages[i]
        prompts.append(encode_prompt(tk, v1.build_prompt(p, form_for_v2(p["passage_id"], 0)[1])))
    # one very short prompt (~200 tokens): cut passage text
    p = dict(passages[len(passages) // 2]); p["text"] = p["text"][:300]
    prompts.append(encode_prompt(tk, v1.build_prompt(p, FORMS[0][1])))
    lens = [len(x) for x in prompts]
    n = 64
    cached, _ = runner.generate(prompts, n, 1.0, None, None, [0] * len(prompts),
                                stop_ids=set(), greedy=True)
    first = cached_first_logits(runner, prompts)
    rows, matches = [], 0
    for b, ids in enumerate(prompts):
        ref, ref_logits = uncached_greedy(tk.model, ids, n)
        maxdiff = float((first[b] - ref_logits).abs().max())
        agree = sum(1 for a, c in zip(ref, cached[b]) if a == c)
        first_div = next((i for i, (a, c) in enumerate(zip(ref, cached[b])) if a != c), None)
        ok = ref == cached[b][:n]
        matches += ok
        rows.append({"prompt_tokens": lens[b], "exact_match_64": ok, "agree_tokens": agree,
                     "first_divergence": first_div, "first_logits_max_abs_diff": maxdiff,
                     "ref_text": tk.tokenizer.decode(ref), "cached_text": tk.tokenizer.decode(cached[b])})
        print(json.dumps(rows[-1])[:400], flush=True)
    # (b) 64 real prompts: batch of 32 (left padding) vs batch size 1 (no padding), greedy 128
    import random as _r
    rng = _r.Random(1929)
    sample = rng.sample(passages, 64)
    real = [encode_prompt(tk, v1.build_prompt(p, form_for_v2(p["passage_id"], 0)[1])) for p in sample]
    b_rows, b_same = [], 0
    bsz = 8 if args.fp32 else 32
    for s in range(0, 64 if not args.fp32 else 16, bsz):
        chunk = real[s:s + bsz]
        batched, _ = runner.generate(chunk, 128, 1.0, None, None, [0] * len(chunk), stop_ids=set(), greedy=True)
        for ids, bt in zip(chunk, batched):
            single, _ = runner.generate([ids], 128, 1.0, None, None, [0], stop_ids=set(), greedy=True)
            same = single[0] == bt
            b_same += same
            div = next((i for i, (x, y) in enumerate(zip(single[0], bt)) if x != y), None)
            b_rows.append({"prompt_tokens": len(ids), "identical_128": same, "first_divergence": div})
    # (c) speed probe: 32 real prompts x 512 tokens sampled (full record length)
    if args.skip_speed:
        real = real[:4]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.time()
    runner.generate(real[:32], 512, 0.55, 0.9, 40, list(range(32)), stop_ids=set())
    torch.cuda.synchronize()
    secs = time.time() - t0
    speed = {"batch": 32, "new_tokens": 512, "seconds": round(secs, 2),
             "projected_hours_4154": round(4154 / 32 * secs / 3600, 2),
             "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    pa = matches >= 7 and max(r["first_logits_max_abs_diff"] for r in rows) < 0.5
    pb = b_same >= 60
    pc = speed["projected_hours_4154"] <= 2.0
    passed = pa and pb and pc
    out = {"test": "V1_cached_decoder",
           "pass_rule": "(a) >=7/8 exact greedy 64 vs uncached and max logit diff < 0.5; "
                        "(b) >=60/64 identical greedy 128, batch-32 vs batch-1; (c) projected <= 2 h",
           "exact_matches": matches, "n": len(prompts), "passed_a": pa,
           "batch_vs_single_identical": b_same, "passed_b": pb, "passed_c": pc,
           "passed": passed, "speed": speed, "talkie_receipt": receipt, "rows": rows, "rows_b": b_rows}
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)


def speed(args):
    """Same speed probe as test V1 on RunPod (32 real prompts x 512 sampled tokens: 37.5 s on A100),
    plus a profiler table of 20 decode steps, to explain cluster slowness."""
    import torch, random as _r
    tk, _ = load(args)
    runner = KVRunner(tk.model)
    passages = read_passages(args.input)
    sample = _r.Random(1929).sample(passages, 64)
    real = [encode_prompt(tk, v1.build_prompt(p, form_for_v2(p["passage_id"], 0)[1])) for p in sample]
    print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                      "threads": torch.get_num_threads(), "sdpa_flash": torch.backends.cuda.flash_sdp_enabled(),
                      "sdpa_mem_eff": torch.backends.cuda.mem_efficient_sdp_enabled()}), flush=True)
    runner.generate(real[:4], 8, 0.55, 0.9, 40, [0, 1, 2, 3], stop_ids=set())  # warm-up
    torch.cuda.synchronize(); t0 = time.time()
    runner.generate(real[:32], 512, 0.55, 0.9, 40, list(range(32)), stop_ids=set())
    torch.cuda.synchronize()
    print(json.dumps({"speed_probe_seconds_32x512": round(time.time() - t0, 2), "runpod_a100_reference": 37.49}), flush=True)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        runner.generate(real[:32], 20, 0.55, 0.9, 40, list(range(32)), stop_ids=set())
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15), flush=True)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10), flush=True)


def generate(args):
    import torch
    from talkie.chat import truncate_at_stop  # chat-template leak guard, as in _stream_raw
    tk, receipt = load(args)
    runner = KVRunner(tk.model)
    passages = read_passages(args.input)
    if args.passage_ids:
        keep = set(Path(args.passage_ids).read_text(encoding="utf-8").split())
        passages = [p for p in passages if p["passage_id"] in keep]
    variants = [int(v) for v in args.variants.split(",")]
    out_path = Path(args.output)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["candidate_id"] for l in open(out_path, encoding="utf-8") if l.strip()}
    jobs = []
    for variant in variants:  # all passages for variant 0, then 1, then 2 (plan: time-limit order)
        for p in passages:
            cid = f"{p['passage_id']}::v{variant}"
            if cid in done:
                continue
            form_name, instr = form_for_v2(p["passage_id"], variant)
            prompt = v1.build_prompt(p, instr)
            seed = args.seed + int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16)
            jobs.append((cid, p, variant, form_name, prompt, encode_prompt(tk, prompt), seed))
    if args.limit:
        jobs = jobs[:args.limit]
    print(json.dumps({"stage": "jobs", "todo": len(jobs), "already_done": len(done)}), flush=True)
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    # within each variant, group similar lengths to reduce padding
    order = sorted(range(len(jobs)), key=lambda i: (jobs[i][2], len(jobs[i][5])))
    stop_text = lambda s: "</answer>" in s or truncate_at_stop(s)[1]
    t_start = time.time(); n_done = 0; stopped_early = False
    with out_path.open("a", encoding="utf-8") as sink:
        for s in range(0, len(order), args.batch_size):
            if deadline and time.time() > deadline:
                print(json.dumps({"stage": "time_limit_stop", "done": n_done}), flush=True)
                stopped_early = True
                break
            chunk = [jobs[i] for i in order[s:s + args.batch_size]]
            t0 = time.time()
            # Step 1: Talkie writes the question (after the fixed prefix "<question>").
            qstop = lambda s: "</question>" in s or "<answer>" in s or truncate_at_stop(s)[1]
            q_outs, q_reasons = runner.generate([c[5] for c in chunk], 160, args.temperature,
                                                args.top_p, args.top_k, [c[6] for c in chunk],
                                                stop_ids=tk._stop_ids, stop_text_fn=qstop,
                                                decode_fn=tk.tokenizer.decode)
            # Step 2 (plan v2.9): code closes the question with the fixed, non-semantic delimiter
            # "</question>\n<answer>" and Talkie continues the same assistant turn with the answer.
            q_texts = []
            for toks in q_outs:
                qt = tk.tokenizer.decode(toks)
                qt = truncate_at_stop(qt)[0]
                qt = qt.split("</question>")[0].split("<answer>")[0].rstrip()
                q_texts.append(qt)
            a_prompts = [c[5] + tk.tokenizer.encode(qt + ANSWER_BRIDGE, allowed_special="all")
                         for c, qt in zip(chunk, q_texts)]
            outs, reasons = runner.generate(a_prompts, args.max_tokens, args.temperature,
                                            args.top_p, args.top_k, [c[6] + 1 for c in chunk],
                                            stop_ids=tk._stop_ids, stop_text_fn=stop_text,
                                            decode_fn=tk.tokenizer.decode, min_new=args.min_answer_tokens)
            secs = time.time() - t0
            for (cid, p, variant, form_name, prompt, ids, seed), qt, qwhy, toks, why in zip(
                    chunk, q_texts, q_reasons, outs, reasons):
                text = tk.tokenizer.decode(toks)
                text_trunc, leaked = truncate_at_stop(text)
                ans_text = (text_trunc if leaked else text)
                if "</answer>" not in ans_text and why in ("stop", "stop_text"):
                    ans_text = ans_text.rstrip() + "</answer>"  # answer ended by Talkie's own end token
                raw = v1.ASSISTANT_PREFIX + qt + ANSWER_BRIDGE + ans_text
                q, a, status = v1.parse_completion(raw)
                sink.write(json.dumps({
                    "candidate_id": cid, "passage_id": p["passage_id"], "source_id": p["source_id"],
                    "source_title": p["title"], "source_author": p["author"],
                    "source_year": p["publication_year"], "source_text_sha256": p["text_sha256"],
                    "form": form_name, "variant": variant, "seed": seed,
                    "model": "talkie-1930-13b-it", "model_revision": v1.MODEL_REVISION,
                    "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                    "max_tokens": args.max_tokens, "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "prompt_tokens": len(ids), "assistant_prefix": v1.ASSISTANT_PREFIX,
                    "decoder": "kv_cache_batched_v2 (scripts/v2/talkie_kv.py)",
                    "raw_completion": raw, "raw_untruncated": text, "generated_tokens": len(toks),
                    "talkie_question_text": qt, "question_finish_reason": qwhy,
                    "answer_bridge": ANSWER_BRIDGE, "protocol": "two-step same-turn v2.9", "min_answer_tokens": args.min_answer_tokens,
                    "finish_reason": why, "template_leak_truncated": bool(leaked),
                    "generation_status": "ok", "parse_status": status,
                    "question": q, "answer": a, "batch_seconds": round(secs, 2),
                }, ensure_ascii=False) + "\n")
            sink.flush()
            n_done += len(chunk)
            el = time.time() - t_start
            print(json.dumps({"stage": "batch", "done": n_done, "todo": len(jobs), "batch_seconds": round(secs, 1),
                              "rate_per_min": round(n_done / el * 60, 1),
                              "parsed_in_batch": sum(1 for o in outs if o),
                              "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}), flush=True)
    if not stopped_early:
        print(json.dumps({"stage": "generate_complete", "done": n_done}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate-kv", "generate", "speed"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cache-dir")
    ap.add_argument("--variants", default="0,1")
    ap.add_argument("--passage-ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--seed", type=int, default=1929)
    ap.add_argument("--temperature", type=float, default=0.55)
    ap.add_argument("--top-p", type=float, default=0.90)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--fp32", action="store_true", help="validate-kv in float32 (diagnostic)")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--min-answer-tokens", type=int, default=0)
    args = ap.parse_args()
    {"validate-kv": validate_kv, "generate": generate, "speed": speed}[args.mode](args)


if __name__ == "__main__":
    main()
