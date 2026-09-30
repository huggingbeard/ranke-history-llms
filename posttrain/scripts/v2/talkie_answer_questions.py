#!/usr/bin/env python3
"""Night run 2026-09-29: Talkie-1930-13B-IT answers the QUESTIONS printed in period question-and-answer books
directly (no passage). Reason: in the 5-way blind grading (analysis/results_judging5.json) Talkie's own direct
answers scored 63.2 on history substance vs 13-18 for Ranke trained on Talkie's passage-based records, so
Talkie's direct answers may be much better training text. Period model writes; no modern model touches the text.
Same decoder and decoding as eval_talkie.py (the ceiling check): Talkie's chat template, greedy, stop at its end
token, exact-checked KV-cache decoder. Resumable: questions already in --out are skipped. One JSONL row per
question, written after every batch."""
import argparse, hashlib, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import generate_talkie_pairs as v1  # noqa: E402
from talkie_kv import KVRunner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)      # JSONL with candidate_id, question
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-minutes", type=float, default=1e9)   # stop cleanly before the Slurm limit
    a = ap.parse_args()
    from talkie.chat import format_prompt, truncate_at_stop
    t_start = time.time()
    tk, receipt = v1.load_pinned_talkie("talkie-1930-13b-it", v1.MODEL_REVISION, "cuda", a.cache_dir)
    runner = KVRunner(tk.model)
    items = [json.loads(l) for l in open(a.questions, encoding="utf-8") if l.strip()]
    if a.limit:
        items = items[:a.limit]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["candidate_id"] for l in open(out, encoding="utf-8") if l.strip()}
    todo = [it for it in items if it["candidate_id"] not in done]
    # similar prompt lengths in one batch -> less padding
    todo.sort(key=lambda it: len(it["question"]))
    print(json.dumps({"stage": "start", "total": len(items), "done_before": len(done), "todo": len(todo)}), flush=True)
    with out.open("a", encoding="utf-8") as sink:
        for s in range(0, len(todo), a.batch_size):
            if (time.time() - t_start) / 60 > a.max_minutes:
                print(json.dumps({"stage": "time_limit_stop", "written": s}), flush=True)
                return
            chunk = todo[s:s + a.batch_size]
            prompts = [format_prompt(it["question"].strip()) for it in chunk]
            ids = [tk.tokenizer.encode(p, allowed_special="all") for p in prompts]
            t0 = time.time()
            toks, why = runner.generate(ids, a.max_new_tokens, 1.0, None, None, [0] * len(chunk),
                                        stop_ids=tk._stop_ids, greedy=True)
            secs = time.time() - t0
            for it, p, tt, w in zip(chunk, prompts, toks, why):
                text = truncate_at_stop(tk.tokenizer.decode(tt))[0].strip()
                sink.write(json.dumps({"candidate_id": it["candidate_id"], "question": it["question"],
                                       "book_answer": it.get("answer"), "source_id": it.get("source_id"),
                                       "title": it.get("title"), "year": it.get("year"), "category": it.get("category"),
                                       "rendered_prompt": p, "answer": text, "new_tokens": len(tt),
                                       "termination": "eos" if w == "stop" else "length",
                                       "max_new_tokens": a.max_new_tokens, "batch_seconds": round(secs, 2),
                                       "model": "talkie-lm/talkie-1930-13b-it", "model_revision": v1.MODEL_REVISION,
                                       "decoding": "greedy, kv_cache_batched_v2"}, ensure_ascii=False) + "\n")
            sink.flush()
            print(json.dumps({"stage": "batch", "done": s + len(chunk), "of": len(todo), "seconds": round(secs, 1)}),
                  flush=True)
    print(json.dumps({"stage": "done", "rows_file": sum(1 for _ in open(out, encoding="utf-8")),
                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "talkie": receipt}), flush=True)


if __name__ == "__main__":
    main()
