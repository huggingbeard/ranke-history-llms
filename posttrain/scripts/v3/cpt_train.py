#!/usr/bin/env python3
"""Plan v3 option 3: continued pretraining (CPT) of Ranke-4B-1929 v2 on period (<=1913) reference text.

Full fine-tune (all weights), bf16, gradient checkpointing, 8-bit paged AdamW (bitsandbytes), cosine LR with warmup,
packed 2,048-token sequences with EOS between chunks. Built for short cluster jobs:
  * --max-minutes: stop cleanly before the Slurm limit and save a resumable checkpoint
    (model + optimizer + scheduler + data position + RNG) to OUT/ckpt_last (atomic rename).
  * restart with the same command -> resumes from OUT/ckpt_last.
  * progress.jsonl: one row per logged step (loss, lr, tokens seen, tokens/s, GPU memory).
  * --heldout: mean loss / perplexity on held-out period chunks at start, at every checkpoint, and at the end.
When all data is consumed, saves OUT/final (HF format, safetensors) and writes OUT/DONE.
"""
import argparse, json, math, os, random, time
from pathlib import Path

BASE_REV = "fb4cede16ed59e1249ebeba2b4e5b9ec2a560642"


def log(path, **kw):
    kw["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)          # cpt_train.jsonl (field "text")
    ap.add_argument("--heldout", required=True)       # cpt_heldout.jsonl
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)   # 4 x 8 x 2048 = 65,536 tokens per update
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max-minutes", type=float, default=100)
    ap.add_argument("--max-updates", type=int, default=0, help="pilot: stop after N updates")
    ap.add_argument("--heldout-max", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1929)
    a = ap.parse_args()
    t_start = time.time()
    import torch
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    assert BASE_REV in Path(a.base).parts
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prog = out / "progress.jsonl"
    torch.manual_seed(a.seed); random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True)

    # ---- data: tokenize once, cache token stream (resume-safe) ----
    cache = out / "tokens_train.pt"
    if cache.exists():
        stream = torch.load(cache)
    else:
        rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
        random.Random(a.seed).shuffle(rows)
        ids = []
        for r in rows:
            ids.extend(tok.encode(r["text"], add_special_tokens=False)); ids.append(tok.eos_token_id)
        stream = torch.tensor(ids, dtype=torch.int32)
        torch.save(stream, cache)
    n_seq = len(stream) // a.seq
    seqs = stream[: n_seq * a.seq].view(n_seq, a.seq)
    tokens_per_update = a.micro * a.accum * a.seq
    total_updates = n_seq // (a.micro * a.accum)
    held = [json.loads(l)["text"] for l in open(a.heldout, encoding="utf-8")][: a.heldout_max]

    ck = out / "ckpt_last"
    src = ck / "model" if ck.exists() else a.base
    model = AutoModelForCausalLM.from_pretrained(src, local_files_only=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = bnb.optim.PagedAdamW8bit(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = get_cosine_schedule_with_warmup(opt, a.warmup, total_updates)
    state = {"update": 0, "tokens": 0}
    if ck.exists():
        opt.load_state_dict(torch.load(ck / "optim.pt", weights_only=False))
        sched.load_state_dict(torch.load(ck / "sched.pt", weights_only=False))
        state = json.loads((ck / "state.json").read_text())
        torch.set_rng_state(torch.load(ck / "rng.pt", weights_only=False))
    log(prog, stage="start", resumed=ck.exists(), update=state["update"], total_updates=total_updates,
        train_tokens=int(n_seq * a.seq), tokens_per_update=tokens_per_update, lr=a.lr, gpu=torch.cuda.get_device_name())

    @torch.no_grad()
    def heldout_loss():
        model.eval(); tot = n = 0
        for t in held:
            x = torch.tensor([tok.encode(t, add_special_tokens=False)[: a.seq]], device="cuda")
            if x.shape[1] < 16:
                continue
            l = model(input_ids=x, labels=x).loss.float().item()
            tot += l * (x.shape[1] - 1); n += x.shape[1] - 1
        model.train()
        return tot / n

    def save_ckpt(tag):
        tmp = out / f"ckpt_tmp"
        if tmp.exists():
            import shutil; shutil.rmtree(tmp)
        (tmp / "model").mkdir(parents=True)
        model.save_pretrained(tmp / "model", safe_serialization=True)
        tok.save_pretrained(tmp / "model")
        torch.save(opt.state_dict(), tmp / "optim.pt"); torch.save(sched.state_dict(), tmp / "sched.pt")
        torch.save(torch.get_rng_state(), tmp / "rng.pt")
        (tmp / "state.json").write_text(json.dumps(state))
        if ck.exists():
            import shutil; shutil.rmtree(ck)
        tmp.rename(ck)
        log(prog, stage="checkpoint", tag=tag, **state)

    if state["update"] == 0:
        log(prog, stage="heldout", update=0, loss=heldout_loss())
    model.train()
    t0 = time.time(); tok_t0 = state["tokens"]
    while state["update"] < total_updates:
        u = state["update"]
        for m in range(a.accum):
            i = (u * a.accum + m) * a.micro
            x = seqs[i:i + a.micro].long().cuda()
            loss = model(input_ids=x, labels=x).loss / a.accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        state["update"] += 1; state["tokens"] += tokens_per_update
        if state["update"] % 10 == 0 or state["update"] == 1:
            el = time.time() - t0
            log(prog, stage="update", update=state["update"], loss=round(loss.item() * a.accum, 4),
                lr=sched.get_last_lr()[0], tokens=state["tokens"],
                tok_per_s=round((state["tokens"] - tok_t0) / max(el, 1e-6)),
                gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        if (a.max_updates and state["update"] - (tok_t0 // tokens_per_update) >= a.max_updates) or \
                (time.time() - t_start) / 60 > a.max_minutes:
            save_ckpt("time_or_pilot_limit")
            log(prog, stage="heldout", update=state["update"], loss=heldout_loss())
            return
    save_ckpt("end")
    log(prog, stage="heldout", update=state["update"], loss=heldout_loss())
    model.save_pretrained(out / "final", safe_serialization=True); tok.save_pretrained(out / "final")
    (out / "DONE").write_text(json.dumps(state))
    log(prog, stage="done", **state)


if __name__ == "__main__":
    main()
