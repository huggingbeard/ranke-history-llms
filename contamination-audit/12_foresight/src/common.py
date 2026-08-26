# -*- coding: utf-8 -*-
"""Shared primitives for the foresight sub-study. Factored from the repo's canonical scoring idiom
(surprise_balloon.py / score_v2.py / lm_battery.py) so every foresight script scores identically.

slp(model, tok, prompt, cont) : sum_k log P(cont_k | prompt, cont_<k), no BOS  (the ONE primitive).
Surprise S = -slp.  Everything (ΔS, PMI strain, calibrated forced choice, entropy) is built on slp.
"""
import os, sys, gc, math, statistics as st

try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
_HF = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt"
if os.path.exists(_HF):
    os.environ.setdefault("HF_TOKEN", open(_HF).read().strip())

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
# Repo layout: code lives in src/, everything else in sibling folders off the repo root. ROOT resolves
# whether a script is run from src/ (after the reorg) or from the flat root (backward-compatible).
ROOT = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "src" else HERE


def _sub(name):
    p = os.path.join(ROOT, name)
    os.makedirs(p, exist_ok=True)
    return p


RESULTS = _sub("results")    # all .json / .csv data outputs
FIGURES = _sub("figures")    # all .png figures
REPORTS = _sub("reports")    # all rendered .html reports + the walkthrough source
PAPER = os.path.join(ROOT, "paper")        # LaTeX source + compiled PDF + numbers.tex
PASSAGES = os.path.join(ROOT, "passages")  # public-domain source passages (Arm B)
ZWEIG = os.path.join(ROOT, "zweig")        # Zweig 'World of Yesterday' texts
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32


def wait_for_vram(min_gb=9.0, poll_s=60, max_wait_s=4 * 3600):
    """Block until the GPU has at least min_gb free (other jobs may hold the card). Warns every 5 polls."""
    import subprocess, time
    if DEV != "cuda":
        return
    waited = 0
    n = 0
    while True:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30).stdout
            free_gb = float(out.strip().splitlines()[0]) / 1024.0
        except Exception:
            return  # can't query; let the load attempt (and its OOM retry) handle it
        if free_gb >= min_gb:
            if n:
                print(f"[vram] {free_gb:.1f} GB free — proceeding", flush=True)
            return
        n += 1
        if n % 5 == 1:
            print(f"[vram] only {free_gb:.1f} GB free (< {min_gb}); waiting for other GPU jobs...", flush=True)
        time.sleep(poll_s)
        waited += poll_s
        if waited >= max_wait_s:
            print(f"[vram] waited {waited//60} min; attempting load anyway", flush=True)
            return


def load(mid, retries=6, retry_s=300, dtype=None):
    """Load tokenizer + model (streamed). Waits for free VRAM; retries on CUDA OOM
    (the user's other GPU jobs may transiently hold the card).

    DTYPE — STANDARD DEFAULT = float32 SINCE 2026-07-11. Diagnosis (src/test_slp_batch2.py): bf16
    scoring carries ~0.04-nat kernel-accumulation noise that VARIES WITH BATCH SHAPE (batch-32 vs
    sequential differed by up to 0.23 nats / 0.038 probability), whereas fp32 is batch-invariant to
    3e-5 and no slower here (these are short-sequence, bandwidth-bound forwards; fp32 4B = ~16 GB,
    fits the RTX 4090). fp32 + slp_pairs batching = exact, replicable, and x4.5 faster than the old
    bf16-sequential path. Pass dtype=common.DT for the old bf16 footprint (~9 GB) on small cards;
    results saved BEFORE 2026-07-11 were bf16-sequential — expect kernel-noise-sized differences on
    rerun, with no verdict flips measured.

    IMPORTANT: the caller must drop its own reference before loading the next model:
        del model; common.free()
    (free(model) alone only deletes the callee's reference — the old model stays on the GPU and,
    historically, was silently paged out by WDDM; with wait_for_vram it would deadlock instead.)"""
    import time
    dt = torch.float32 if dtype is None else dtype
    min_gb = 17.0 if (DEV == "cuda" and dt == torch.float32) else 9.0
    tok = AutoTokenizer.from_pretrained(mid)
    for attempt in range(retries):
        gc.collect()                       # release anything the caller has already dropped
        if DEV == "cuda":
            torch.cuda.empty_cache()
        wait_for_vram(min_gb=min_gb)
        try:
            model = AutoModelForCausalLM.from_pretrained(mid, dtype=dt, low_cpu_mem_usage=True).to(DEV).eval()
            return tok, model
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[vram] OOM loading {mid} (attempt {attempt+1}/{retries}); sleeping {retry_s}s", flush=True)
            time.sleep(retry_s)
    raise RuntimeError(f"could not load {mid} after {retries} OOM retries")


def free(model=None):
    """Release GPU memory. NOTE: passing `model` cannot drop the CALLER's reference — the caller
    must `del model` first (or rely on load()'s pre-load gc). Kept accepting an argument for
    backward compatibility with older scripts."""
    if model is not None:
        del model
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()


def toklen(tok, s):
    return len(tok(s, add_special_tokens=False).input_ids)


def slp(model, tok, prompt, cont):
    """sum log P(cont | prompt), no special tokens. Matches every scorer in the repo."""
    pid = tok(prompt, add_special_tokens=False).input_ids
    cid = tok(cont, add_special_tokens=False).input_ids
    full = pid + cid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(cid)))


def slp_pairs(model, tok, pairs, batch_size=32):
    """BATCHED slp over (prompt, cont) pairs -> list of sums, in order. THE STANDARD DEFAULT for
    scoring loops (2026-07-11): identical tokenization and indexing to slp(), one forward per chunk
    instead of one per pair. Rows are right-padded with an attention mask; under causal attention the
    padding sits strictly after every scored position, so it cannot influence the scored logits.
    Numerical equivalence to the sequential loop is verified by src/test_slp_batch.py (run it again
    after any change here or any torch upgrade). Use slp() only for one-off calls."""
    out = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    for i0 in range(0, len(pairs), batch_size):
        chunk = pairs[i0:i0 + batch_size]
        pids = [tok(p, add_special_tokens=False).input_ids for p, _ in chunk]
        cids = [tok(c, add_special_tokens=False).input_ids for _, c in chunk]
        fulls = [p + c for p, c in zip(pids, cids)]
        L = max(len(f) for f in fulls)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, f in enumerate(fulls):
            ids[r, :len(f)] = torch.tensor(f)
            mask[r, :len(f)] = 1
        with torch.no_grad():
            lp = torch.log_softmax(
                model(ids.to(model.device), attention_mask=mask.to(model.device)).logits.float(), dim=-1)
        for r, (p_, c_) in enumerate(zip(pids, cids)):
            out.append(sum(lp[r, len(p_) + k - 1, c_[k]].item() for k in range(len(c_))))
        del lp
    return out


def slp_batch(model, tok, prompt, conts, batch_size=32):
    """Batched slp: ONE prompt, MANY continuations (the battery pattern: 4 options per stem, or the
    date-spike's 17 candidate years). Returns [slp(prompt, c) for c in conts]."""
    return slp_pairs(model, tok, [(prompt, c) for c in conts], batch_size)


def cal_score(model, tok, subject, neutral, cont):
    """Calibrated (popularity-removed) score of a continuation: c = slp(subject,cont) - slp(neutral,cont).
    c = -strain(cont). Positive => the loaded pre-shock subject RAISES this outcome above its neutral base rate."""
    return slp(model, tok, subject, cont) - slp(model, tok, neutral, cont)


def strain(model, tok, subject, neutral, cont):
    """PMI strain = S_cond - S_marg = -c. Positive => the outcome STRAINS the worldview under this framing."""
    s_cond = -slp(model, tok, subject, cont)
    s_marg = -slp(model, tok, neutral, cont)
    return s_cond, s_marg, s_cond - s_marg


def set_entropy(model, tok, subject, neutral, options):
    """Distribution over a small outcome set via softmax of calibrated scores; returns (probs, entropy_nats).
    probs aligned to `options`; options[0] is the ex-post-true outcome by convention."""
    c = [cal_score(model, tok, subject, neutral, o) for o in options]
    m = max(c)
    ex = [math.exp(x - m) for x in c]
    Z = sum(ex)
    p = [e / Z for e in ex]
    H = -sum(pi * math.log(pi + 1e-12) for pi in p)
    return p, H


def generate(model, tok, prompt, n=40, sample=False, temp=0.7):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        if sample:
            out = model.generate(**ids, max_new_tokens=n, do_sample=True, temperature=temp, top_p=0.95,
                                 repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
        else:
            out = model.generate(**ids, max_new_tokens=n, do_sample=False, repetition_penalty=1.3,
                                 pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ---- small stats helpers (mirror score_v2 / traffic_light) ----
def binom_ge(k, n, p=0.25):
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def ci95(k, n):  # Wilson
    if n == 0:
        return (0.0, 0.0)
    z = 1.96; p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def two_z(z):
    return math.erfc(abs(z) / math.sqrt(2))
