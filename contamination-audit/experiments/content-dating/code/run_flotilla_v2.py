"""FLOTILLA DRIVER v2 (2026-07-22) — judge every passage with each local model via vLLM on the 4090.

Run INSIDE WSL with the flotilla venv:
    ~/venvs/flotilla/bin/python run_flotilla_v2.py --all          # loop all models, subprocess each
    ~/venvs/flotilla/bin/python run_flotilla_v2.py --model <id>   # one model, in-process

Files used:
  passages: /mnt/d/UZHechist Dropbox/Joachim Voth/claudecode/ranke-contamination-audit/experiments/content-dating/passages/passages_*.parquet
  outputs:  .../content-dating/outputs/judg_<model>.parquet  (+ timings_<model>.json)

Changes vs v1 (documented for the PI — PROVISIONAL until reviewed):
  - llm.chat() with each model's own chat template instead of raw llm.generate() on an instruct
    model (v1's raw-completion prompts would silently break gpt-oss's harmony format and degrade
    every instruct model).
  - per-model SUBPROCESS (--all mode): guarantees VRAM release between models; one model failing
    (OOM, bad quant) cannot kill the sweep.
  - Llama-3.2 via unsloth/ mirrors (meta-llama repos are license-gated 403 for this token;
    identical weights, ungated).
  - Qwen3.6-35B-A3B: the high-download AWQ quants are 25 GB and DO NOT fit 24 GB VRAM. Unmodified
    ~20-21 GB quants are tried in order until one loads and passes a smoke test.
  - reasoning models: gpt-oss-20b runs at reasoning_effort=low with max_tokens=256; Qwen3.6 runs
    with enable_thinking=False if its template supports it. Parse takes the LAST BEFORE/AFTER after
    stripping any think/analysis segment; plain models keep max_tokens=6 and first-word parse.
  - slice checkpointing (part files) + per-model timings json (passages/s feeds the 2M-book cost
    projection in the final report).
"""
import os, sys, json, re, glob, time, argparse, subprocess

# WSL environment (verified 2026-07-22 on the office 4090, driver 560.94 + vllm 0.25.1+cu129):
#  - pinned memory works on this WSL2 kernel (6.6.87) but vLLM disables it by default -> opt in
#  - FlashInfer JIT-compiles its sampler with nvcc, which needs sudo to install; we decode greedily,
#    so the torch-native sampler is equivalent -> disable
#  - FLASH_ATTN is precompiled (no nvcc); gpt-oss needs attention sinks -> TRITON_ATTN (triton
#    bundles its own ptxas), set per-model below
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

if os.name == "nt":
    DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
else:
    DROPBOX = "/mnt/d/UZHechist Dropbox/Joachim Voth/claudecode/ranke-contamination-audit/experiments/content-dating"
PASS = os.path.join(DROPBOX, "passages"); OUT = os.path.join(DROPBOX, "outputs")
SLICE = 32768

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct-AWQ",
    "unsloth/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "QWEN36_35B_A3B",              # placeholder -> first loadable candidate below
    "openai/gpt-oss-20b",
]
QWEN36_CANDIDATES = [
    "shieldstar/Qwen3.6-35B-A3B-int4-AutoRound-EC",      # 20.9 GB, unmodified AutoRound (GPTQ-format marlin kernels, pre-downloaded)
    "nvidia/Qwen3.6-35B-A3B-NVFP4",                      # official NVIDIA quant (needs marlin fp4 emulation on Ada)
    "LeaderboardModel1/Qwen3.6-35B-A3B-autoround-W4A16", # 19.6 GB
    "88plug/Qwen3.6-35B-A3B-W4A16",                      # 21.1 GB
]
REASONING = {"openai/gpt-oss-20b"}                       # + any Qwen3.6 candidate (handled below)

PROMPT = ('Read this passage from a scanned book and judge: was its text WRITTEN before 1929, or after 1929? '
          'Old, foreign-language, poetic, or historical writing is BEFORE. Modern prose, a post-1929 event/person/'
          'technology, or a modern perspective is AFTER. Ignore OCR noise. Passage:\n"""{p}"""\n'
          'Answer with only one word: BEFORE or AFTER.')

def is_reasoning(model):
    return model in REASONING or "Qwen3.6" in model

def parse(txt, reasoning):
    t = txt.strip()
    if reasoning:
        for sep in ("</think>", "assistantfinal", "final"):
            if sep in t:
                t = t.rsplit(sep, 1)[1]; break
        hits = re.findall(r"\b(BEFORE|AFTER)\b", t.upper())
        return {"BEFORE": 0, "AFTER": 1}[hits[-1]] if hits else None
    u = t.upper()
    if "AFTER" in u[:20]: return 1
    if "BEFORE" in u[:20]: return 0
    return None

def run_model(model, smoke_only=False, passages_file=None, suffix=""):
    import pandas as pd
    from vllm import LLM, SamplingParams
    reasoning = is_reasoning(model)
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN" if "gpt-oss" in model else "FLASH_ATTN"
    q = "awq" if "AWQ" in model.upper() else None       # gptq/autoround/nvfp4/mxfp4 are auto-detected
    # Qwen3.6-35B 4-bit weights are ~21 GB on a 24 GB card shared with the Windows display
    # (~1.5 GB held by the host): util 0.93 is the ceiling, eager mode frees CUDA-graph memory
    # for KV, and prompts are <=~900 tokens so 2048 context suffices
    kw = dict(model=model, dtype="auto",
              gpu_memory_utilization=0.93 if "Qwen3.6" in model else 0.90,
              max_model_len=2048 if "Qwen3.6" in model else 4096, trust_remote_code=True)
    if "Qwen3.6" in model:
        # AutoRound-EC wedged on the full run in eager+9k-KV (passed smoke, then a silent
        # kernel/scheduling hang at 130 W). Retry profile: cap concurrent seqs; eager only for
        # the ~21 GB quants (the 19.6 GB W4A16 leaves room for CUDA graphs).
        kw["max_num_seqs"] = int(os.environ.get("FLOTILLA_MAX_SEQS", "16"))
        if "W4A16" not in model: kw["enforce_eager"] = True
    if q: kw["quantization"] = q
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0, max_tokens=256 if reasoning else 6)
    ctk = {}
    if "Qwen3.6" in model: ctk = {"enable_thinking": False}
    if model == "openai/gpt-oss-20b": ctk = {"reasoning_effort": "low"}

    def judge(passages_text):
        convs = [[{"role": "user", "content": PROMPT.format(p=p[:8000])}] for p in passages_text]
        try:
            outs = llm.chat(convs, sp, chat_template_kwargs=ctk, use_tqdm=False) if ctk else \
                   llm.chat(convs, sp, use_tqdm=False)
        except TypeError:
            outs = llm.chat(convs, sp, use_tqdm=False)
        return [parse(o.outputs[0].text, reasoning) for o in outs]

    smoke = judge(["The astronaut checked her email after the television broadcast of the 1969 moon landing.",
                   "His Majesty's ship anchored in the roads of Portsmouth, whence the convoy of 1807 was to sail."])
    print(f"SMOKE {model}: {smoke} (want [1, 0])", flush=True)
    if smoke[0] is None and smoke[1] is None:
        raise RuntimeError(f"smoke test unparseable for {model}")
    if smoke_only:
        return

    files = [os.path.join(PASS, passages_file)] if passages_file else \
            sorted(glob.glob(os.path.join(PASS, "passages_*.parquet")))
    passages = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"{len(passages)} passages across {passages.id.nunique()} books", flush=True)
    safe = model.replace("/", "__") + suffix
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); n_new = 0
    parts = []
    for s in range(0, len(passages), SLICE):
        pf = os.path.join(OUT, f"judg_{safe}.part{s // SLICE:03d}.parquet")
        parts.append(pf)
        if os.path.exists(pf): continue
        sl = passages.iloc[s:s + SLICE].copy()
        sl["pred"] = judge(sl["passage"].tolist())
        sl[["id", "label", "length", "strategy", "idx", "pred"]].to_parquet(pf)
        n_new += len(sl)
        el = time.time() - t0
        print(f"  slice {s // SLICE + 1}/{(len(passages) - 1) // SLICE + 1} done "
              f"({n_new / max(el, 1):.0f} passages/s)", flush=True)
    full = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    full.to_parquet(os.path.join(OUT, f"judg_{safe}.parquet"))
    for p in parts: os.remove(p)
    wall = time.time() - t0
    with open(os.path.join(OUT, f"timings_{safe}.json"), "w") as f:
        json.dump({"model": model, "n_passages": len(full), "n_new_this_run": n_new,
                   "wall_seconds_this_run": round(wall, 1),
                   "passages_per_second": round(n_new / max(wall, 1), 2),
                   "null_preds": int(full.pred.isna().sum())}, f, indent=1)
    print(f"{model}: {len(full)} judged, {int(full.pred.isna().sum())} unparseable -> judg_{safe}.parquet", flush=True)

def resolve_qwen36():
    """Return the first Qwen3.6-35B candidate that initializes + smokes on this GPU (cached)."""
    cache = os.path.join(OUT, "_qwen36_resolved.txt")
    if os.path.exists(cache):
        return open(cache).read().strip()
    for cand in QWEN36_CANDIDATES:
        print(f"trying Qwen3.6 candidate: {cand}", flush=True)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--model", cand, "--smoke-only"])
        if r.returncode == 0:
            os.makedirs(OUT, exist_ok=True)
            open(cache, "w").write(cand)
            return cand
        print(f"  candidate failed (exit {r.returncode}), trying next", flush=True)
    print("NO Qwen3.6-35B-A3B candidate fits/loads — skipping that model", flush=True)
    return None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--passages-file", help="single passages parquet (delta run), e.g. passages_contamC.parquet")
    ap.add_argument("--suffix", default="", help="output-name suffix for delta runs, e.g. .contamC")
    a = ap.parse_args()
    if a.model:
        run_model(a.model, smoke_only=a.smoke_only, passages_file=a.passages_file, suffix=a.suffix)
    elif a.all:
        for m in MODELS:
            if m == "QWEN36_35B_A3B":
                m = resolve_qwen36()
                if m is None: continue
            safe = m.replace("/", "__")
            if os.path.exists(os.path.join(OUT, f"judg_{safe}.parquet")):
                print(f"SKIP {m} (already done)", flush=True); continue
            print(f"\n=== {m} ===", flush=True)
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--model", m])
            if r.returncode != 0:
                print(f"MODEL FAILED {m} (exit {r.returncode}) — continuing with next", flush=True)
    else:
        ap.print_help()
