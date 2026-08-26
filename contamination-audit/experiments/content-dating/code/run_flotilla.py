"""FLOTILLA DRIVER (run on the 4090). For each local model, judge every extracted passage "written before
or after 1929?" via vLLM (batched). Writes one judgments parquet per model. The N-passages and aggregation
axes are derived later in analyze.py by subsetting these per-passage judgments.

Setup on the 4090:
    pip install vllm pandas pyarrow
    # data (passages_*.parquet) comes from extract_passages.py -> Dropbox/passages/
Run:
    python run_flotilla.py            # all models
    python run_flotilla.py --model Qwen/Qwen2.5-7B-Instruct   # one model
"""
import os, json, re, argparse, glob
import pandas as pd
DROPBOX=r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
PASS=os.path.join(DROPBOX,"passages"); OUT=os.path.join(DROPBOX,"outputs"); os.makedirs(OUT,exist_ok=True)

# the flotilla. quant tags fit the 4090 (24 GB). adjust ids to the exact quantized repos you pull.
MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct-AWQ",
    "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    # user picks — strong + fast production candidates:
    "Qwen/Qwen3.6-35B-A3B",            # MoE ~3B active; verify exact quantized repo id
    "openai/gpt-oss-20b",              # verify exact repo id / quant
]
PROMPT=('Read this passage from a scanned book and judge: was its text WRITTEN before 1929, or after 1929? '
        'Old, foreign-language, poetic, or historical writing is BEFORE. Modern prose, a post-1929 event/person/'
        'technology, or a modern perspective is AFTER. Ignore OCR noise. Passage:\n"""{p}"""\n'
        'Answer with only one word: BEFORE or AFTER.')

def parse(txt):
    t=txt.strip().upper()
    if "AFTER" in t[:20]: return 1
    if "BEFORE" in t[:20]: return 0
    return None

def run_model(model, passages):
    from vllm import LLM, SamplingParams
    q = "awq" if "AWQ" in model else ("gptq" if "GPTQ" in model.upper() else None)
    llm = LLM(model=model, quantization=q, dtype="auto", gpu_memory_utilization=0.90,
              max_model_len=4096, trust_remote_code=True)
    sp = SamplingParams(temperature=0, max_tokens=4)
    prompts=[PROMPT.format(p=p[:8000]) for p in passages["passage"].tolist()]
    outs=llm.generate(prompts, sp)
    passages=passages.copy(); passages["pred"]=[parse(o.outputs[0].text) for o in outs]
    safe=model.replace("/","__")
    passages[["id","label","length","strategy","idx","pred"]].to_parquet(os.path.join(OUT,f"judg_{safe}.parquet"))
    print(f"{model}: {len(passages)} passages judged -> judg_{safe}.parquet", flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--model"); a=ap.parse_args()
    passages=pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(PASS,"passages_*.parquet"))],ignore_index=True)
    print(f"{len(passages)} passages across {passages.id.nunique()} books")
    for m in ([a.model] if a.model else MODELS):
        try: run_model(m, passages)
        except Exception as e: print(f"SKIP {m}: {str(e)[:150]}", flush=True)
