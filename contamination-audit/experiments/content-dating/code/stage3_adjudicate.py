"""Stage 3, part 2: gpt-oss adjudication of grep hit contexts (run INSIDE WSL on the 4090).

v1 (2026-07-24). For each book with hits, gpt-oss sees up to 6 excerpts (+-800 chars around each
marker/year hit) and answers: does any excerpt show text WRITTEN after 1929? The full marker list
fires on ~60% of clean books (homographs, OCR-garbled years) — the adjudicator carries the
discrimination. Output: intermediates\\stage3_verdicts.parquet.
"""
import os, json, re
import pandas as pd

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

if os.name == "nt":
    DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
else:
    DROPBOX = "/mnt/d/UZHechist Dropbox/Joachim Voth/claudecode/ranke-contamination-audit/experiments/content-dating"
I = os.path.join(DROPBOX, "intermediates")

PROMPT = ('This book is catalogued as written before 1929. An automatic scan flagged the excerpts below '
          'as containing possibly-modern vocabulary or year mentions — but the scan is noisy and most '
          'flags are innocent (OCR errors garbling old years, old words that merely resemble modern '
          'terms, library or digitization stamps). Read the excerpts and judge ONLY whether any of them '
          'shows text genuinely WRITTEN after 1929: name the exact phrase or date IN THE TEXT that '
          'proves it. If you cannot point to such a phrase, answer false.\nEXCERPTS:\n"""{ctx}"""\n'
          'Return JSON: {{"post1929": true/false, "which": "<=20 words quoting the proving phrase"}}')

def main():
    from vllm import LLM, SamplingParams
    d = pd.read_parquet(os.path.join(I, "stage3_hits.parquet"))
    d = d[d.any_hit == 1].reset_index(drop=True)
    print(f"{len(d)} books with hits to adjudicate", flush=True)
    llm = LLM(model="openai/gpt-oss-20b", dtype="auto", gpu_memory_utilization=0.90,
              max_model_len=8192, trust_remote_code=True)
    sp = SamplingParams(temperature=0, max_tokens=384)
    convs = [[{"role": "user", "content": PROMPT.format(ctx=r.contexts[:18000])}] for r in d.itertuples()]
    outs = llm.chat(convs, sp, chat_template_kwargs={"reasoning_effort": "low"}, use_tqdm=False)
    verdicts, whys = [], []
    for o in outs:
        t = o.outputs[0].text
        for sep in ("assistantfinal", "final"):
            if sep in t: t = t.rsplit(sep, 1)[1]; break
        try:
            j = json.loads(t[t.find("{"):t.rfind("}") + 1])
            verdicts.append(bool(j.get("post1929"))); whys.append(str(j.get("which", ""))[:150])
        except Exception:
            verdicts.append(None); whys.append(t.strip()[:150])
    d["adjudicated_post1929"] = verdicts
    d["why"] = whys
    d.to_parquet(os.path.join(I, "stage3_verdicts.parquet"))
    print(d.groupby("panel").adjudicated_post1929.agg(
        n="count", removed=lambda x: int((x == True).sum())).to_string(), flush=True)

if __name__ == "__main__":
    main()
