"""Reprint policy pass (PI decision 2026-07-23): "reprint only after separate oss 20b pass —
any later-date intro note or editorial text? if so, out."

Runs gpt-oss-20b (vLLM, WSL, 4090 — eugenics track confirmed finished) over the 80
contam_reprint_old_body books. Input per book: front matter (first 12,000 chars) + back matter
(last 3,000) — where editorial apparatus lives. Verdict: modern_editorial -> REMOVE, else KEEP.
Output: repo reprint_verdicts.csv (id, modern_editorial, evidence, keep_remove).
Run INSIDE WSL: ~/venvs/flotilla/bin/python reprint_editorial_pass.py
"""
import os, json, re
import pandas as pd
import pyarrow.parquet as pq

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

if os.name == "nt":
    DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit"
else:
    DROPBOX = "/mnt/d/UZHechist Dropbox/Joachim Voth/claudecode/ranke-contamination-audit"
CORPUS = os.path.join(DROPBOX, "temp-guohui", "decontamination", "datasets", "history-llms", "sample-10gb", "data")
REPO = os.path.dirname(os.path.abspath(__file__))

PROMPT = ('This book is a modern reprint of an older work (catalogued pre-1929 content, printed after 1929). '
          'POLICY QUESTION: does this volume contain ANY editorial material written after 1929 — a later-date '
          'introduction, foreword, preface, editor\'s or translator\'s note, afterword, endnotes, or commentary '
          'attributable to the modern edition? A bare reset of the old text (title page, old preface, the work '
          'itself) does NOT count. Below: the front matter, then the back matter.\n'
          'FRONT:\n"""{front}"""\nBACK:\n"""{back}"""\n'
          'Return JSON: {{"modern_editorial": true/false, "evidence": "<=20 words quoting or naming the material"}}')

def main():
    from vllm import LLM, SamplingParams
    mc = pd.read_csv(os.path.join(REPO, "manifest_contamC.csv"), dtype={"id": str})
    reps = mc[mc.cls == "contam_reprint_old_body"]
    import pyarrow as pa, pyarrow.compute as pc
    texts = {}
    for sh, g in reps.groupby("shard"):
        t = pq.read_table(os.path.join(CORPUS, sh), columns=["id", "text"])
        t = t.filter(pc.is_in(t.column("id"), value_set=pa.array(list(g.id))))
        for i, tx in zip(t.column("id").to_pylist(), t.column("text").to_pylist()):
            texts[i] = tx or ""
    print(f"{len(texts)} reprint texts loaded", flush=True)
    llm = LLM(model="openai/gpt-oss-20b", dtype="auto", gpu_memory_utilization=0.90,
              max_model_len=8192, trust_remote_code=True)
    sp = SamplingParams(temperature=0, max_tokens=384)
    ids = list(texts)
    convs = [[{"role": "user", "content": PROMPT.format(
        front=" ".join(texts[i][:12000].split()), back=" ".join(texts[i][-3000:].split()))}] for i in ids]
    outs = llm.chat(convs, sp, chat_template_kwargs={"reasoning_effort": "low"}, use_tqdm=False)
    rows = []
    for i, o in zip(ids, outs):
        t = o.outputs[0].text
        for sep in ("assistantfinal", "final"):
            if sep in t: t = t.rsplit(sep, 1)[1]; break
        try:
            j = json.loads(t[t.find("{"):t.rfind("}") + 1])
            me, ev = bool(j.get("modern_editorial")), str(j.get("evidence", ""))[:120]
        except Exception:
            me, ev = None, t.strip()[:120]
        rows.append({"id": i, "modern_editorial": me, "evidence": ev,
                     "policy": "REMOVE" if me else ("KEEP" if me is False else "UNPARSED")})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(REPO, "reprint_verdicts.csv"), index=False)
    print(d.policy.value_counts().to_string(), flush=True)
    for _, r in d[d.policy == "REMOVE"].head(6).iterrows():
        print(f"  REMOVE {r.id[:8]}: {r.evidence}", flush=True)

if __name__ == "__main__":
    main()
