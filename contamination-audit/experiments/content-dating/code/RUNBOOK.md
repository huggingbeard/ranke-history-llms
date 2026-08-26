# Runbook — 4090 (office PC)

Everything here runs on the office box with the 4090. Code comes from this repo (git clone); data lives in
**Dropbox** at `.../ranke-contamination-audit/experiments/content-dating/` (syncs automatically).

## 0. Setup
```
git pull
pip install vllm datasets pandas pyarrow pyahocorasick httpx
```

## 1. Positives — 1,000 definitely-post-1929 modern-original books
```
python pull_positives.py
```
Streams `institutional_books` from HF (token in claudecode/api/HF.txt), keeps year>=1947, drops reprints,
writes text -> Dropbox/positives_text, manifest_positives.csv -> repo. **Verify the HF dataset id + field
names printed on the first record** (schema may differ) and adjust the top of the script.

## 2. Certify the negatives are really clean (two well-calibrated models)
Use **deepseek-v4-flash (OpenRouter) + Llama-3.3-70B**. Calibration confirmed in pilots: Llama-70B and
deepseek-flash do NOT over-flag old/foreign/poetic content (gemini-flash-lite did — don't use it).
- On the 4090, run Llama-70B **locally** (no API limits). deepseek-flash stays on OpenRouter (no daily cap).
- Certified clean = NEITHER model flags any body chunk. Single-model flags -> review queue.
- (The paid Groq key would also work once its billing restriction is cleared, but local Llama makes it moot.)

## 3. Extract passages
```
python extract_passages.py
```
Random + grep-directed passages at lengths {250,500,1000,2000}, 16 each, -> Dropbox/passages/.

## 4. Flotilla — judge every passage with each model
```
python run_flotilla.py
```
Models: Qwen2.5 {0.5,1.5,3,7,14B} + Llama-3.2 {1,3B} + Phi-3.5-mini + Qwen3.6-35B-A3B + gpt-oss-20b.
14B/20B/35B use AWQ/GPTQ quant to fit 24 GB. Writes one judgments parquet per model.
**Verify the exact quantized repo ids for Qwen3.6-35B-A3B and gpt-oss-20b before running.**

## 5. Analyze
```
python analyze.py
```
Produces `results.csv` and prints the headline: cheapest config (model x N-passages x snippet length x
sampling) that hits **recall >= 0.95**. Recall is the target; precision secondary.

## Notes
- N-passages axis {1,2,4,8,16} is derived by subsetting — no re-run.
- Snippet length is a real re-run axis (model must see the actual length); handled in extract_passages.
- Datasets never go to git (see .gitignore). Only code + manifests (ids/labels) are committed.
