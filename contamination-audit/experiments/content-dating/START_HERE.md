# START HERE — instructions for the office-PC Claude (the 4090 box)

You are picking up a prepared experiment. You have **no prior context**; everything you need is here.
Read this file top to bottom, then run it. Ask the user before anything destructive or costly beyond the run.

## What this is

Project **Ranke**: training language models with a hard knowledge cutoff (e.g. 1929). The training corpus
must contain no post-cutoff content. Our production dating pipeline reads only a book's **front matter**
(~3% of it), so it is blind to post-1929 content buried in the body.

**This experiment answers:** how reliably, and how cheaply, can a model detect post-1929 *content* by
reading sampled body passages? We measure precision/recall as a function of **model size × passages-per-book
× snippet length × sampling strategy**, to find the cheapest config that scales to ~2M books.

**Recall is the target. Precision is secondary** (over-removing a clean book is cheap; keeping a
contaminated one is not).

## Where everything is

- **Code + manifests:** this repo, `experiments/content-dating/` (you are in it). `git pull` for the latest.
- **Data (large):** Dropbox `…\claudecode\ranke-contamination-audit\experiments\content-dating\`
  → `code\` (mirror of these scripts), `intermediates\`, `data\ passages\ outputs\ positives_text\`.
- **Corpus text for the clean negatives:** Dropbox
  `…\ranke-contamination-audit\temp-guohui\decontamination\datasets\history-llms\sample-10gb\` (32 parquet shards).
- **API tokens:** Dropbox `…\claudecode\api\` → `HF.txt`, `openrouter.txt`, `gemini.txt`, `groq api key.txt`.
- **Anachronism markers** (for grep-directed sampling): `markers_1945.json`, already copied into `code\`.

## Decisions already made — do NOT relitigate

- **Certifier pair = deepseek-v4-flash (OpenRouter) + Llama-3.3-70B.** Both are well-calibrated for this task.
  On this box, **run Llama-3.3-70B locally** (vLLM) so there are no API limits; deepseek-flash stays on OpenRouter.
- **Do NOT use gemini-flash-lite as a judge** — it over-flags old content (it called Hebrew, Latin, poetry,
  and an 1845 legal charge "post-1929"). Verified failure.
- **Groq:** the paid keys are billing-restricted (overdue payment); the free key is 1,000 req/day + 12k tok/min —
  too small. Local Llama on the 4090 replaces it. Ignore Groq unless the user says the billing is fixed.
- **Flotilla (model-size axis):** Qwen2.5 {0.5,1.5,3,7,14B} + Llama-3.2 {1,3B} + Phi-3.5-mini +
  **Qwen3.6-35B-A3B** + **gpt-oss-20b** (the last two are the strong+fast production candidates). Big ones
  in AWQ/GPTQ quant to fit 24 GB.
- **Negatives = 999 clean institutional_books** (`manifest_negatives.csv`); `cac2c2a9` already removed.

## The task — run in this order (details in RUNBOOK.md)

1. `pull_positives.py` — stream `institutional_books` from HF, keep 1,000 post-1946 modern-original books.
   **VERIFY** the printed dataset id + field names on the first record; adjust the top of the script if the
   schema differs.
2. **Certify the negatives** (`certify_negatives.py`, switch Llama to local) — deepseek-flash + local Llama-70B
   read most of each book's body; keep only books NEITHER flags. Single-model flags → review queue.
3. `extract_passages.py` — random + grep-directed passages at lengths {250,500,1000,2000}, 16 each → Dropbox.
4. `run_flotilla.py` — judge every passage with each model via vLLM. **VERIFY** the exact quantized repo ids
   for Qwen3.6-35B-A3B and gpt-oss-20b before running. One judgments parquet per model.
5. `analyze.py` — precision/recall/F1 surfaces; prints the **cheapest config with recall ≥ 0.95**.

## What to deliver back

- `outputs/results.csv` (the full surface) committed/synced.
- A short written finding: the recall/precision curve vs model size, how few/short passages a small local
  model needs, whether grep-directed beats random, and the recommended production config for 2M books.
- Flag any book the certifier pair disagreed on (the review queue).

## Environment

- 4090 (24 GB). `pip install vllm datasets pandas pyarrow pyahocorasick httpx`.
- Windows paths in the scripts point at the Dropbox locations above — they resolve on this box via Dropbox sync.
- Keep datasets OUT of git (see `.gitignore`); commit only code, manifests, and `results.csv`.
