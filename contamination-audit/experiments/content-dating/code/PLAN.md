# Experiment — whole-document content dating

**Question.** Our edition-dater reads only front matter (~3% of a book). It cannot see post-1929
*content* buried in the body. This experiment measures how reliably a model can date a book's **content**
from sampled body passages, and finds the **cheapest configuration** that gives high recall.

**Recall is the target. Precision is secondary** (over-removing is cheap; keeping contamination is not).

## Data — 2,000 books, one source family (institutional_books) so the model can't cheat on OCR/format

| Class | n | Definition | Source |
|---|---|---|---|
| **clean** (pre-1929) | 1,000 | all date signals ≤1929, no anachronism markers, deepseek-confirmed pre-1929 original | our kept corpus (local sample) |
| **dirty** (post-1929) | 1,000 | **modern original** content (NOT a reprint of an old work); metadata+stage2 both >1946; confirmed | `institutional_books` on HuggingFace, streamed & filtered |

Label is by **construction** — the whole book is one class, so a 1965 book means *every* passage is
"post-1929". No per-passage hand-labeling. The experiment measures how *detectable* that is.

**Critical filter for positives:** exclude reprints of old works. A 1960 reprint of an 1850 novel has an
*old* body and would poison the positive class. Keep only genuine post-1946 authorship.

## Classifier

Per book: sample N passages of length L. Each passage → model judges "written before or after 1929?".
Aggregate → book decision. Primary aggregation = **any-positive** (one modern passage ⇒ flag; recall-max).

## Variables swept

| Axis | Values | Notes |
|---|---|---|
| A · model size | **dense scaling curve:** Qwen2.5 {0.5,1.5,3,7,14B}. **cross-family:** Llama-3.2 {1,3B}, Phi-3.5-mini. **strong+fast production candidates:** Qwen3.6-35B-A3B (MoE, ~3B active → fast despite 35B total), gpt-oss-20b | all on the 4090 via quantization |
| B · passages/book N | 1, 2, 4, 8, 16 | **derived by subsetting** the max-N run (no re-run) |
| C · snippet length L | 250, 500, 1000, 2000 chars | real re-run axis (model must see the actual length) |
| D · sampling | random, grep-directed | grep-directed = anachronism-vocab points model at suspicious passages |

**Model roles:** the dense Qwen2.5 ladder gives the clean "recall vs model size" curve; Llama/Phi check it
isn't Qwen-specific; **Qwen3.6-35B-A3B and gpt-oss-20b are the production candidates** — the key question
for them is whether a fast, strong model hits recall ≥ 0.95 with *few short* passages, because that is what
must scale to 2M books (the A3B MoE is fast per token; whether it holds up on 2M is exactly what we test).

**Efficiency:** run each model once on the full 16-passage set per (L, strategy); derive all N from subsets.

## Metrics

Book-level precision, recall, F1, AUC as surfaces over A×B×C×D. Headlines:
1. cheapest config with **recall ≥ 0.95**
2. how few / how short passages a *small local* model needs
3. how much grep-direction beats random

## Files

```
build_negatives.py    select 1,000 clean books from our corpus (RUN HERE)
pull_positives.py     stream institutional_books from HF, filter post-1946 modern-original (RUN ON OFFICE PC / here)
extract_passages.py   random + grep-directed passages at each length -> Dropbox
run_flotilla.py       vLLM driver: models x passages -> judgments CSV (4090)
analyze.py            metrics + recall/precision surfaces
manifest.csv          book ids + labels + source (in repo; NO text)
```

Data (book text, passages, model outputs) lives in **Dropbox**
(`.../experiments/content-dating/`), never in git. Manifest (ids+labels) lives in the repo.

## Phases

1. **Pilot** — 50 clean + 50 dirty, 2 models (small + mid) — shake out harness + confirm labels.
2. **Source positives** — the main data lift (HF pull + confirm).
3. **Full sweep** — 4090, overnight.
4. **Analyze** — surfaces → recommended config.

## Compute budget (rough)

~1,000 books × 3 lengths × 2 strategies × up to 16 passages = ~96k judgments per model. Small models
(0.5–3B) are seconds-fast batched; 14B is the bottleneck. Feasible overnight on one 4090 with vLLM.
Start with 500+500 and 3 lengths if time-boxed.
