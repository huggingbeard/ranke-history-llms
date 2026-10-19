# Plan: does the Qwen3.5 design give a better Ranke model? A small test before any full run

**Version 1.0, 2026-10-19.** Claude, for JV. Status: PLAN, nothing run. Every number marked [ASSUMED] or [TBD] needs checking.
**Files used:**
- `C:\Users\hvoth\Documents\GitHub\ranke-history-llms\ranke-4b\prerelease_notes.md` (how Ranke-4B was trained);
- the Qwen3.5 design files on Hugging Face (`Qwen/Qwen3.5-0.8B-Base/config.json`, `Qwen/Qwen3.5-35B-A3B/config.json`);
- the Ranke-4B design file on the UZH cluster (`/scratch/ukrenk/hf-models/Ranke-4B-1913@99fd739…/config.json`).

## 1. Why now

- **Ranke-4B was sized for 80 billion training tokens.** It is "Chinchilla-optimal": about 20 tokens per parameter (`prerelease_notes.md`, line 4). It uses the dense Qwen3 design: one network, and every part works on every token.
- **We now have up to 600 billion tokens** [TBD: before or after cleaning]. That much text could train a much larger model, or a mixture-of-experts model.
  - A mixture-of-experts model has many sub-networks ("experts") and uses only a few per token. It has more capacity at the same cost per token.
- **The newer Qwen3.5 design** changes three things at once:
  - most layers use "linear attention", which is cheaper for long texts;
  - the vocabulary is larger (248,320 tokens against 151,936);
  - its large versions are mixtures of experts.
- **Before anyone pays for a full run** (about 3,000–4,000 CHF on the UZH cluster, or the same 10,000+ card-hours at Jülich [ASSUMED]), a small run must show that the new design actually helps on Ranke's old text.

## 2. The question, and how we decide

**Question:** at the same training compute, on the same pre-1900 text, does a Qwen3.5 design predict held-out old text better than today's Qwen3 design?

**Pass mark, fixed before any run:**
1. **Main measure: bits per byte on held-out pre-1900 text.**
   - Bits per byte = how many bits the model needs, on average, to predict each byte of text it has not seen.
   - Lower is better.
   - I use bytes, not tokens, because the two designs use different vocabularies, so their per-token losses cannot be compared.
2. **A design "wins" if its bits per byte are at least 2% lower** than the Qwen3 design's, at equal training compute. It must also be no worse on the year tests (step 5).
3. **Mixture of experts "wins"** if it is at least 3% lower at equal compute per token, and its training speed per card is at most 30% lower.
4. **Anything smaller is a tie.** On a tie we keep the current design, because switching has costs (section 6).

## 3. The models (all trained from scratch; only the designs are reused, never Qwen's weights)

| Model | Design | Size | Vocabulary | What it tests |
|---|---|---|---|---|
| **A** (reference) | Qwen3, dense, standard attention (today's Ranke design) | ~0.6B (Qwen3-0.6B shape) | Qwen3, 151,936 | baseline |
| **B** | Qwen3.5, dense: 18 of 24 layers linear attention | ~0.8B (Qwen3.5-0.8B shape) | Qwen3.5, 248,320 | new design + new vocabulary |
| **B′** | same as B | ~0.8B | **Qwen3** vocabulary | separates "design" from "vocabulary" |
| **C** | Qwen3.5 mixture of experts, scaled down by us | ~1B in total, ~0.3B used per token [ASSUMED shape: 24 layers, 32 experts, 4 per token] | Qwen3.5 | mixture of experts |

- **B against B′** tells us how much of any gain comes from the vocabulary alone.
- **Model A and model B differ in size.** The 0.8B partly reflects the larger vocabulary table. So all comparisons are at **equal training compute** (≈ 6 × parameters used per token × tokens), not equal size. I also report bits per byte against compute as a curve, from saved checkpoints.

## 4. Data and training settings (copied from Ranke-4B wherever possible)

- **Text:** a fixed sample of the pre-1900 Ranke training text.
  - Quick version: **4 billion tokens** per model.
  - Fuller version: **16 billion tokens** (Chinchilla amount for about 0.8B).
  - The same documents in the same order for every model, each cut with its own vocabulary.
- **Held-out set:** 50 million tokens of pre-1900 text never used in training (a separate set of whole documents).
- **Settings as in `prerelease_notes.md` line 61:**
  - AdamW;
  - warm-up then stable then decay learning rate (500 warm-up steps, decay over the last 20%);
  - 4,096-token pieces;
  - about 2 million tokens per step.
- **Peak learning rate:** a short sweep at 3 values per design on 0.5 billion tokens. Each design gets its best rate; a fixed rate could favour one design by accident.
- **Seeds:** one seed each in the quick version. The fuller version uses 2 seeds for A and the winner, to see run-to-run noise.

## 5. Evaluation (all local or on the same cluster; no paid calls)

1. **Bits per byte** on the 50-million-token held-out set (main measure).
2. **Bits per byte by text type:** books, newspapers, non-English text (French, German, Spanish). The larger vocabulary may help mainly the non-English part.
3. **The existing Ranke year tests** (`ranke-decontamination\model-audit\july\`, the date tests).
   - At 0.6–1B parameters and 4 billion tokens, these may be too hard for every model. They then serve only as a check that the new design is "not worse".
4. **Training speed:** tokens per second per card for each design. It sets the cost of a full run.

## 6. What switching would cost (decided only if a design wins)

- Retraining the whole Ranke family: the 1900 base model and each cutoff model (1913, 1929, 1933, 1939, 1946).
- New models are not comparable one-to-one with the current Ranke-4B. The old and new families need the same tests side by side.
- Training code: the Ranke training code is a fork of torchtitan. Whether it already has the Qwen3.5 layers (linear attention, mixture of experts) is [TBD]; check this first (step 1 of section 7).

## 7. Steps

| # | Step | Where | Pass mark / stop rule |
|---|---|---|---|
| 1 | 10-step test of designs A, B, C on one card | Jülich or UZH | all train without error; loss falls (fallback below) |
| 2 | Build the 4-billion-token sample + 50-million-token held-out set; cut it with both vocabularies; save hashes | where the corpus lives (CSCS/Jülich) | same documents for all; hashes recorded |
| 3 | Learning-rate sweep, 3 rates × 4 models × 0.5B tokens | 1 card each | pick the best rate per model by held-out bits per byte |
| 4 | **Quick run:** A, B, B′, C × 4B tokens; checkpoint every 0.5B tokens | 1–4 cards per model | finish; bits-per-byte curve saved per checkpoint |
| 5 | Evaluation (section 5) + report against the pass mark (section 2) | 1 card | **STOP here** unless a design wins by the pre-set margin |
| 6 | Fuller run (16B tokens, 2 seeds for A and the winner) only if step 5 shows a win | Jülich (preferred) | win holds at 4× the data |

**Fallback for step 1:** if the Ranke torchtitan fork lacks the Qwen3.5 layers, train all four models with Hugging Face transformers + FSDP instead. Model A is then re-trained there too, so all four share one code base.

## 8. Cost and time [ASSUMED from Ranke-4B's measured speed and standard compute rules]

- **Compute per quick model:** 6 × 0.6–0.8B × 4B tokens ≈ 1.5–1.9 × 10¹⁹ operations.
- **Card time:**
  - H200 at 40% of its peak speed: about **11–14 card-hours per model**.
  - Mixture of experts and linear attention often run at 25–30% of peak, so up to about 20 card-hours.
- **Quick version** (4 models + sweep + checks): about **70–90 card-hours**.
  - UZH cluster, H200 at 0.305 CHF per hour: **about 25 CHF**.
  - Jülich GH200: similar card-hours, under Guohui's allocation [TBD: which allocation, and whether it has room].
- **Fuller version:** about 4 times the quick version, about 300–350 card-hours (about 100 CHF on the UZH cluster).
- **Wall time:** the quick version takes 1–2 days on 4 cards, plus queue time.

## 9. Where to run it

| | Jülich (via Guohui) | UZH cluster (our login ukrenk) |
|---|---|---|
| Cards | GH200, like CSCS Alps where Ranke-4B was trained | H200 and A100 |
| Training code | the Ranke torchtitan fork, if it runs there [TBD] | must be set up; follow `ranke-edition-dating\validation\cost_test\CLUSTER_VLLM_RUNBOOK.md` (login-node check first) |
| Data | close to the corpus if Guohui has a copy [TBD] | needs a copy of the 4B-token sample (≈ 16 GB per vocabulary [ASSUMED]) |
| Who runs it | Guohui | Claude, under JV's approval |
| Best for | the fuller version (step 6) | the quick version (steps 1–5), if the data can be copied |

**Recommendation:**
- Quick version on the UZH cluster, if Guohui can send the 4-billion-token pre-1900 sample. It is cheap (about 25 CHF), and we control the queue.
- Fuller version at Jülich with the Ranke code, only if the quick version shows a win.

## 10. Open facts needed before step 1

1. Is "600 billion tokens" counted before or after cleaning and de-duplication? How many tokens exist before 1900?
2. Where is the Ranke training code (torchtitan fork), and who can run it at Jülich? Does it support Qwen3.5 layers?
3. Can Guohui provide the 4-billion-token pre-1900 sample and the 50-million-token held-out set?
4. Which Jülich allocation would pay, and does it have room for about 100 card-hours?
