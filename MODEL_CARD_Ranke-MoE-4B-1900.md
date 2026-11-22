# Model card: Ranke-MoE-4B-1900 (internal test model, 2026-11-22)

A time-locked language model that has read only text written before 1900. It is a mixture-of-experts (MoE) model:
each layer has 64 small expert networks, and a router picks 8 of them for every token. It was trained as a fair test
against the dense **Ranke-4B-1900 v2** (August 2026), to decide how the 50,000 GH200 card-hour grant is spent on the
next fleet of Ranke models. **Not released.** Weights are not in this repository (see "Files").

## Size (counted from the weight file)

| | Parameters |
|---|---|
| **Total (stored)** | **4.19 billion** (4,189,954,048) |
| **Used per token** | **1.02 billion** (1,019,060,224) |
| Experts, all layers | 3.62 billion (8 of 64 used per token: 453 million) |
| Attention, routers, norms | 255 million (always used) |
| Word table, shared by input and output | 311 million (always used) |

Weights: 8.4 GB in bf16 (Hugging Face `Qwen3MoeForCausalLM` format).

## Architecture (Qwen3-MoE layout)

- 24 layers, model width 2,048; experts are SwiGLU blocks of width 384; 64 experts per layer, top-8, router
  weights renormalised; no shared expert.
- Attention: 16 query heads sharing 4 key-value heads, head size 128; RoPE base 1,000,000; context 4,096 tokens;
  documents packed without attention across documents.
- Vocabulary: Qwen3 tokenizer, 151,936 tokens (identical files to Ranke-4B).

## Training

| Item | Value |
|---|---|
| Data | pre-1900 documents of the August 2026 Ranke-4B training data (same documents and year rule as dense): 93.8B unique tokens; sources: institutional books 38%, American Stories newspapers 26%, other English books 18%, French 7%, German 6%; 64% English, 15% German, 12% French; 0.1% of documents held out by id hash |
| Tokens read | 86.3B (= dense Ranke-4B-1900 v2): 41,160 steps x 512 x 4,096; 84.5B unique, 1.9B re-read |
| Optimizer | AdamW (betas 0.9/0.95, weight decay 0.1) |
| Learning rate | warm-up 500 steps, then constant 6e-4; no cool-down (dense v2 had none either) |
| Expert balance | balance loss 0.01 per micro-batch (one card's 32,768-token share of a step) |
| Hardware | 8 H200 to 61.9B tokens (~490 card-hours); 8 H100 to 86.3B tokens (31.4 h, ~251 card-hours) |
| Speed | 35,367 tokens/s per H200 card (30.5% of peak; dense v2: 42.7%); 7.9 card-hours per billion tokens (dense: 27.6) |
| Software, runs | torchtitan 0.4.0; Slurm 8513359 (pilot), 8519163 (main), 8620897 (extension to 86.3B) |

## Results against dense Ranke-4B-1900 v2 (both at 86.3B tokens; same pieces, same scorer)

Bits per byte on held-out text (lower is better); 10,000 pieces per set.

| Test text | MoE | Dense | MoE / dense (95% interval) |
|---|---|---|---|
| 1900-1913, unseen by both | 1.0356 | 1.0130 | 1.0223 (1.0210-1.0238) |
| 1914-1929, unseen by both | 1.0830 | 1.0600 | 1.0217 (1.0205-1.0229) |
| pre-1900 held-out (dense saw these) | 0.9845 | 0.9731 | 1.0117 (1.0093-1.0142) |

At 61.9B tokens the 1900-1913 gap was 1.0295. Leak test (calibrated date-spike + matched pairs, 565 events) at 61.9B
tokens: no sign of post-1899 knowledge; not yet repeated on the final model.

## Files

- Weights (Hugging Face format): `G:\detritus\ranke-moe-4b\hf_ext86_step41160\`;
  `model-00001-of-00001.safetensors` SHA-256 `1e5581a67f48e56508512dd1e4a6c18fd9375f5de0dbffc7c2aae6d03dc95787`.
- Training save (torchtitan, 8 shards): `/scratch/hvoth/ranke_moe_4b/runs/base_ext86/checkpoint/step-41160` (cluster).
- Code, logs, plan: `history-LLM-Ranke/experiments/moe/` (local repo).
