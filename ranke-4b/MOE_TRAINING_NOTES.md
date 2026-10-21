---
title: "Ranke mixture-of-experts training: what works"
---

**Version 1.0, 2026-10-21 08:25.** Claude, for JV.

This is a running notebook. Each entry has a date and the job that shows it. When something new is learned, add it here the same day and raise the version number.

**Evidence:**

- `MOE_SMOKE_RESULTS.html` (v1.1);
- the metrics in `G:\detritus\ranke-moe-smoke\runs\<job>\`;
- the log `D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\LOG.md`.

**Cluster rules** (queues, time limits, storage) live in `C:\Users\hvoth\cluster\CLUSTER_LESSONS.md`, not here.

## 1. What works

All of these were tested on 2 UZH H200 cards with the OLMoE-1B-7B design: 16 layers, 64 experts, 8 used per token, 6.9B parameters.

| # | What | Setting | Evidence |
|---|---|---|---|
| W1 | Training runs without invalid numbers (NaN) | `pad_token_id=None` (also bos/eos None) in `OlmoeConfig` | job 7378909: passes step 26; job 7379835: 300 steps, 0 NaN |
| W2 | Experts stay balanced | balance loss per layer, averaged over layers (`--balance-loss=per-layer`), coefficient 0.01, as in MegaBlocks | job 7379835: worst layer 1.86× (mark ≤3×), least busy 0.200× (mark ≥0.1×) at step 299 |
| W3 | Experts stay balanced (second half of the fix) | long warm-up: 200 steps to a peak learning rate of 4e-4 | job 7379835 against job 7379027: with a 20-step warm-up, balance failed at step 100 (4.5×) |
| W4 | Smooth validation curve | shuffle the 4096-token sequences across books, with a fixed seed (`--shuffle-seed 0`) | job 7379835: falls at every check, 12.35 → 4.87; book order went up and down (job 7379027) |
| W5 | Save and restart repeat the run | PyTorch distributed checkpoint (model + optimizer) and a fixed data order | job 7379027: restarted losses within 0.71% over 50 steps (mark 1%) |
| W6 | Memory and speed | FSDP2, bf16 weights, fp32 gradient reduction, activation recomputation, micro-batch 2 × 4096 | about 15,500–15,800 tokens per second per card; peak 73.5 of 141 GB |
| W7 | Save size and time | — | 82 GB in 43 s; reload about 1 min |
| W8 | Live monitoring | trainer flag `--tb` (TensorBoard) plus the JSON-line logs; watcher `C:\Users\hvoth\cluster\trainwatch\` | job 7379835 |
| W9 | Balance logged per layer | `balance.jsonl`: share of the busiest and least busy expert in each layer | this caught the collapse that the pooled loss hid |

## 2. What does not work (do not repeat)

| # | What | Why it fails | Evidence |
|---|---|---|---|
| F1 | Hugging Face default balance loss (pools all layers) | it reports "nearly even" (8.7; even = 8) while every layer has collapsed onto 8 experts | jobs 7377866 and 7377214 |
| F2 | Hugging Face default `pad_token_id=1` with the Qwen vocabulary | token 1 is the quote mark; its embedding stays frozen at zero → invalid number (NaN) at step 26 | jobs 7377866, 7378053, 7378135, 7378554 |
| F3 | Warm-up of 20 steps | the experts collapse in every layer by step 10 and recover too slowly | job 7379027 |
| F4 | Book-ordered training data | validation loss goes up and down: 6.98 → 7.22 → 6.27 | job 7379027 |
| F5 | CPU checks of the full model on the login node | the login node has about 1 GB memory per process; use a 1-card job instead | 2026-10-20 |

## 3. Ruled out as causes (no need to test again)

- The validation pass does not cause the NaN (job 7378135).
- The step-26 text alone does not cause it; a fresh model handles that text (job 7378335).
- The fused expert kernel `grouped_mm` does not cause it. The plain loop gives the same losses and the same NaN, at 3× lower speed (job 7378554).
- The memory-splitting layer (FSDP) does not cause the collapse. A plain model routes the same way (job 7377638).
- The balance loss is wired correctly: it reaches the router weights (job 7378135).

## 4. Not yet checked against the original OLMoE recipe

These are still Hugging Face defaults or our own choices.

- The starting weights: Hugging Face draws them with spread 0.02. OLMo-core's choice is not checked.
- The router precision: our router runs in bf16. MegaBlocks can compute the balance loss in fp32.
- The batch size: we use 131,072 tokens per step; OLMoE used about 4 million.
- The warm-up length: we use 200 steps; OLMoE used about 2,500.
- The z-loss coefficient: we use 0.001, which is the value in the plan.

**Proper check before a real run:** a short run of the same size with OLMo-core (Ai2's official OLMoE training code), compared on loss and balance.

## 5. Settings to copy

Trainer `moe_smoke_train.py`, job script `moe_warmup_test.sbatch`:

```
--balance-loss=per-layer --warmup 200 --shuffle-seed 0 --tb --val-every 25 --balance-every 10
OlmoeConfig(..., pad_token_id=None, bos_token_id=None, eos_token_id=None)
export PYTHONPATH=/scratch/ukrenk/pylibs/tensorboard
```
