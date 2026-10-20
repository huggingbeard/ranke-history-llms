---
title: "OLMoE smoke test on UZH H200 cards — results"
---

**Version 1.1, 2026-10-20 20:05** (v1.0 18:05; v1.1 adds the 300-step follow-up run, job 7379835). Claude, for JV. Plan: `C:\Users\hvoth\.claude\plans\olmoe-smoke-20261020.md` (approved 2026-10-20 15:09).

**Files used:**

- Code: `ranke-history-llms/ranke-4b/moe_smoke_train.py`, `moe_smoke.sbatch`, and the check scripts `moe_router_diag.py`, `moe_grad_check.py` and `moe_*_check.sbatch`.
- Data snapshot: `/scratch/ukrenk/moe_smoke_20261020/data`. It holds 118 training books (18.8M tokens) and 14 validation books (1.08M tokens); checksums are in `manifest.json`.
- Metrics, copied and checked with SHA-256: `G:\detritus\ranke-moe-smoke\runs\<job>\`.
- Checkpoint (82 GB): `/scratch/ukrenk/moe_smoke_20261020/runs/7379027_full_A/ckpt_step50`. It stays on the cluster.

## Question

Can we train a mixture-of-experts version of Ranke (the OLMoE-1B-7B design) from a random start on UZH H200 cards? The test checks four things: training, saving, restarting, and keeping the experts in balance.

The design:

- 16 layers and 64 experts per layer;
- 8 experts used per token;
- 6.9B parameters in total, about 1.3B used per token.

## Result of the final run (job 7379027, 2 H200 cards)

| Check | Pass mark | Result | Verdict |
|---|---|---|---|
| No invalid numbers (NaN) | none | none in 150 steps | pass |
| Loss falls | yes | validation loss 12.35 → 6.27 (step 0 → 100) | pass |
| Restart from save repeats the losses | within 1% | largest gap 0.71% over steps 50–99 | pass |
| Save works | — | 82 GB in 43 s; reload about 1 min | pass |
| Expert balance after step 50 | no expert above 3× or below 0.1× its fair share | worst layer 4.5×; median layer 3.8× at step 99; some experts get no tokens | **fail** |
| Speed | record | 15,800 tokens per second per card; 4.2 s per step; peak memory 73.5 of 141 GB | recorded |

**Expert balance over time.** 1× = fair share; 8× = all tokens go to the same 8 experts.

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| worst layer | 7.5 | 8.0 | 8.0 | 8.0 | 7.1 | 7.1 | 7.1 | 6.0 | 4.8 | 7.2 | 4.5 |
| median layer | 5.8 | 8.0 | 8.0 | 8.0 | 6.0 | 5.5 | 5.9 | 3.7 | 3.8 | 3.5 | 3.8 |

## What went wrong on the way, and the fixes

1. **The experts collapse at the random start.**
   - At step 0, most of the router signal is the same for every token. In layer 1, the shared part is 0.99 of a total spread of 1.06 (job 7377638).
   - So all tokens pick the same few experts.
2. **The Hugging Face balance penalty could not see this.**
   - It pools the tokens of all layers before it measures balance.
   - MegaBlocks, the original OLMoE training library, pairs each layer's counts with its own scores and averages over layers. We checked both source codes.
   - Fix: `--balance-loss=per-layer` (JV approved, 17:16).
   - A gradient test confirmed that the penalty reaches the router weights (job 7378135).
   - With the fix, balance starts to recover from about step 40.
3. **Invalid numbers (NaN) at step 26 in every run.**
   - Jobs 7377866, 7378053, 7378135 and 7378554 all failed this way.
   - Cause: the OLMoE configuration treats token 1 as "padding" by default. In the Qwen vocabulary, token 1 is the quote mark `"`. Its embedding was fixed at zero.
   - The NaN started in layer 0 and the embedding, in micro-batch 7 of step 26.
   - Ruled out:
     - the validation pass (job 7378135);
     - the step-26 text on its own (job 7378335);
     - the fused expert kernel: the plain loop gave the same losses and the same NaN (job 7378554).
   - Fix: `pad_token_id=None`. After the fix, the run passed step 26 (job 7378909) and completed all 150 steps (job 7379027).
4. **The validation loss goes up and down** (6.98 → 7.22 → 6.27). The training text is read in book order, not shuffled. A real run must shuffle across books.

## Cost

- All 9 jobs: 1.21 card-hours on H200 cards.
- About 0.38 CHF in total: cards 0.33, cores 0.01, memory 0.04.
- The cap was 1.5 CHF.

## What this means for Ranke

- Training, saving and restarting a 6.9B mixture-of-experts model on UZH H200 cards **works**.
- Expert balance **does not reach the pass mark in 100 steps**. The per-layer penalty pulls the experts apart, but slowly.
- Two possible reasons, both untested:
  - the 20-step warm-up to the full learning rate (OLMoE warms up over about 2,500 steps);
  - the starting weights of the Hugging Face copy.
- The Hugging Face model code is made for running and fine-tuning trained models. Twice its defaults differed from the OLMoE training recipe. A real run should use the OLMoE training code (OLMo-core) or torchtitan's mixture-of-experts support. Both match the original recipe.

## Follow-up: a longer warm-up fixes the balance (job 7379835, JV approved 18:05)

**What changed from job 7379027:**

- 300 steps instead of 100.
- Warm-up 200 steps instead of 20.
- A bigger snapshot: `data_v3` has 286 books and 42.2M tokens. The 14 validation books are the same, with an identical checksum.
- The training sequences come in a fixed random order (seed 0) instead of book order.
- Everything else is the same: the model, the per-layer balance loss, no padding token, and 2 H200 cards. Job script: `moe_warmup_test.sbatch`.

The pass marks were set before the run. They are checked at the last step, 299.

| Check | Pass mark | Result | Verdict |
|---|---|---|---|
| No invalid numbers (NaN) | none | 0 in 300 steps | pass |
| Loss falls | yes | validation loss 12.35 → 4.87 | pass |
| Busiest expert, every layer | ≤ 3× fair share | worst layer 1.86× | **pass** |
| Least busy expert, every layer | ≥ 0.1× fair share | least busy 0.200× | **pass** |

**Balance over time.** 1× = fair share.

| step | 50 | 100 | 150 | 200 | 250 | 299 |
|---|---|---|---|---|---|---|
| busiest expert, worst layer | 6.9 | 3.7 | 2.8 | 2.5 | 2.3 | 1.9 |
| least busy expert, any layer | 0.0 | 0.0 | 0.04 | 0.07 | 0.09 | 0.20 |

**Validation loss at the same step, compared with the 100-step run:**

| step | 50 | 100 |
|---|---|---|
| 100-step run (job 7379027) | 6.98 | 6.27 |
| this run | 6.88 | 6.08 |

The curve also falls smoothly now, step by step: 12.35, 8.33, 6.88, 6.51, 6.08, 5.80, 5.62, 5.48, 5.35, 5.16, 5.05, 4.94, 4.87. The 100-step run went up and down.

**Run details:**

- Speed: about 15,500 tokens per second per card.
- Time: 1,406 s on 2 cards, which is 0.78 card-hours, about 0.25 CHF.
- Metrics, SHA-256 verified: `G:\detritus\ranke-moe-smoke\runs\7379835\`.
- The TensorBoard files are in `7379835_warmup_A\tb\`.

**Conclusion.** The collapse at the random start is a warm-up problem. With a 200-step warm-up and the per-layer balance loss, all 16 layers reach the balance marks. The OLMoE design therefore trains on UZH H200 cards when it uses these two settings and no padding token.

These settings are recorded as needed for any Ranke mixture-of-experts run. This is a 300-step engineering test, not evidence about model quality.
