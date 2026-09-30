# Talkie x Ranke-1929 post-training — PLAN v2 (period models judge each other)

Version: v2.2 · Date: 2026-09-28 23:35 (v2.0 23:20; v2.1 JV changes: judges, Codex job; v2.2 DeepSeek-V4-Pro round-1 fixes + A100) Europe/Zurich · Plan ID: `talkie-ranke-mutual-0928`
Author: Claude Opus 5.5 (taking over from Codex). Supersedes `PLAN.md` (v1, kept unchanged).
Files used:
- `G:\detritus\ranke1929_talkie_sft_20260928\PLAN.md`, `STATUS.md`, `MODEL_LOCK.json`
- `G:\detritus\ranke1929_talkie_sft_20260928\scripts\generate_talkie_pairs.py`, `train_ranke_lora.py`, `judge_blind_outputs.py`
- `G:\detritus\ranke1929_talkie_sft_20260928\eval\frozen_eval_84.jsonl` (SHA-256 156eb18d…6277), `eval\JUDGE_PROMPT.md`
- `G:\detritus\ranke_period_qa_scale_20260928\data\history_passages_v1.jsonl` (2,077 passages, SHA-256 803d7139…ba04)
- Talkie source: `C:\Users\hvoth\AppData\Local\Temp\talkie-upstream\src\talkie\model.py`, `generate.py`

## CHECKED (what exists, so nothing is rebuilt)
- STATUS.md 23:05: "No full Talkie generation has started yet." One Talkie record was ever produced (A100 gate, 235.5 s, failed format; fix = assistant prefix `<question>`, never re-tested). Codex gate `6510f2b9ee` / Slurm `6657946` was PENDING (Priority).
- Base Ranke answers to all 84 frozen prompts exist: `experiments\base84_20260928_172133\eval\blind_output_matrix.jsonl` (84 rows). Surface stats: 74/84 hit token ceiling, 72/84 repetition, 29/84 invent a new `Question:` turn.
- v1 plan used a **modern selector** (`select_talkie_with_luna.py`) to accept/reject Talkie records. v2 removes it.
- Talkie code (`model.py`): `forward()` has **no key/value cache**; every new token recomputes the whole sequence. `_stream_raw` is one prompt at a time. This is why one record took ~4 minutes; 4,154 records would take days.
- Training script exists and is sound (answer-only loss, LoRA r8 q/v, 4-bit base). Reused unchanged for SFT.
- Existing modern eval judge: Codex `gpt-6-astra` through `codex exec` with `eval\JUDGE_PROMPT.md`.

## The rule JV set (hard constraint)
No modern model creates, edits, selects, filters, or ranks any training text. Only the two period models do that:
**Talkie-1930-13B-IT** (cutoff 1930) and **Ranke-4B-1929 v2** (cutoff 1929). Modern models appear only in the final evaluation of saved answers.

## 1. Construct
Does training Ranke-1929 on period-written, passage-grounded question-and-answer data — with the period models as each other's judges — make Ranke answer historical and political questions better (more correct and reasoned, direct, complete, period-fitting), and does more data help more?

## 2. Mechanism (four stages)
**Stage A — Talkie writes.** Unchanged locked prompt, model revision, forms (6 question types), sampling (v1 generator defaults: temperature 0.55, top_p 0.90, top_k 40, max 512 new tokens, per-record seed), assistant prefix `<question>`. Two candidates per passage → up to 4,154 records. Run with a new **batched, key/value-cached decoder** that uses Talkie's exact weights and layer code (only the cache is new). Every raw completion saved.

**Stage B — Ranke judges Talkie (grounding).** Mechanical rejection first (v1 rules: schema, 90–220 words, copy-overlap, protected eval terms, duplicate questions). Then Ranke-1929 base scores each surviving record by *grounding gain* = mean per-token log-probability of the answer given (passage + question) minus the same given (question only). High gain = the answer uses what the passage says; near zero or negative = generic filler or content not in the passage (the main risk of Talkie inserting 1913–1929 facts). Ranke also scores answer fluency (mean log-prob given question only) to drop garbled text. Rule, fixed before seeing scores: keep records with grounding gain above the 20th percentile and fluency above the 5th percentile. Ranke produces numbers only, no text.

**Stage C — SFT dose curve.** One seeded order of accepted records; nested sets of 300 / 1,000 / 3,000 (or all accepted if fewer than 3,000, stated in the report). Train three LoRA adapters with the unchanged v1 script and settings (r8, α16, q/v, lr 2e-4, 3 epochs, max length 1,024, seed 1929).

**Stage D — Talkie judges Ranke (preference round).** 600 held-out Talkie questions (passages whose records were NOT used in SFT). The SFT-3000 adapter (chosen in advance, not by eval results) writes 4 sampled answers per question. Talkie sees the passage, the question and two answers, and answers "A" or "B" (we read its next-token probabilities, no free text). Each pair is judged in both orders to cancel position bias; only pairs where Talkie prefers the same answer in both orders count. Best vs worst answer per question → preference pairs → one DPO adapter on top of SFT-3000 (β 0.1, 1 epoch, lr 5e-6).

## 3. Threats to validity and how the design handles them
1. *Talkie leaks 1913–1929 facts.* Grounding gain penalizes content not supported by the pre-1913 passage; the eval scores anachronism separately (period_fit). Not fully closed: a supported-looking sentence can still carry a late term. Reported, not hidden.
2. *Grounding gain rewards copying.* Copy-overlap rule removes near-copies first; we report the gain distribution against copy overlap.
3. *Talkie judge has position bias or cannot tell good from bad.* Both-order rule; plus a validation test with a known answer (below). If it fails, Stage D is dropped and reported as failed.
4. *Gains are only shorter/cleaner answers (format learning), not history.* Report substance separately from directness/fluency; report length and end-of-answer rates; compare paired per prompt.
5. *Training-eval overlap.* Protected eval terms removed from training (v1 rule); report low-overlap and transfer items separately.
6. *Different decoding between conditions.* All five conditions — including base, regenerated — use one script, one decoding setting, same machine. The old base matrix is kept as a cross-check.
7. *Judge bias toward a model.* Replies anonymized and shuffled per item; two independent modern judge families (DeepSeek-V4 Flash; Claude Sonnet); agreement reported.
8. *Dose confounded with number of optimizer updates.* Same as v1: stated as a dose curve, not a token-matched estimate.

## 4. Validation tests with pass/fail bars (run before scaling each stage)
- **V1 cache decoder:** on 8 prompts, greedy 64 tokens with the cache must equal the original uncached `forward()` output on ≥ 7/8 prompts, and last-step logits max abs difference < 0.5 (bf16). Fail → fall back to uncached batched decoding (slower) or stop.
- **V2 Talkie format:** 16-record smoke; pass if ≥ 12/16 parse into question+answer and the answers read as grounded on inspection (I read all 16 and save them).
- **V3 Ranke grounding judge:** compare each true (passage, answer) pair with the same answer put next to a random other passage. Pass if the true passage gives higher gain in ≥ 90% of 300 pairs.
- **V4 Talkie preference judge:** 100 pairs of (Talkie's own grounded answer) vs (base Ranke answer to the same question — known to be poor: 74/84 base answers run to the token ceiling). Pass if Talkie prefers the Talkie answer in ≥ 80% and gives the same choice in both orders in ≥ 70%.
- **V5 training rehearsal:** 32-record, 1-epoch training run completes, loss falls, adapter reloads and generates.

## 4b. Changes after DeepSeek-V4-Pro review round 1 (`review\deepseek_v4pro_round1.md`)
- **Harder test of the Ranke grounding judge (V3b).** Build corrupted answers automatically: replace one middle sentence of a Talkie answer with a sentence from an answer about a different passage. Pass if the intact answer gets the higher grounding gain in ≥ 80% of 300 pairs. Also: I read 40 accepted and 40 rejected records and save them to an inspection page for JV. This reading checks the filter only; no single record is added, removed or changed by hand or by any modern model.
- **Harder test of the Talkie preference judge (V4, replaced).** Pairs are (intact Talkie answer) vs (the same answer with one sentence swapped in from an unrelated answer), and vs (the same answer cut off halfway). Pass if Talkie prefers the intact answer in ≥ 75% of 200 pairs, with the same choice in both orders in ≥ 70%. Fail → the preference round is dropped and the report says so. The easy test (Talkie answer vs base Ranke answer) is also reported, but it does not decide.
- **Yield pilot.** After the 16-record format check, a 200-record pilot runs through every filter. If the projected number of accepted records is below 3,000, a third candidate per passage is generated (the next question type in the fixed rotation) before the full run. The 300/1,000/3,000 doses stay fixed; if the final pool is still below 3,000, the top dose is "all accepted" and the report states the number.
- **Time limit.** If Talkie writing is not finished 3 hours after the pod starts, it stops; a random passage subset of fixed seed is used, and the report says so.
- **End-to-end dry run.** 32 records go through every stage (write, judge, train, answer, save) before the full run.
- **Eval overlap.** Reject any training question whose word-set overlap with any of the 84 eval prompts exceeds 0.5 (Jaccard), in addition to the protected-term list.
- **DPO monitoring.** 10% of preference pairs are held out; report preference accuracy and reward margin on them.
- **Statistics.** Add paired permutation tests next to the bootstrap intervals; state that 84 prompts give limited power.
- **Dates.** The latest source passage was published in 1906 (`MODEL_LOCK.json`).

## 5. Evaluation (modern models only here)
Five conditions answer all 84 frozen prompts: base, SFT-300, SFT-1000, SFT-3000, SFT-3000+DPO. Same wrapper as training (`Question: …\nAnswer:`), greedy, per-item `max_new_tokens` from the frozen file, stop at a new `Question:` line (recorded, reported).
Judges (JV, 23:22): **DeepSeek-V4 lite/Flash** via OpenRouter and **Claude Sonnet** (on-plan subagents, no metered Anthropic API), both with the unchanged `eval\JUDGE_PROMPT.md`. Codex is no longer a judge. Report per condition: means with bootstrap 95% intervals, paired per-prompt wins/losses vs base, dose slope, history-substance on the 36 history prompts, period fluency on all 84, Russo-Japanese-War/Angell slice, length and completion rates, full answer examples.
Decision rule unchanged from v1 (effect: any trained condition beats base on substance without more anachronism; slope: SFT-3000 beats SFT-1000 by ≥ 5/100, in ≥ 4 of 6 reasoning groups, and wins more prompts than it loses). DPO effect: SFT-3000+DPO vs SFT-3000, same rule.

## 6. Compute and cost
- Cluster queue held every Talkie job PENDING (Priority) for hours today. So: **one RunPod A100 80 GB** (JV 23:26: cheaper; 80 GB holds Talkie 13B at 26.6 GB plus cache, and Ranke 4B), ~$1.2–2.5/hour. Expected 6–8 hours → **~$10–20; hard cap $40**. Pod self-destructs after 9 hours by an on-pod timer, plus a local watchdog. Teardown triple-check: (1) terminate call response, (2) API pod list is empty, (3) second list 5 minutes later is still empty.
- DeepSeek-V4-Pro plan review ~$0.10; DeepSeek-V4 Flash eval judging (84 items × 5 replies in one call per item) < $1.
- Sonnet eval judging: on-plan subagents, no metered cost.
- Codex's Talkie cluster job `6510f2b9ee` / Slurm `6657946` is confirmed CANCELLED (0 s used) and its 5-minute Codex heartbeat is PAUSED (checked 23:24). No Talkie job is in the cluster queue.

## 7. Saved per-item evidence (N > 9 rule)
Every stage writes one JSONL row per item with inputs, raw output, scores, model revision, seed, and hashes, with resume: `talkie_raw.jsonl`, `mechanical.jsonl`, `ranke_grounding.jsonl`, `sft_300/1000/3000.jsonl`, `ranke_samples.jsonl`, `talkie_prefs.jsonl`, `dpo_pairs.jsonl`, `eval_matrix.jsonl`, `judge_sonnet/*.json`, `judge_deepseek_flash/*.json`. All copied back to `G:\detritus\ranke1929_talkie_sft_20260928\runs\v2_<timestamp>\` before teardown.

## 8. What I will NOT do
No modern model sees training text. No change to the frozen eval, judge prompt, source corpus, Talkie prompt, or Ranke base revision. No change to v1 SFT settings. No results called final before JV review.

## Heavy steps this plan runs
1. OpenRouter calls: DeepSeek-V4-Pro plan review (until VERDICT: SHIP); later DeepSeek-V4 Flash eval judging.
2. RunPod: create one H100/A100 pod; run Stages A–D + eval generation; terminate.
3. Training: SFT ×3 (+ rehearsal), DPO ×1 on the pod.
4. Cluster: none (Codex job already cancelled).

Approved by JV 2026-09-28 23:22 (`GATE-APPROVE talkie-ranke-mutual-0928`) with the judge changes above.
