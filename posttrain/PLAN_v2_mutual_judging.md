# Talkie x Ranke-1929 post-training — PLAN v2 (period models write and judge; SFT dose curve)

Version: v2.5 · Date: 2026-09-28 23:52 Europe/Zurich · Plan ID: `talkie-ranke-mutual-0928`
History: v2.0 23:20 first draft · v2.1 23:25 JV changes (eval judges, Codex job) · v2.2 23:33 DeepSeek-V4-Pro round-1 fixes, A100 (copy: `review\PLAN_v2.2_snapshot.md`) · v2.3 23:40 JV drops the preference (DPO) round; section 4 rewritten to match (DeepSeek round-2 must-fix) · v2.4 23:47 DeepSeek round-3 fixes (late-fact word list, gain > 0 floor, injection test, answer-overlap check, small-pool dose rule, judge agreement statistics) · v2.5 23:52 DeepSeek round-4 fixes (larger decoder test, batch-size-1 cross-check, longer late-fact list, judge calibration check).
Author: Claude Opus 5.5 (taking over from Codex). Supersedes `PLAN.md` (v1, kept unchanged).
Files used:
- `G:\detritus\ranke1929_talkie_sft_20260928\PLAN.md`, `STATUS.md`, `MODEL_LOCK.json`
- `G:\detritus\ranke1929_talkie_sft_20260928\scripts\generate_talkie_pairs.py`, `validate_talkie_pairs.py`, `train_ranke_lora.py`, `judge_blind_outputs.py`
- `G:\detritus\ranke1929_talkie_sft_20260928\eval\frozen_eval_84.jsonl` (SHA-256 156eb18d…6277), `eval\JUDGE_PROMPT.md`
- `G:\detritus\ranke_period_qa_scale_20260928\data\history_passages_v1.jsonl` (2,077 passages, SHA-256 803d7139…ba04; latest publication 1906)
- Talkie source: `C:\Users\hvoth\AppData\Local\Temp\talkie-upstream\src\talkie\model.py`, `generate.py`
- Reviews: `review\deepseek_v4pro_round1.md`, `review\deepseek_v4pro_round2.md`

## CHECKED (what exists, so nothing is rebuilt)
- STATUS.md 23:05: "No full Talkie generation has started yet." One Talkie record was ever produced (A100 test, 235.5 s, failed format; fix = assistant prefix `<question>`, never re-tested).
- Base Ranke answers to all 84 frozen prompts exist: `experiments\base84_20260928_172133\eval\blind_output_matrix.jsonl`. Surface stats: 74/84 hit the token ceiling, 72/84 repeat, 29/84 invent a new `Question:` turn.
- v1 used a modern selector (`select_talkie_with_luna.py`). v2 removes it.
- Talkie `forward()` has no key/value cache; each new token recomputes the whole sequence. `_stream_raw` does one prompt at a time. This explains ~4 minutes per record.
- The SFT script `train_ranke_lora.py` is sound (answer-only loss, LoRA r8 q/v, 4-bit base). Reused unchanged.
- Codex's cluster job `6510f2b9ee` / Slurm `6657946`: CANCELLED, 0 s used. Codex heartbeat `continue-ranke-period-qa-experiment`: PAUSED. No Talkie job in the cluster queue (checked 23:24).

## Hard rule (JV)
No modern model creates, edits, selects, filters or ranks any training text. Only the period models do: **Talkie-1930-13B-IT** writes; **Ranke-4B-1929 v2** judges. Modern models only grade saved final answers.

## 1. Construct
Does SFT of Ranke-1929 on period-written, passage-grounded question-and-answer data make Ranke answer historical and political questions better (correct, reasoned, direct, complete, period-fitting), and does more data help more?

## 2. Mechanism
**Stage A — Talkie writes.** Locked prompt, revision `8033675b…`, 6 question forms, v1 sampling (temperature 0.55, top_p 0.90, top_k 40, max 512 new tokens, per-record seed), assistant prefix `<question>`. Two candidates per passage (up to 4,154). A new batched decoder with a key/value cache runs Talkie's exact weights and layer maths; only the cache is new. Every raw completion is saved.

**Stage A protocol change (v2.9, after test V2 failed 0/16 at 00:20).** With only the `<question>` prefix, Talkie wrote a question and then ended its turn on all 16 smoke records (no answer; saved in `runs\v2_20260929_0015\smoke16\`). Fix, same kind as the approved prefix: code inserts the fixed delimiter `</question>\n<answer>` after Talkie's question, and Talkie continues the same assistant turn and writes the answer. If Talkie ends the turn with its own end token, `</answer>` is appended; answers cut by the 512-token limit stay unparsed and are rejected. Prompt, passage, model and sampling are unchanged; Talkie writes every question and answer word. Retest: 16/16 parsed (`smoke16b\`).

**Stage B — Ranke judges Talkie.** First the v1 mechanical rules (tags, 90–220 words, copy overlap, protected eval terms, duplicate questions) plus: reject a question whose word-set overlap (Jaccard) with any of the 84 eval prompts exceeds 0.5. Then Ranke-1929 base computes for each record:
- *grounding gain* = mean per-token log-probability of the answer given (passage + question) minus the same given (question only). High = the answer uses the passage. Near zero or negative = filler or content not in the passage, which is the main route for 1913–1929 facts.
- *fluency* = mean per-token log-probability of the answer given the question only.
**Late-fact word list (deterministic, added v2.4).** Reject any record whose question or answer contains a year 1914–1939 in digits or spelled out ("nineteen fourteen" … "nineteen thirty-nine"), or any of: "Great War", "World War", "League of Nations", "Versailles", "Bolshevik", "Soviet", "Lenin", "Trotsky", "Stalin", "Mussolini", "Fascis", "Weimar", "Armistice", "Brest-Litovsk", "Lusitania", "Somme", "Verdun", "Gallipoli", "Sarajevo", "Wilson's Fourteen", "Locarno", "Kellogg", "Hitler", "Irish Free State", "Balfour Declaration", "Sykes-Picot", "suffrage granted in 1918", "wireless broadcast", "tank warfare", "trench warfare", "poison gas", "U-boat", "Marne", "Ypres", "no man's land", "Western Front", "Eastern Front", "Passchendaele", "Jutland", "Tannenberg", "Caporetto", "Dardanelles campaign", "Zimmermann", "Kerensky", "Petrograd", "Leningrad", "Red Army", "White Army", "Comintern", "Spartacist", "Kapp Putsch", "Ruhr occupation", "Dawes Plan", "Young Plan", "reparations commission", "Treaty of Trianon", "Treaty of Sevres", "Treaty of Lausanne", "Ataturk", "Kemal", "Czechoslovakia", "Yugoslavia", "Kingdom of Serbs", "mandate territory", "Volstead", "Harding", "Coolidge", "Hoover", "Lloyd George ministry", "Ramsay MacDonald", "Labour government", "Easter Rising", "Black and Tans", "Anschluss", "aeroplane bombing", "air raid", "Zeppelin raid", "talking picture". ("radio", "gramophone", "cinema", "motion picture" removed in v2.8: in use before 1914.) The list is fixed now and saved with the run.
Rule, fixed now: keep records with grounding gain > 0 (the passage must help) AND above the 20th percentile, and fluency above the 5th percentile. If the gain > 0 floor leaves fewer than 3,000, more candidates are generated (a third per passage); the floor is not lowered. The 20th percentile is a light cut chosen to keep enough data for a 3,000 dose; the report shows the gain distribution and records near the cut. Ranke outputs numbers only. Context window: Ranke-4B allows 40,960 tokens; passages are at most about 1,500 tokens, so passage + question + answer fits; the code cuts any context above 3,000 tokens from the left and reports the count (expected 0). Acknowledged: the word list cannot be complete; grounding gain and the period_fit evaluation are the backstops.

**Stage C — SFT dose curve.** One seeded order (seed 1929) of accepted records; nested sets of 300 / 1,000 / 3,000 (if fewer than 3,000 are accepted, the top dose = all accepted; if fewer than 1,000, doses become 100 / 300 / all; stated in the report). Three LoRA adapters with the unchanged v1 script and settings (r8, α16, q/v, lr 2e-4, 3 epochs, max length 1,024, micro-batch 1, grad-accum 8, seed 1929).

(The preference/DPO round of v2.0–v2.2 is dropped by JV, 23:35: the Talkie-as-judge risk is too high for one night; SFT-only is simpler to read.)

## 3. Threats to validity
1. *Talkie leaks 1913–1929 facts.* Grounding gain penalizes content the pre-1913 passage does not support; eval scores period_fit separately. Not fully closed; reported.
2. *Gain rewards copying.* Copy-overlap rule first; report gain against copy overlap.
3. *Gain rewards surface word overlap, not truth.* Test V3 below uses corrupted answers on the same topic; plus an inspection page of 40 accepted / 40 rejected records for JV. The inspection checks the filter only; no record is changed or picked by hand or by a modern model.
4. *Gains are format learning (shorter, cleaner answers), not history.* Substance reported separately from directness/fluency; length and completion rates; paired per-prompt comparison.
5. *Train-eval overlap.* Protected terms + question Jaccard rule. Added v2.4: each training answer is compared (word-set Jaccard) with each eval item's prompt + scoring criteria; records above 0.5 are excluded, and the full overlap distribution is reported. Low-overlap and transfer items reported separately.
6. *Different decoding between conditions.* All four conditions, base included (regenerated), use one script, one decoding setting, one machine. Old base matrix kept as a cross-check.
7. *Judge bias.* Replies anonymized and shuffled per item; two modern judge families; agreement reported.
8. *Dose confounded with number of updates.* Stated as a dose curve, not a token-matched estimate.
9. *Power.* 84 prompts give limited power; paired permutation tests and bootstrap intervals, stated plainly.

## 4. Validation tests (pass/fail, run before scaling)
- **V1 result (00:00).** In bf16 the written bars failed: 1/8 exact greedy vs the original uncached code; 10/64 batch-32 vs batch-1; max first-step logit difference 1.44. Diagnosis in float32: 8/8 exact greedy 64 tokens, first-step logit differences 6e-5 to 1.1e-4. So the cache code is exact and the bf16 differences are rounding; the original Talkie code also runs in bf16, so no reorganised bf16 computation reproduces it bit for bit. Speed: 1.35 hours projected for 4,154 records. Decision: keep the cached decoder in bf16; reported.
- **V1 cached decoder.** (a) 8 prompts of varied length (about 200 to 1,500 tokens), batched together with left padding: greedy 64 tokens must equal the original uncached `forward()` on ≥ 7/8; first-step logits max abs difference < 0.5. (b) Added v2.5: 64 real prompts, greedy 128 tokens, batch of 32 vs batch size 1 (no padding at all): ≥ 60/64 identical. (c) Throughput: projected time for 4,154 records ≤ 2 hours on the A100. The logit bar stays at 0.5, not 1e-3: bf16 has about 3 significant digits, and logits of size ~20 differ by ~0.1 between any two correct bf16 implementations (different summation order), so 1e-3 would fail a correct decoder. Greedy-token identity is the stronger practical test. Fallback if (a) or (b) fails: the same cached decoder at batch size 1 on length-sorted prompts is not needed; instead batches of equal-length prompts (no padding, no mask) — a simple variant that removes the only new risky part (padding masks). If that also fails: stop and report.
- **V2 Talkie format.** 16 records; pass if ≥ 12/16 parse into question + answer. All 16 saved and read. Fail → stop; no full run.
- **V3 Ranke grounding judge (deciding test).** Corrupted answers: one middle sentence of a Talkie answer replaced by a sentence from an answer about a different passage. Pass if the intact answer has higher gain in ≥ 80% of 300 pairs. Reported, not deciding: true passage vs random passage (expect ≥ 90%). Fail → stop before training and report; no fallback to a modern filter.
- **V3 bar stays 80% (reply to DeepSeek round 4):** the swap changes one sentence of about eight, about 12% of answer tokens, so a per-token mean moves only a little; 80% is a demanding bar for that signal. The late-fact word list and V3b cover the anachronism risk directly.
- **1913 stays allowed** (reply to DeepSeek round 6): the eval date limit is 31 December 1913, so 1913 is inside the period.
- **Gain floor:** keep records with gain > max(0, P20 of all mechanically accepted records). No arbitrary 0.1 cut; the gain distribution per dose is reported.
- **V3b late-fact injection (informational since v2.7:** the injected sentences contain list words, so the word list already removes such records; a failure here would not show a real gap). Insert one fixed template sentence with a post-1913 fact (e.g. "This arrangement was swept away by the Treaty of Versailles in 1919."; 12 templates, fixed now) into the middle of 300 answers. Pass if the clean answer has higher gain than the injected one in ≥ 90% of pairs. Honest limit: a record-level mean cannot reliably reject one bad sentence among eight by an absolute cut; that job is done by the late-fact word list, and this test checks the direction of the judge.
- **V4 yield pilot.** 200 records through every filter. If the projected accepted count is below 3,000, generate a third candidate per passage (next form in the fixed rotation) after the first two for all passages are done.
- **V5 end-to-end dry run.** 32 records, 1 epoch: must finish with finite loss, save the adapter, reload it and generate a non-empty answer (the training script raises an error otherwise). Fail → stop. Loss decrease is checked on the 300-record run (mean loss of the first 20 updates vs the last 20); if it does not fall, stop before the larger runs.
- **Minimum pool.** If fewer than 100 records are accepted after all filters and third candidates, stop: no dose curve is possible.
- **Time limit.** If Talkie writing is not done 3 hours after the pod starts, it stops; order of work: candidates 1–2 for all passages first, third candidates last. Report states any cut.

## 5. Evaluation (modern models only here)
Four conditions answer all 84 frozen prompts: base, SFT-300, SFT-1000, SFT-3000. Wrapper as in training (`Question: …\nAnswer:`), greedy, per-item `max_new_tokens` from the frozen file; answers cut at a new `Question:` line (recorded).
**Judge calibration (added v2.5).** Before the main grading, each judge grades 20 Talkie training answers twice: clean, and with one fixed late-fact template sentence inserted (V3b templates). Pass if the injected version gets a lower period_fit score in ≥ 16/20. A judge that fails has its period_fit scores flagged and left out of the anachronism rule; if both fail, the report states that the anachronism conclusion is unreliable.
Judges (JV, 23:22): **DeepSeek-V4 Flash** via OpenRouter and **Claude Sonnet** (on-plan subagents, no metered Anthropic API), both with the unchanged `eval\JUDGE_PROMPT.md`, blind, replies shuffled per item.
Report per condition: means with bootstrap 95% intervals, paired permutation tests, per-prompt wins/losses vs base, dose slope, history substance on the 36 history prompts, period fluency on all 84, Russo-Japanese-War/Angell slice, length and completion rates, judge agreement (Spearman correlation of overall scores, weighted kappa per axis, share of items where both judges rank the same condition best), fluency distribution of accepted records, full answer examples.
Decision rule (v1, with v2.8 definition): effect if any trained condition beats base on substance without more anachronism. "Without more anachronism" (v2.10, non-inferiority) = the paired bootstrap 95% interval (10,000 resamples of prompts) of the difference in mean period_fit (condition − base; 0–4 scale; per-prompt means over the judges that passed calibration) lies entirely above −0.25, AND the number of replies with a non-empty anachronism list is not more than base + 3 of 84; slope if SFT-3000 beats SFT-1000 by ≥ 5 points on the 0–100 substance score (= 0.2 on the judges' 0–4 scale; about one step on one of five prompts), in ≥ 4 of 6 reasoning groups, and wins more prompts than it loses.
**Metric definitions (v2.9/v2.10):** substance score = judge `substance` (0–4) × 25, averaged over the judges, per prompt, on the 36 `history_reasoning_36` prompts (0–100 scale). **Effect** also requires a one-sided paired permutation test (10,000 permutations) of trained vs base substance below 0.05 after Holm correction across the three trained conditions. A prompt-level **win** = the condition's per-prompt substance score is strictly higher (ties count for neither). The **six reasoning groups** are the six `category` values of the 36 history prompts in `eval\frozen_eval_84.jsonl`: causal_explanation, comparison, chronology_and_mechanism, actor_motives, argument_and_judgment, pure_transfer (6 prompts each); a group improves if its mean substance is strictly higher.

## 6. Compute and cost
- Cluster queue held every Talkie job PENDING (Priority) for hours today. So: **one RunPod A100 80 GB** (JV 23:26: cheaper; 80 GB holds Talkie at 26.6 GB plus cache, and Ranke 4B). ~$1.2–2.5/hour; expected 5–7 hours → **~$8–18; hard cap $40**.
- Pod deletes itself after 9 hours (on-pod timer), plus a local watchdog. Teardown triple-check: (1) terminate response, (2) API pod list empty, (3) list still empty 5 minutes later.
- DeepSeek-V4-Pro plan review ~$0.10. DeepSeek-V4 Flash judging < $1. Sonnet judging on-plan.

## 7. Saved per-item evidence
One JSONL row per item with inputs, raw output, scores, model revision, seed, hashes; resume support: `talkie_raw.jsonl`, `mechanical.jsonl`, `ranke_grounding.jsonl`, `sft_300/1000/3000.jsonl`, `eval_matrix.jsonl`, `judge_sonnet/*.json`, `judge_deepseek_flash/*.json`. All copied to `G:\detritus\ranke1929_talkie_sft_20260928\runs\v2_<timestamp>\` and hash-checked before teardown.

## 8. What I will NOT do
No modern model sees training text before evaluation. No change to the frozen eval, judge prompt, source corpus, Talkie prompt, Ranke base revision, or v1 SFT settings. No DPO. Nothing called final before JV review.

## Heavy steps
1. OpenRouter: DeepSeek-V4-Pro plan review until VERDICT: SHIP; DeepSeek-V4 Flash eval judging.
2. RunPod: one A100 80 GB pod for Stages A–C and eval answers; then terminate.
3. Training: SFT rehearsal + 3 SFT runs on the pod.
4. Cluster: none.

Approved by JV 2026-09-28 23:22 (`GATE-APPROVE talkie-ranke-mutual-0928`) with the judge changes; DPO dropped by JV 23:35.
