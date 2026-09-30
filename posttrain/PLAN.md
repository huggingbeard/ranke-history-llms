# Talkie to Ranke-1929 supervised training experiment

Date: 28 September 2026

## Question

Does more period-grounded question-and-answer training improve how Ranke-4B-1929 v2 answers historical and political questions? We need both an effect over the base model and a positive slope as the training dose rises.

## Locked roles

- **Writer:** `talkie-lm/talkie-1930-13b-it`, revision `8033675be6360ae0127fa75f941c12d52064f1dc`.
- **Student:** `uzh-echist-org/Ranke-4B-1929_v2`, revision `fb4cede16ed59e1249ebeba2b4e5b9ec2a560642`.
- **Sources:** the fixed 2,077-passage, eight-book corpus in `G:\detritus\ranke_period_qa_scale_20260928\data\history_passages_v1.jsonl`, SHA-256 `803d71395ad53355f61e04e249def5a1e2798c4e48d83a05d9af49d4a909ba04`. Every source was published by 1913.
- **Modern selector:** may accept or reject a complete Talkie record. It may not add, remove, or rewrite any training word.
- **Evaluation:** 84 prompts frozen before any Ranke-1929 SFT at `G:\detritus\ranke1929_talkie_sft_20260928\eval\frozen_eval_84.jsonl`, SHA-256 `156eb18d5796a2db123b40ad641b30cb74c4763964077f2c1bca5e8c75796277`. This is an ID-namespaced concatenation of the unchanged 36-question history-reasoning battery and the unchanged earlier 48-question period-chat battery. The latter contains the Russo-Japanese War, Angell, legal, conversation, and period-writing tests.

Talkie has a 1930 cutoff, later than the source boundary. The admissibility rule therefore concerns the generated record, not Talkie's inherited weights: every historical claim in the answer must be supported by the supplied pre-1913 passage. Unsupported claims cause rejection. This prevents new 1913-1929 facts from entering the adapter.

## Data construction

Each source passage receives two different jobs, drawn from these forms:

1. cause and consequence;
2. comparison or contrast;
3. chronology and mechanism;
4. motive and constraint;
5. argument, objection, and judgment;
6. institutional explanation.

Talkie sees the source metadata, the passage, one requested form, and a strict output schema. It writes one question and one answer. The answer should be 90-220 words, make a claim, give a connected explanation, and end cleanly. The question must be answerable from the passage but must not quote it mechanically.

The generator saves the prompt, raw output, parsed record, model revision, source identifier, and seed. It never repairs malformed output with a modern model.

Mechanical rejection removes records that fail the schema, exceed length limits, copy too much of the passage, contain protected evaluation terms, or duplicate another question. The modern selector then returns all labels in one call: grounded, answerable, historically coherent, useful reasoning, clean prose, no anachronism, and accept. Its output is a decision record only.

## Nested training doses

Freeze one seeded ordering of accepted records. Create nested sets from that ordering:

- QA-300: first 300 records;
- QA-1000: first 1,000 records;
- QA-3000: first 3,000 records.

The three adapters start from the same Ranke-1929 base revision. They use answer-only loss, the same prompt wrapper, LoRA targets, learning rate, epoch count, maximum length, seed, and decoding settings. The larger doses therefore use more examples and more optimizer updates. The result is a training-dose curve, not a token-matched data-only estimate.

Initial settings: LoRA rank 8, alpha 16, dropout 0, `q_proj` and `v_proj`, AdamW, learning rate 2e-4, maximum length 1,024, micro-batch one, gradient accumulation eight, three epochs, seed 1929. A small training rehearsal must pass before the three full runs.

## Evaluation

Generate complete answers for the same 84 sealed questions from:

- base Ranke-1929;
- QA-300;
- QA-1000;
- QA-3000.

Use identical decoding settings. Preserve every raw answer before grading. Grade the saved answers after generation on separate dimensions:

- historical substance and reasoning;
- directness and completion;
- period-appropriate fluency;
- unsupported claims or anachronism.

Report means, paired prompt-level changes, bootstrap intervals, group results, output length, end-token rate, and complete examples. The Russo-Japanese War and Angell questions are reported as a protected slice, not as the whole result. Training-absent topics and low-overlap questions test transfer.

## Decision rule

The recipe shows a useful effect if at least one trained condition improves historical substance over the base without increasing unsupported claims. It shows a useful slope if QA-3000 improves over QA-1000 by at least five points out of 100, improves at least four of six reasoning groups, and wins more prompts than it loses in paired comparison.

If scores improve mainly through shorter, cleaner answers, the result is fluency adaptation. If gains occur only near training passages, the result is source-specific learning. If QA-3000 does not clear the slope rule, stop scaling this recipe and change the data form.

## Run order

1. Generate 16 smoke records with Talkie and inspect them against the locked rules.
2. Generate roughly 4,154 candidates, two per passage.
3. Apply mechanical gates and one-pass modern selection.
4. Freeze and hash the nested 300/1,000/3,000 sets.
5. Generate and save the base-model evaluation outputs.
6. Rehearse training on a small sample.
7. Train QA-300, QA-1000, and QA-3000.
8. Generate all trained-model outputs and grade the saved answer set.
9. Publish the slope table and complete prompt-level evidence.
