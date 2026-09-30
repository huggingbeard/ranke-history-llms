# Talkie x Ranke-1929 — next round (plan v3, DRAFT for approval)

Version: v3.0-draft · Date: 2026-09-29 14:25 Europe/Zurich · Plan ID: `ranke-nextround-0929`
Files used: `G:\detritus\ranke1929_talkie_sft_20260928\LOG.md`, `PLAN_v2_mutual_judging.md`,
`G:\detritus\ranke_period_qa_run1_20260927\data\luna_accepted_literal.jsonl` (4,023 native Q+A pairs),
`G:\detritus\period_qa_sources_20260927\sources\`, `D:\UZHechist Dropbox\Joachim Voth\history-wvs\posttrain\2026-09-27_original_proposal.txt`.

## CHECKED
- Round 2 (today): any SFT lifts history substance 6.9 -> 26-31/100 (Sonnet); no dose slope 300 -> 2,213. Same pattern as the earlier Typewriter-1913 run (24.4 -> 48.8 -> 52.1).
- Native period Q+A pool exists (4,023 literal pairs, 11 books; mostly law; 855 Mangnall history candidates).

## JV decisions (2026-09-29 14:09)
Option 8 first (Talkie ceiling). Then options 2 (many more sources), 3 (continued pretraining on period text), 4 (bigger LoRA), 5 (longer training), 6 (rule-based best-of-8 / DPO pairs by rules). Option 1 reframed: books already in Q+A form; a modern model may SELECT pairs, never write or edit.

## Heavy steps this approval covers (cluster only; RunPod is NOT authorized)
1. Cluster (S3IT, via router): Talkie ceiling eval; 5-way re-grade inputs; SFT arms on native-Q+A data and Talkie data (LoRA r8 vs r64 all modules; 3 vs 8 epochs); rule-based best-of-8 sampling; continued-pretraining pilot on period history text (small, <= 4 card-hours).
2. OpenRouter: DeepSeek-V4 Flash grading, hard cap $5 total. Sonnet grading on-plan subagents.
3. Cap: cluster <= 5 CHF for this round. Every job sized from measured speed, checked at 10 minutes (card use), checkpointed.

Full design with pass/fail tests follows before the training arms run; the ceiling eval and re-grade run first.
