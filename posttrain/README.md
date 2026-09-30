# Post-training Ranke-4B-1929 with period sources and Talkie-13B answers

Version 1 · 2026-09-30 · Code and notes only. Outputs and weights live elsewhere (see below).

**Question.** Can a time-locked 4B model (Ranke-4B-1929) learn to answer history questions well
from period material only: question-answer pairs printed in pre-1914 books, and answers written by
Talkie-13B (`talkie-lm/talkie-1930-13b-it`), a larger model trained only on pre-1931 text?
No modern model writes or edits training text. Modern models (Sonnet, DeepSeek-V4 Flash) only grade.

## Where things are

| What | Where |
|---|---|
| Code, plans, trail | this folder (`scripts/`, `eval/`, `LOG.md`, `STATUS.md`, `PLAN*.md`, `review/`) |
| Outputs: training sets, answers, grades, analysis | `D:\UZHechist Dropbox\Joachim Voth\history-wvs\posttrain\talkie_sft_2026-09-28\` |
| Working copy incl. LoRA adapters + continued-pretraining model (~16 GB) | `G:\detritus\ranke1929_talkie_sft_20260928\` |
| Cluster copy | `/scratch/hvoth/ranke1929-talkie-sft/v2/run/` |
| Earlier 27-Sep copying / pilot work | `D:\UZHechist Dropbox\Joachim Voth\history-wvs\posttrain\` (dated folders) |

`LOG.md` is the authoritative trail (append-only). `eval/frozen_eval_84.jsonl` is the frozen test:
36 history-reasoning prompts + 48 period chat prompts; `eval/JUDGE_PROMPT.md` is the grading rubric.

## State on 2026-09-30 (preliminary, Sonnet grades)

History content score 0-100 on the 36 history prompts: untrained Ranke 6.9; book Q+A pairs 27.8;
Talkie answers only 36.8 (best); Talkie-13B itself 72.2. Talkie answers are the only ingredient that
helps; continued pretraining on period text, encyclopaedia pairs, 47 more books and 5 vs 3 passes do not.
Caveat: every Ranke model so far was trained and answered on a 4-bit (NF4) copy of the base; this was
never approved and is dropped from the next round (`PLAN_v4_close_the_gap_DRAFT.md`).

## Script map

- `scripts/` — round 1 (Talkie passage records, 300/1,000/3,000 nested SFT, blind grading).
- `scripts/v2/` — round 2 (cluster generation with Talkie, Ranke-as-judge filters, DeepSeek grader).
- `scripts/v3/` — night 29-30 Sep (book Q+A sets, Talkie direct answers, continued pretraining,
  LoRA r64 arms, night orchestrator, night grading + analysis, grader agreement).
- `scripts/v4/` — round 4 set builder (Talkie dose, JV's long-book arm).
