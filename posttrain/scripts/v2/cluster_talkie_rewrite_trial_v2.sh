#!/bin/bash
# Rewrite trial v2 2026-09-30: complete EB11 articles, 3 styles (summary, catechism of ten, causes), 150 rewrites.
# Reuses talkie_answer_questions.py (the "question" field holds instruction + passage), greedy, 512 new tokens max.
set -euo pipefail
R=/scratch/hvoth/ranke1929-talkie-sft/v2
export PATH=/scratch/hvoth/ranke-event120-20260912/.venv/bin:$PATH
export PYTHONPATH=$R/run/scripts/v2/site:$R/run/talkie-src:$R/pydeps HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $R/run; mkdir -p logs out_night
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/v2/talkie_answer_questions.py --questions data_rewrite_v2/rewrite_trial_prompts.jsonl \
  --out out_night/rewrite_trial_v2_answers.jsonl --cache-dir /scratch/hvoth/ranke1929-talkie-sft/hf \
  --batch-size 16 --max-new-tokens 512 --max-minutes 40 2>&1 | grep --line-buffered -v '^checkpoint' | tee -a logs/rewrite_trial_v2.log
echo "ROWS $(wc -l < out_night/rewrite_trial_v2_answers.jsonl)"
