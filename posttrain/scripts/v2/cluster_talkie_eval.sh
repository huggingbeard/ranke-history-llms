#!/bin/bash
# Ceiling check: Talkie-13B answers the 84 frozen eval prompts (JV 2026-09-29, priority 1).
set -euo pipefail
R=/scratch/hvoth/ranke1929-talkie-sft/v2
export PATH=/scratch/hvoth/ranke-event120-20260912/.venv/bin:$PATH
export PYTHONPATH=$R/run/scripts/v2/site:$R/run/talkie-src:$R/pydeps HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $R/run; mkdir -p logs out/eval_talkie
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/v2/eval_talkie.py --battery eval/frozen_eval_84.jsonl --out out/eval_talkie/talkie13b_answers.jsonl \
  --cache-dir /scratch/hvoth/ranke1929-talkie-sft/hf 2>&1 | grep --line-buffered -v '^checkpoint' | tee logs/eval_talkie.log
test "$(wc -l < out/eval_talkie/talkie13b_answers.jsonl)" -eq 84 && echo TALKIE_EVAL_DONE
