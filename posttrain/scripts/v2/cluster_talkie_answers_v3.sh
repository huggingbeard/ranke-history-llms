#!/bin/bash
# Night 2026-09-29 job 2: Talkie-13B answers 9,288 stand-alone English questions from the 47 newly found books.
set -euo pipefail
R=/scratch/hvoth/ranke1929-talkie-sft/v2
export PATH=/scratch/hvoth/ranke-event120-20260912/.venv/bin:$PATH
export PYTHONPATH=$R/run/scripts/v2/site:$R/run/talkie-src:$R/pydeps HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $R/run; mkdir -p logs out_night
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/v2/talkie_answer_questions.py --questions data_night_v3/talkie_questions_v3.jsonl \
  --out out_night/talkie_direct_answers_v3.jsonl --cache-dir /scratch/hvoth/ranke1929-talkie-sft/hf \
  --batch-size 32 --max-minutes ${MAXMIN:-110} 2>&1 | grep --line-buffered -v '^checkpoint' | tee -a logs/talkie_direct_v3.log
echo "ROWS $(wc -l < out_night/talkie_direct_answers_v3.jsonl)"
