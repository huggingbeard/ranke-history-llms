#!/bin/bash
# 2026-09-30 03:05: best candidate on the cluster, 5 epochs (answers after each) - also tests whether longer training
# helps (JV 22:32 double-descent question). Same recipe as the comparison arms (cluster_arms45.sh).
set -euo pipefail
V=/scratch/hvoth/ranke1929-talkie-sft/v2
export HF_HOME=/scratch/hvoth/ranke1929-talkie-sft/hf HF_HUB_OFFLINE=1
export PATH=/scratch/hvoth/ranke-event120-20260912/.venv/bin:$PATH
export PYTHONPATH=$V/run/scripts/v2/site:$V/pydeps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $V/run
BASE=$V/run/out_v3/cpt_full/final; NAME=cpt_everything_plus_r64_5ep; O=out_night/$NAME; mkdir -p $O
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ ! -f $O/adapter/receipt.json ]; then
  python scripts/v3/train_ranke_lora_v3.py --base $BASE --base-revision cpt_full --data data_night_arms/train_everything_plus.jsonl \
    --out $O/adapter --epochs 5 --lr 1e-4 --lora-r 64 --lora-alpha 128 --targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --micro-tokens 4096 --verify-reload 2>&1 | grep --line-buffered -E '"stage": "(data_ready|model_ready|epoch_saved|complete|reload_check_passed)"|update": (1|1000|5000|10000|15000|20000|25000),|Traceback|Error' | cut -c1-300
fi
test -f $O/adapter/receipt.json
SPEC=""; for E in 1 2 3 4 5; do SPEC="$SPEC --adapter ${NAME}_ep$E=$O/adapter/epoch_$E"; done
python scripts/v3/eval_adapters_v3.py --battery eval/frozen_eval_84.jsonl --base $BASE --base-revision cpt_full $SPEC \
  --out $O/eval_answers.jsonl 2>&1 | grep --line-buffered -E 'eval_done|Traceback|Error'
echo "ARM_DONE $NAME $(date -u)"
