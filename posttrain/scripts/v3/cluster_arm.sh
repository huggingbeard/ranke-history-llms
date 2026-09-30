#!/bin/bash
# Plan v3 arm: train one LoRA adapter, then answer the 84 eval prompts at chosen epochs.
# Usage: ARM=... DATA=... R=8 ALPHA=16 TARGETS=q_proj,v_proj EPOCHS=3 LR=2e-4 EVAL_EPOCHS="3" bash cluster_arm.sh
set -euo pipefail
V=/scratch/hvoth/ranke1929-talkie-sft/v2
export HF_HOME=/scratch/hvoth/ranke1929-talkie-sft/hf HF_HUB_OFFLINE=1
export PATH=/scratch/hvoth/ranke-event120-20260912/.venv/bin:$PATH
export PYTHONPATH=$V/run/scripts/v2/site:$V/pydeps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $V/run
BASE=$(ls -d $HF_HOME/hub/models--uzh-echist-org--Ranke-4B-1929_v2/snapshots/fb4cede16ed59e1249ebeba2b4e5b9ec2a560642)
O=out_v3/$ARM; mkdir -p $O logs
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "== train $ARM $(date -u)"
if [ ! -f $O/adapter/receipt.json ]; then
  python scripts/v3/train_ranke_lora_v3.py --base $BASE --data $DATA --out $O/adapter --epochs $EPOCHS --lr $LR \
    --lora-r $R --lora-alpha $ALPHA --targets $TARGETS --verify-reload 2>&1 | grep --line-buffered -E '"stage": "(data_ready|model_ready|epoch_saved|complete|reload_check_passed)"|update": (1|100|500|1000|2000),|Traceback|Error' | cut -c1-300
fi
test -f $O/adapter/receipt.json
echo "== eval $ARM $(date -u)"
SPEC=""
for E in $EVAL_EPOCHS; do SPEC="$SPEC --adapter ${ARM}_ep$E=$O/adapter/epoch_$E"; done
python scripts/v3/eval_adapters_v3.py --battery eval/frozen_eval_84.jsonl --base $BASE $SPEC --out $O/eval_answers.jsonl 2>&1 | grep --line-buffered -E 'eval_done|Traceback|Error'
echo "ARM_DONE $ARM $(date -u)"
