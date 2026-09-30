#!/bin/bash
# Night re-plan 2026-09-30 01:50: comparison arms 4 and 5 moved from the 4090 to the cluster (4090 time too short).
# Same recipe as the local arms (LoRA r64 all modules, lr 1e-4, NF4, batched groups of 8, 3 epochs, answers at 1-3),
# on the continued-pretraining model (out_v3/cpt_full/final). Runs arm 5 then arm 4; each skips if already complete.
set -euo pipefail
V=/scratch/hvoth/ranke1929-talkie-sft/v2
export HF_HOME=/scratch/hvoth/ranke1929-talkie-sft/hf HF_HUB_OFFLINE=1
export PATH=/scratch/hvoth/ranke-event120-20260912/.venv/bin:$PATH
export PYTHONPATH=$V/run/scripts/v2/site:$V/pydeps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $V/run
BASE=$V/run/out_v3/cpt_full/final
nvidia-smi --query-gpu=name,memory.total --format=csv
for ARM in cpt_no_encyclopedia_r64:data_night_arms/train_no_encyclopedia.jsonl cpt_everything_r64:data_night_arms/train_everything.jsonl; do
  NAME=${ARM%%:*}; DATA=${ARM#*:}; O=out_night/$NAME; mkdir -p $O
  if [ -n "${ONLY:-}" ] && [ "$NAME" != "$ONLY" ]; then continue; fi
  echo "== train $NAME $(date -u)"
  if [ ! -f $O/adapter/receipt.json ]; then
    python scripts/v3/train_ranke_lora_v3.py --base $BASE --base-revision cpt_full --data $DATA --out $O/adapter \
      --epochs 3 --lr 1e-4 --lora-r 64 --lora-alpha 128 --targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
      --micro-tokens 4096 --verify-reload 2>&1 | grep --line-buffered -E '"stage": "(data_ready|model_ready|epoch_saved|complete|reload_check_passed)"|update": (1|100|500|1000|2000|3000|4000|5000|6000),|Traceback|Error' | cut -c1-300
  fi
  test -f $O/adapter/receipt.json
  echo "== eval $NAME $(date -u)"
  python scripts/v3/eval_adapters_v3.py --battery eval/frozen_eval_84.jsonl --base $BASE --base-revision cpt_full \
    --adapter ${NAME}_ep1=$O/adapter/epoch_1 --adapter ${NAME}_ep2=$O/adapter/epoch_2 --adapter ${NAME}_ep3=$O/adapter/epoch_3 \
    --out $O/eval_answers.jsonl 2>&1 | grep --line-buffered -E 'eval_done|Traceback|Error'
  echo "ARM_DONE $NAME $(date -u)"
done
