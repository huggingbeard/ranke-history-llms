#!/bin/bash
#SBATCH --job-name=moe_smoke
#SBATCH --account=yanagizawa.econ.uzh
#SBATCH --partition=standard
#SBATCH --output=/scratch/ukrenk/moe_smoke_20261020/logs/%x_%j.log
# OLMoE smoke test (2026-10-20). MODE=mini: 1 card, 4-layer model, 6 steps, save at 3, resume 3->6 (code paths only).
# MODE=full: 2 cards, full 16-layer OLMoE-1B-7B shape, 100 steps saving at 50 (run A), then resume from 50 -> 100
# (run B); B's losses must match A's steps 50-99. Submit:
#   sbatch --gres=gpu:H200:1 --cpus-per-task=8  --mem=96G  --time=00:30:00 --export=ALL,MODE=mini moe_smoke.sbatch
#   sbatch --gres=gpu:H200:2 --cpus-per-task=16 --mem=192G --time=<from mini>  --export=ALL,MODE=full moe_smoke.sbatch
# Per-layer balance loss (JV approved 2026-10-20 17:16): --export=ALL,MODE=full,CODE=$B/code_pl,EXTRA=--balance-loss=per-layer
set -uo pipefail
B=/scratch/ukrenk/moe_smoke_20261020
V=/scratch/ukrenk/ranke/vllm-venv
export HF_HOME=$B/hf_home OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CODE=${CODE:-$B/code}; EXTRA=${EXTRA:-}   # CODE = folder with the trainer; EXTRA = added trainer options
cd $CODE
N=$(nvidia-smi -L | wc -l); nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
R=$B/runs/${SLURM_JOB_ID}_${MODE}
( while true; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | sed "s/^/$(date +%s),/" >> $R.gpu.csv; sleep 10; done ) &
GPUMON=$!
T() { echo "=== $(date +%T) $*"; }
if [ "$MODE" = mini ]; then
  T "mini A: steps 0-5, save at 3"
  timeout 900 $V/bin/torchrun --nproc_per_node=$N moe_smoke_train.py --out ${R}_A --layers 4 --global-bs 4 --micro-bs 2 --steps 6 --save-at 3 --warmup 2 --balance-every 1 --val-seqs 4
  T "mini B: resume from step 3"
  timeout 900 $V/bin/torchrun --nproc_per_node=$N moe_smoke_train.py --out ${R}_B --layers 4 --global-bs 4 --micro-bs 2 --steps 6 --resume ${R}_A/ckpt_step3 --warmup 2 --balance-every 1 --val-seqs 4
else
  T "full A: 100 steps, save at 50"
  timeout ${TA:-3000} $V/bin/torchrun --nproc_per_node=$N moe_smoke_train.py --out ${R}_A --steps 100 --save-at 50 $EXTRA
  T "full B: resume from step 50"
  timeout ${TB:-1800} $V/bin/torchrun --nproc_per_node=$N moe_smoke_train.py --out ${R}_B --steps 100 --resume ${R}_A/ckpt_step50 $EXTRA
fi
kill $GPUMON
T "SMOKE_DONE $MODE"
