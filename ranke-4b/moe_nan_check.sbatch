#!/bin/bash
#SBATCH --job-name=moe_nancheck
#SBATCH --account=yanagizawa.econ.uzh
#SBATCH --output=/scratch/ukrenk/moe_smoke_20261020/logs/%x_%j.log
# Two checks (2026-10-20), 2 cards, about 5 minutes. Submit through the partition picker:
#   /scratch/ukrenk/bin/ssub.sh --type H200 --cards 2 -- --time=00:15:00 --cpus-per-task=16 --mem=192G moe_nan_check.sbatch
# 1. moe_grad_check.py on 1 card: does the balance penalty reach the router weights?
# 2. trainer (per-layer penalty), 32 steps, NO validation before the end: does the NaN at step 26 still appear?
set -uo pipefail
B=/scratch/ukrenk/moe_smoke_20261020; V=/scratch/ukrenk/ranke/vllm-venv
export HF_HOME=$B/hf_home OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $B/code_pl
R=$B/runs/${SLURM_JOB_ID}_nancheck; mkdir -p $R
( while true; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | sed "s/^/$(date +%s),/" >> ${R}.gpu.csv; sleep 10; done ) &
GPUMON=$!
echo "=== $(date +%T) grad check"
CUDA_VISIBLE_DEVICES=0 timeout 300 $V/bin/torchrun --nproc_per_node=1 --master_port=29511 moe_grad_check.py --out $R/grad_check.jsonl
echo "=== $(date +%T) 32 steps without validation"
timeout 420 $V/bin/torchrun --nproc_per_node=2 moe_smoke_train.py --out ${R}_noval --steps 32 --val-every 1000 --val-seqs 4 --balance-loss=per-layer
kill $GPUMON
echo "=== $(date +%T) NANCHECK_DONE"
