#!/bin/bash
#SBATCH --job-name=moe_padfix
#SBATCH --account=yanagizawa.econ.uzh
#SBATCH --output=/scratch/ukrenk/moe_smoke_20261020/logs/%x_%j.log
# Does the NaN at step 26 go away once no token is treated as padding? (2026-10-20), 2 cards. Submit:
#   /scratch/ukrenk/bin/ssub.sh --type H200 --cards 2 -- --time=00:12:00 --cpus-per-task=16 --mem=192G moe_padfix_check.sbatch
# 32 steps in the normal data order, fused expert kernel, pad_token_id=None; records
# any micro-batch whose gradients turn non-finite (nan_rank*.jsonl). About 3 minutes.
set -uo pipefail
B=/scratch/ukrenk/moe_smoke_20261020; V=/scratch/ukrenk/ranke/vllm-venv
export HF_HOME=$B/hf_home OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $B/code_pl
R=$B/runs/${SLURM_JOB_ID}_padfix
( while true; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | sed "s/^/$(date +%s),/" >> ${R}.gpu.csv; sleep 10; done ) &
GPUMON=$!
echo "=== $(date +%T) 32 steps, no padding token (pad id fix)"
timeout 900 $V/bin/torchrun --nproc_per_node=2 moe_smoke_train.py --out ${R}_B_locate --steps 32 --val-every 1000 --val-seqs 4 --balance-loss=per-layer --nan-debug 
kill $GPUMON
echo "=== $(date +%T) NANLOCATE_DONE"
