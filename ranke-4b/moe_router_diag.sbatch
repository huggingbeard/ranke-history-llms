#!/bin/bash
#SBATCH --job-name=moe_diag
#SBATCH --account=yanagizawa.econ.uzh
#SBATCH --output=/scratch/ukrenk/moe_smoke_20261020/logs/%x_%j.log
# Router diagnostic (2026-10-20), 1 card, about 3 minutes. Submit through the partition picker:
#   /scratch/ukrenk/bin/ssub.sh --type H200 --cards 1 -- --time=00:12:00 --cpus-per-task=8 --mem=96G moe_router_diag.sbatch
set -uo pipefail
B=/scratch/ukrenk/moe_smoke_20261020; V=/scratch/ukrenk/ranke/vllm-venv
export HF_HOME=$B/hf_home OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $B/code
mkdir -p $B/runs/diag_${SLURM_JOB_ID}
echo "=== $(date +%T) router diag start"
timeout 540 $V/bin/torchrun --nproc_per_node=1 moe_router_diag.py --out $B/runs/diag_${SLURM_JOB_ID}/router_diag.jsonl --layers 4
echo "=== $(date +%T) DIAG_DONE rc=$?"
