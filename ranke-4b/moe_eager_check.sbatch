#!/bin/bash
#SBATCH --job-name=moe_eager
#SBATCH --account=yanagizawa.econ.uzh
#SBATCH --output=/scratch/ukrenk/moe_smoke_20261020/logs/%x_%j.log
# Does the NaN at step 26 go away with the plain (eager) expert routine? (2026-10-20), 2 cards. Submit:
#   /scratch/ukrenk/bin/ssub.sh --type H200 --cards 2 -- --time=00:20:00 --cpus-per-task=16 --mem=192G moe_eager_check.sbatch
# 27 steps in the normal data order with the eager expert routine (a loop over experts, no fused kernel); records
# any micro-batch whose gradients turn non-finite (nan_rank*.jsonl). Eager may be ~4x slower: limit 900 s.
set -uo pipefail
B=/scratch/ukrenk/moe_smoke_20261020; V=/scratch/ukrenk/ranke/vllm-venv
export HF_HOME=$B/hf_home OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $B/code_pl
R=$B/runs/${SLURM_JOB_ID}_eager
( while true; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | sed "s/^/$(date +%s),/" >> ${R}.gpu.csv; sleep 10; done ) &
GPUMON=$!
echo "=== $(date +%T) 27 steps, plain eager expert routine"
timeout 900 $V/bin/torchrun --nproc_per_node=2 moe_smoke_train.py --out ${R}_B_locate --steps 27 --val-every 1000 --val-seqs 4 --balance-loss=per-layer --nan-debug --experts-impl eager
kill $GPUMON
echo "=== $(date +%T) NANLOCATE_DONE"
