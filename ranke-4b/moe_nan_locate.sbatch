#!/bin/bash
#SBATCH --job-name=moe_nanlocate
#SBATCH --account=yanagizawa.econ.uzh
#SBATCH --output=/scratch/ukrenk/moe_smoke_20261020/logs/%x_%j.log
# Where does the NaN gradient at step 26 come from? (2026-10-20), 2 cards, about 5 minutes. Submit:
#   /scratch/ukrenk/bin/ssub.sh --type H200 --cards 2 -- --time=00:15:00 --cpus-per-task=16 --mem=192G moe_nan_locate.sbatch
# A: fresh model trained first on the step-26 sequences (--data-offset 26). B: normal order to step 26, recording the
# first micro-batch and the weights whose gradients turn non-finite (nan_rank*.jsonl).
set -uo pipefail
B=/scratch/ukrenk/moe_smoke_20261020; V=/scratch/ukrenk/ranke/vllm-venv
export HF_HOME=$B/hf_home OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $B/code_pl
R=$B/runs/${SLURM_JOB_ID}_nanloc
( while true; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | sed "s/^/$(date +%s),/" >> ${R}.gpu.csv; sleep 10; done ) &
GPUMON=$!
echo "=== $(date +%T) A: fresh model on step-26 data"
timeout 300 $V/bin/torchrun --nproc_per_node=2 moe_smoke_train.py --out ${R}_A_offset26 --steps 3 --data-offset 26 --val-every 1000 --val-seqs 4 --balance-loss=per-layer --nan-debug
echo "=== $(date +%T) B: normal order, locate NaN"
timeout 420 $V/bin/torchrun --nproc_per_node=2 moe_smoke_train.py --out ${R}_B_locate --steps 27 --val-every 1000 --val-seqs 4 --balance-loss=per-layer --nan-debug
kill $GPUMON
echo "=== $(date +%T) NANLOCATE_DONE"
