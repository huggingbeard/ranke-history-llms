#!/bin/bash
#SBATCH --job-name=moe_warmup
#SBATCH --account=yanagizawa.econ.uzh
#SBATCH --output=/scratch/ukrenk/moe_smoke_20261020/logs/%x_%j.log
# OLMoE balance test with a longer warm-up (JV approved 2026-10-20 18:05). 2 H200 cards, about 25 minutes. Submit:
#   /scratch/ukrenk/bin/ssub.sh --type H200 --cards 2 -- --time=00:40:00 --cpus-per-task=16 --mem=192G moe_warmup_test.sbatch
# Differences from the final smoke run (job 7379027): 300 steps instead of 100; warm-up 200 steps instead of 20;
# bigger snapshot data_v3 (286 books, 42.2M tokens; same 14 validation books); training sequences in a fixed random
# order (seed 0) instead of book order; TensorBoard files in <run>/tb. Same model, per-layer balance loss, no pad token.
# Pass marks (set before the run): no NaN; loss falls; at the last step every layer has its busiest expert <= 3x and
# its least busy expert >= 0.1x fair share.
# Time limit: measured 4.2 s/step x 300 + 1 min load + validation ~ 23 min; x1.5 -> 35 min.
set -uo pipefail
B=/scratch/ukrenk/moe_smoke_20261020; V=/scratch/ukrenk/ranke/vllm-venv
export HF_HOME=$B/hf_home OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/scratch/ukrenk/pylibs/tensorboard   # TensorBoard 2.21.0 + absl-py, kept outside the shared setup
cd $B/code_pl
R=$B/runs/${SLURM_JOB_ID}_warmup
( while true; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | sed "s/^/$(date +%s),/" >> ${R}.gpu.csv; sleep 10; done ) &
GPUMON=$!
echo "=== $(date +%T) 300 steps, warm-up 200, shuffled, TensorBoard"
timeout 2100 $V/bin/torchrun --nproc_per_node=2 moe_smoke_train.py --out ${R}_A --data $B/data_v3 --steps 300 --warmup 200 \
  --shuffle-seed 0 --balance-loss=per-layer --tb --val-every 25 --balance-every 10
kill $GPUMON
echo "=== $(date +%T) WARMUP_DONE"
