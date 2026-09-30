#!/bin/bash
# Continued pretraining piece (plan v3 option 3). Resumable: rerun the same command to continue.
# Usage: DATA=... HELD=... OUT=... MAXMIN=100 [MAXUPD=15] bash cluster_cpt.sh
set -euo pipefail
V=/scratch/hvoth/ranke1929-talkie-sft/v2
export HF_HOME=/scratch/hvoth/ranke1929-talkie-sft/hf HF_HUB_OFFLINE=1
export PATH=/scratch/hvoth/ranke-event120-20260912/.venv/bin:$PATH
export PYTHONPATH=$V/run/scripts/v2/site:$V/pydeps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $V/run
BASE=$(ls -d $HF_HOME/hub/models--uzh-echist-org--Ranke-4B-1929_v2/snapshots/fb4cede16ed59e1249ebeba2b4e5b9ec2a560642)
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/v3/cpt_train.py --base $BASE --data $DATA --heldout $HELD --out $OUT --max-minutes $MAXMIN ${MAXUPD:+--max-updates $MAXUPD} 2>&1 | grep --line-buffered -v "Loading weights"
