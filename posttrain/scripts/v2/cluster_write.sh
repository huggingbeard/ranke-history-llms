#!/bin/bash
# Cluster job (plan v2.11): Talkie writes all records (resume), then Ranke judges, SFT x3, eval answers.
set -euo pipefail
R=/scratch/hvoth/ranke1929-talkie-sft/v2
export RUN_DIR=$R/run HF_HOME=/scratch/hvoth/ranke1929-talkie-sft/hf HF_HUB_OFFLINE=1
export PATH=/scratch/hvoth/ranke-event120-20260912/.venv/bin:$PATH
export PYTHONPATH=$R/run/talkie-src:$R/pydeps
cd $RUN_DIR; mkdir -p logs
nvidia-smi --query-gpu=name,memory.total --format=csv
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
BS=32; if [ "$MEM" -lt 70000 ]; then BS=12; fi
echo "batch size $BS"
python -c "import talkie, peft, bitsandbytes; print(talkie.__file__, peft.__version__)"
echo "== Talkie writing $(date -u)"
python scripts/v2/gen_talkie_v2.py generate --input data/history_passages_v1.jsonl --output out/talkie_raw.jsonl \
  --variants 0,1,2 --batch-size $BS --max-hours ${MAX_HOURS:-1.6} --cache-dir /scratch/hvoth/ranke1929-talkie-sft/hf 2>&1 | grep -v '^checkpoint' | tee logs/generate_full_cluster.log | grep -E 'stage": "(jobs|generate_complete|time_limit_stop)|Error|Traceback' || true
wc -l out/talkie_raw.jsonl
grep -q generate_complete logs/generate_full_cluster.log && echo WRITING_DONE || echo WRITING_PARTIAL_RESUBMIT
