#!/bin/bash
# Cluster Ranke job (plan v2.11): merge Talkie files, Ranke scoring, SFT doses, eval answers.
# Restartable: finished steps are skipped.
set -euo pipefail
R=/scratch/hvoth/ranke1929-talkie-sft/v2
export RUN_DIR=$R/run HF_HOME=/scratch/hvoth/ranke1929-talkie-sft/hf HF_HUB_OFFLINE=1
export PATH=/scratch/hvoth/ranke-event120-20260912/.venv/bin:$PATH
export PYTHONPATH=$R/run/scripts/v2/site:$R/run/talkie-src:$R/pydeps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $RUN_DIR; mkdir -p logs
nvidia-smi --query-gpu=name,memory.total --format=csv
[ -f out/talkie_raw_all.jsonl ] || python - <<'PY'
import json, hashlib
rows, seen = [], set()
for f in ["out/talkie_raw.jsonl", "out/talkie_raw_variant1.jsonl", "out/talkie_raw_variant2.jsonl"]:
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        assert r["candidate_id"] not in seen, r["candidate_id"]
        seen.add(r["candidate_id"]); rows.append(l)
assert len(rows) == 6231, len(rows)
open("out/talkie_raw_all.jsonl", "w", encoding="utf-8").writelines(rows)
print("merged", len(rows), hashlib.sha256(open("out/talkie_raw_all.jsonl","rb").read()).hexdigest())
PY
python -c "import torch; print('cudnn sdpa enabled:', torch.backends.cuda.cudnn_sdp_enabled())"
JUDGE_FAILED_PROCEED=1 bash scripts/v2/stage_bc.sh 2>&1 | tee -a logs/stage_bc_cluster.log | grep --line-buffered -E '^==|accepted|loss first|V3|skip|stopping|STAGE_BC_DONE|Error|Traceback|STOP|"stage": "(filter|select|score)"' || true
test -f out/STAGE_BC_DONE && echo ALL_DONE
