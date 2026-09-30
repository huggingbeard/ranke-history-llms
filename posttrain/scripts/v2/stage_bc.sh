#!/bin/bash
# Stages B + C + eval answers (plan v2.4). Stops on any failed deciding test.
set -euo pipefail
cd ${RUN_DIR:-/workspace/run}
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1
R=$(ls -d $HF_HOME/hub/models--uzh-echist-org--Ranke-4B-1929_v2/snapshots/fb4cede16ed59e1249ebeba2b4e5b9ec2a560642)
O=out
S=scripts/v2
echo "== filter $(date -u)"
python $S/ranke_judge.py filter --out-dir $O/stageB --raw $O/talkie_raw_all.jsonl --sources data/history_passages_v1.jsonl --eval eval/frozen_eval_84.jsonl
echo "== V3 validate $(date -u)"
[ -f $O/stageB/V3_validation.json ] || python $S/ranke_judge.py validate --out-dir $O/stageB --sources data/history_passages_v1.jsonl --base $R --n 300
python - <<'EOF'
import json,sys
v=json.load(open("out/stageB/V3_validation.json"))
print({k:v[k] for k in v if k!="rows"})
import os
if not v["passed"]:
    if os.environ.get("JUDGE_FAILED_PROCEED") == "1":
        print("V3 FAILED - JV decided 2026-09-29 13:06: train on all mechanically accepted records; Ranke scores kept for analysis only")
    else:
        open("out/STOPPED_V3_FAILED","w").write("V3/V3b failed"); sys.exit("V3 FAILED - stop before training")
EOF
echo "== score $(date -u)"
python $S/ranke_judge.py score --out-dir $O/stageB --sources data/history_passages_v1.jsonl --base $R
python $S/ranke_judge.py select --out-dir $O/stageB --sources data/history_passages_v1.jsonl ${JUDGE_FAILED_PROCEED:+--no-threshold}
python - <<'PY'
import sys
n = sum(1 for l in open("out/stageB/stageB_accepted.jsonl") if l.strip())
print("accepted", n)
if n < 100:
    open("out/STOPPED_POOL_TOO_SMALL", "w").write(str(n)); sys.exit("fewer than 100 accepted - stop")
PY
[ -f $O/sft/manifest.json ] || python $S/build_nested_v2.py --accepted $O/stageB/stageB_accepted.jsonl --out-dir $O/sft
echo "== V5 dry run $(date -u)"
[ -f $O/dryrun/receipt.json ] || python scripts/train_ranke_lora.py --base $R --data $O/sft/qa_300.jsonl --out $O/dryrun --epochs 1 --limit 32 --verify-reload
for N in 300 1000 3000; do
  echo "== train qa_$N $(date -u)"
  if [ -f $O/adapters/qa_$N/receipt.json ]; then echo "qa_$N already trained - skip"; continue; fi
  if [ -n "${STOP_AFTER_DOSE:-}" ] && [ "$N" -gt "$STOP_AFTER_DOSE" ]; then echo "stopping before qa_$N (checkpoint job)"; exit 0; fi
  python scripts/train_ranke_lora.py --base $R --data $O/sft/qa_$N.jsonl --out $O/adapters/qa_$N --verify-reload
  if [ "$N" = "300" ]; then
    python - <<'PY'
import json, sys
r = json.load(open("out/adapters/qa_300/receipt.json"))
print("loss first20", r["first_20_mean_loss"], "last20", r["last_20_mean_loss"])
if not r["last_20_mean_loss"] < r["first_20_mean_loss"]:
    open("out/STOPPED_LOSS_NOT_FALLING", "w").write(json.dumps(r)); sys.exit("loss did not fall - stop")
PY
  fi
done
echo "== eval answers $(date -u)"
python scripts/generate_blind_eval.py --battery eval/frozen_eval_84.jsonl --root $O --base $R --expected-prompts 84
echo "STAGE_BC_DONE $(date -u)"
touch $O/STAGE_BC_DONE
