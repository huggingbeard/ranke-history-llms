#!/bin/bash
# Pod setup for plan v2.4. Run from /workspace/run. Logs to /workspace/run/logs/setup.log
set -euo pipefail
cd /workspace/run
mkdir -p logs out
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(cat /workspace/run/secrets/hf.txt)
# safety: pod terminates itself after 9 hours no matter what
nohup python /workspace/run/scripts/v2/selfdestruct.py 32400 > logs/selfdestruct.log 2>&1 &
echo "selfdestruct armed pid $! at $(date -u)"
python -m pip install -q "transformers==5.12.1" "peft==0.18.1" "accelerate==1.14.0" bitsandbytes tiktoken \
  "huggingface_hub[hf_transfer]" hf_transfer numpy
python -m pip install -q -e /workspace/run/talkie-upstream
python - <<'EOF'
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name())
print("transformers", transformers.__version__, "peft", peft.__version__)
from huggingface_hub import snapshot_download, hf_hub_download
p = snapshot_download("uzh-echist-org/Ranke-4B-1929_v2", revision="fb4cede16ed59e1249ebeba2b4e5b9ec2a560642")
print("RANKE_PATH", p)
for f in ("rl-refined.pt", "vocab.txt"):
    print("TALKIE", hf_hub_download("talkie-lm/talkie-1930-13b-it", f, revision="8033675be6360ae0127fa75f941c12d52064f1dc"))
EOF
sha256sum $HF_HOME/hub/models--talkie-lm--talkie-1930-13b-it/snapshots/8033675be6360ae0127fa75f941c12d52064f1dc/rl-refined.pt
echo SETUP_DONE
