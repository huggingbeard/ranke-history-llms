#!/usr/bin/env python3
"""Night run 2026-09-29, local RTX 4090 (submitted through the router's local queue).
Same recipe as the cluster arm script cluster_arm.sh: train one LoRA adapter with train_ranke_lora_v3.py, then
answer the 84 frozen eval prompts with eval_adapters_v3.py at the chosen epochs. Resumable: training is skipped
when its receipt.json exists; eval skips conditions already written.
Usage (venv G:\\detritus\\ranke_lora_smoke_20260927\\.venv):
  python night_local_arm.py --arm NAME --data FILE [--base DIR --base-revision ID] [--epochs 3 --r 64 --alpha 128
         --targets all --lr 1e-4 --eval-epochs 1,2,3] [--raw-eval NAME] [--limit N --no-eval]"""
import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
PY = sys.executable
BASE = r"C:\Users\hvoth\.cache\huggingface\hub\models--uzh-echist-org--Ranke-4B-1929_v2\snapshots\fb4cede16ed59e1249ebeba2b4e5b9ec2a560642"
REV = "fb4cede16ed59e1249ebeba2b4e5b9ec2a560642"
ALL = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--data")
ap.add_argument("--base", default=BASE)
ap.add_argument("--base-revision", default=REV)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--r", type=int, default=64)
ap.add_argument("--alpha", type=int, default=128)
ap.add_argument("--targets", default="all")
ap.add_argument("--lr", default="1e-4")
ap.add_argument("--eval-epochs", default="1,2,3")
ap.add_argument("--raw-eval", help="also answer the 84 prompts with the base model itself, under this name")
ap.add_argument("--limit", type=int)
ap.add_argument("--no-eval", action="store_true")
ap.add_argument("--no-train", action="store_true")
a = ap.parse_args()

env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
           PYTHONPATH=str(ROOT / "scripts" / "v2" / "site"), PYTHONIOENCODING="utf-8")
out = ROOT / "runs" / "night_20260929" / a.arm
out.mkdir(parents=True, exist_ok=True)


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, cwd=str(ROOT), check=True)


if not a.no_train and not (out / "adapter" / "receipt.json").exists():
    cmd = [PY, str(ROOT / "scripts" / "v3" / "train_ranke_lora_v3.py"), "--base", a.base, "--base-revision",
           a.base_revision, "--data", a.data, "--out", str(out / "adapter"), "--epochs", str(a.epochs), "--lr", a.lr,
           "--lora-r", str(a.r), "--lora-alpha", str(a.alpha), "--targets", ALL if a.targets == "all" else a.targets,
           "--verify-reload", "--micro-tokens", "2048", "--grad-ckpt"]
    if a.limit:
        cmd += ["--limit", str(a.limit)]
    run(cmd)
if not a.no_eval:
    spec = []
    if a.raw_eval:
        spec += ["--adapter", f"{a.raw_eval}=NONE"]
    if not a.no_train:
        spec += sum((["--adapter", f"{a.arm}_ep{e}={out / 'adapter' / f'epoch_{e}'}"] for e in a.eval_epochs.split(",")), [])
    run([PY, str(ROOT / "scripts" / "v3" / "eval_adapters_v3.py"), "--battery", str(ROOT / "eval" / "frozen_eval_84.jsonl"),
         "--base", a.base, "--base-revision", a.base_revision, "--out", str(out / "eval_answers.jsonl")] + spec)
print("ARM_DONE", a.arm, flush=True)
