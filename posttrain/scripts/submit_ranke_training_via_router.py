#!/usr/bin/env python3
"""Stage fixed Talkie data and submit the three nested Ranke QLoRA runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
ROUTER = Path(r"C:\Users\hvoth\cluster\router\ch.py")
REMOTE = "/scratch/hvoth/ranke1929-talkie-sft"
PYTHON = "/scratch/hvoth/ranke-event120-20260912/.venv/bin/python"
BASE_REPO = "uzh-echist-org/Ranke-4B-1929_v2"
BASE_REVISION = "fb4cede16ed59e1249ebeba2b4e5b9ec2a560642"
SIZES = (300, 1000, 3000)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ids(path: Path, expected: int) -> list[str]:
    ids = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row.get("question"), str) or not row["question"].strip():
                    raise ValueError(f"bad question in {path}")
                if not isinstance(row.get("answer"), str) or not row["answer"].strip():
                    raise ValueError(f"bad answer in {path}")
                ids.append(str(row["candidate_id"]))
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f"{path}: expected {expected} unique rows, found {len(ids)}")
    return ids


def run(args: list[str]) -> None:
    shown = subprocess.list2cmdline(args)
    print(shown, flush=True)
    subprocess.run(args, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=str(ROOT / "datasets"))
    ap.add_argument("--run-tag", default="scale_20260928")
    ap.add_argument("--gpu-type", default="A100", choices=("A100", "H100", "H200"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite-stage", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.datasets).resolve()
    files = {size: data_dir / f"talkie_qa_{size}.jsonl" for size in SIZES}
    ids = {size: read_ids(path, size) for size, path in files.items()}
    if not set(ids[300]).issubset(ids[1000]) or not set(ids[1000]).issubset(ids[3000]):
        raise ValueError("training sets are not nested")

    tag = args.run_tag.replace("-", "_")
    stage = ROOT / "scratch" / f"stage_training_{tag}"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "datasets").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "train_ranke_lora.py", stage / "train_ranke_lora.py")
    shutil.copy2(ROOT / "scripts" / "resolve_locked_model.py", stage / "resolve_locked_model.py")
    manifest = {"base_repo": BASE_REPO, "base_revision": BASE_REVISION, "sets": {}}
    for size, source in files.items():
        target = stage / "datasets" / source.name
        shutil.copy2(source, target)
        manifest["sets"][str(size)] = {
            "rows": size,
            "sha256": sha256(target),
            "file": source.name,
        }
    (stage / "stage_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    remote_inputs = f"{REMOTE}/inputs"
    push = [sys.executable, str(ROUTER), "push", "--local", str(stage), "--remote", remote_inputs]
    if args.overwrite_stage:
        push.append("--overwrite")
    if not args.dry_run:
        run(push)
    else:
        print("DRY RUN: " + subprocess.list2cmdline(push), flush=True)

    remote_stage = f"{remote_inputs}/{stage.name}"
    experiment = f"{REMOTE}/experiments/{tag}"
    hf_cache = f"{REMOTE}/hf_ranke"
    hashes = " ".join(
        f"echo '{manifest['sets'][str(size)]['sha256']}  {remote_stage}/datasets/talkie_qa_{size}.jsonl' | sha256sum -c -;"
        for size in SIZES
    )
    setup = (
        f"set -euo pipefail; "
        f"{PYTHON} -m py_compile {remote_stage}/train_ranke_lora.py {remote_stage}/resolve_locked_model.py; "
        f"{hashes} "
        f"BASE=$({PYTHON} {remote_stage}/resolve_locked_model.py "
        f"--repo {BASE_REPO} --revision {BASE_REVISION} --cache-dir {hf_cache}); "
        f"test -d \"$BASE\"; echo BASE_SNAPSHOT=$BASE; "
    )
    smoke_out = f"{experiment}/smoke_8"
    test_cmd = (
        setup
        + f"test ! -e {smoke_out}; "
        + f"{PYTHON} {remote_stage}/train_ranke_lora.py "
          f"--base \"$BASE\" --base-revision {BASE_REVISION} "
          f"--data {remote_stage}/datasets/talkie_qa_300.jsonl "
          f"--out {smoke_out} --epochs 1 --limit 8 --verify-reload; "
        + f"test -f {smoke_out}/receipt.json; echo RANKE_EIGHT_RECORD_GATE_PASSED"
    )
    task = "${SLURM_ARRAY_TASK_ID}"
    size_expr = "${sizes[$SLURM_ARRAY_TASK_ID]}"
    full_cmd = (
        setup
        + f"sizes=(300 1000 3000); size={size_expr}; "
        + f"data={remote_stage}/datasets/talkie_qa_$size.jsonl; "
        + f"out={experiment}/adapters/qa_$size; test ! -e \"$out\"; "
        + f"{PYTHON} {remote_stage}/train_ranke_lora.py "
          f"--base \"$BASE\" --base-revision {BASE_REVISION} "
          f"--data \"$data\" --out \"$out\" --epochs 3 --verify-reload; "
        + f"test -f \"$out/receipt.json\"; echo RANKE_SCALE_TASK_COMPLETE task={task} size=$size"
    )
    router_args = [
        sys.executable, str(ROUTER), "submit",
        "--name", f"ranke1929-sft-{tag}",
        "--needs-gpu", "--gpu-mem", "20", "--gpu-type", args.gpu_type,
        "--batch", "--est-minutes", "240", "--est-rows", "4300",
        "--purpose", "work", "--cluster-only-test",
        "--cpus", "1", "--mem", "64G", "--time", "08:00:00",
        "--array", "0-2", "--test-cmd", test_cmd, "--cmd", full_cmd,
    ]
    if args.dry_run:
        router_args.append("--dry-run")
    print(json.dumps(manifest, indent=2), flush=True)
    return subprocess.call(router_args)


if __name__ == "__main__":
    raise SystemExit(main())
