#!/usr/bin/env python3
"""Stage the frozen battery and submit blind generation for all Ranke conditions."""

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
EXPECTED_PROMPTS = 84


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run(parts: list[str]) -> None:
    print(subprocess.list2cmdline(parts), flush=True)
    subprocess.run(parts, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default=str(ROOT / "eval" / "frozen_eval_84.jsonl"))
    ap.add_argument("--run-tag", default="scale_20260928")
    ap.add_argument("--gpu-type", default="A100", choices=("A100", "H100", "H200"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite-stage", action="store_true")
    args = ap.parse_args()

    battery = Path(args.battery).resolve()
    rows = load_jsonl(battery)
    if len(rows) != EXPECTED_PROMPTS:
        raise ValueError(f"expected {EXPECTED_PROMPTS} frozen prompts, found {len(rows)}")
    battery_hash = sha256(battery)
    tag = args.run_tag.replace("-", "_")
    stage = ROOT / "scratch" / f"stage_eval_{tag}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "generate_blind_eval.py", stage / "generate_blind_eval.py")
    shutil.copy2(ROOT / "scripts" / "resolve_locked_model.py", stage / "resolve_locked_model.py")
    shutil.copy2(battery, stage / "frozen_eval_84.jsonl")
    with (stage / "smoke_eval_2.jsonl").open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows[:2]:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "base_repo": BASE_REPO,
        "base_revision": BASE_REVISION,
        "battery_rows": len(rows),
        "battery_sha256": battery_hash,
    }
    (stage / "stage_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    remote_inputs = f"{REMOTE}/inputs"
    push = [sys.executable, str(ROUTER), "push", "--local", str(stage), "--remote", remote_inputs]
    if args.overwrite_stage:
        push.append("--overwrite")
    if args.dry_run:
        print("DRY RUN: " + subprocess.list2cmdline(push), flush=True)
    else:
        run(push)

    remote_stage = f"{remote_inputs}/{stage.name}"
    experiment = f"{REMOTE}/experiments/{tag}"
    hf_cache = f"{REMOTE}/hf_ranke"
    setup = (
        "set -euo pipefail; "
        f"{PYTHON} -m py_compile {remote_stage}/generate_blind_eval.py {remote_stage}/resolve_locked_model.py; "
        f"echo '{battery_hash}  {remote_stage}/frozen_eval_84.jsonl' | sha256sum -c -; "
        f"BASE=$({PYTHON} {remote_stage}/resolve_locked_model.py "
        f"--repo {BASE_REPO} --revision {BASE_REVISION} --cache-dir {hf_cache}); "
        "test -d \"$BASE\"; echo BASE_SNAPSHOT=$BASE; "
    )
    smoke_root = f"{REMOTE}/eval_smoke/{tag}"
    test_cmd = (
        setup
        + f"rm -rf {smoke_root}; mkdir -p {smoke_root}; "
        + f"{PYTHON} {remote_stage}/generate_blind_eval.py "
          f"--battery {remote_stage}/smoke_eval_2.jsonl --root {smoke_root} "
          f"--base \"$BASE\" --base-revision {BASE_REVISION} --expected-prompts 2 --batch-size 2; "
        + f"test $(wc -l < {smoke_root}/eval/blind_output_matrix.jsonl) -eq 2; "
        + "echo RANKE_EVAL_TWO_PROMPT_GATE_PASSED"
    )
    full_cmd = (
        setup
        + f"test -f {experiment}/adapters/qa_300/receipt.json; "
        + f"test -f {experiment}/adapters/qa_1000/receipt.json; "
        + f"test -f {experiment}/adapters/qa_3000/receipt.json; "
        + f"{PYTHON} {remote_stage}/generate_blind_eval.py "
          f"--battery {remote_stage}/frozen_eval_84.jsonl --root {experiment} "
          f"--base \"$BASE\" --base-revision {BASE_REVISION} "
          f"--expected-prompts {EXPECTED_PROMPTS} --batch-size 8; "
        + f"test $(wc -l < {experiment}/eval/blind_output_matrix.jsonl) -eq 336; "
        + f"echo RANKE_BLIND_MATRIX_COMPLETE sha256=$(sha256sum {experiment}/eval/blind_output_matrix.jsonl | cut -d' ' -f1)"
    )
    router_args = [
        sys.executable, str(ROUTER), "submit",
        "--name", f"ranke1929-eval-{tag}",
        "--needs-gpu", "--gpu-mem", "20", "--gpu-type", args.gpu_type,
        "--batch", "--est-minutes", "120", "--est-rows", "336",
        "--purpose", "work", "--cluster-only-test",
        "--cpus", "1", "--mem", "64G", "--time", "04:00:00",
        "--test-cmd", test_cmd, "--cmd", full_cmd,
    ]
    if args.dry_run:
        router_args.append("--dry-run")
    print(json.dumps(manifest, indent=2), flush=True)
    return subprocess.call(router_args)


if __name__ == "__main__":
    raise SystemExit(main())
