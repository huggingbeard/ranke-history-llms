#!/usr/bin/env python3
"""Submit a local-or-cluster Ranke base evaluation through the compute router."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
ROUTER = Path(r"C:\Users\hvoth\cluster\router\ch.py")
VENV_PYTHON = Path(r"G:\detritus\ranke_lora_smoke_20260927\.venv\Scripts\python.exe")
GENERATOR = ROOT / "scripts" / "generate_blind_eval.py"
BASE_REVISION = "fb4cede16ed59e1249ebeba2b4e5b9ec2a560642"
BASE = Path(
    r"C:\Users\hvoth\.cache\huggingface\hub\models--uzh-echist-org--Ranke-4B-1929_v2"
    rf"\snapshots\{BASE_REVISION}"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--run-tag")
    args = parser.parse_args()

    for required in (ROUTER, VENV_PYTHON, GENERATOR, BASE):
        if not required.exists():
            raise FileNotFoundError(required)

    if args.full:
        battery = ROOT / "eval" / "frozen_eval_84.jsonl"
        expected = 84
        batch_size = 8
        estimate = 25
        name = "ranke1929-base-eval-84"
        default_tag = "base84"
    else:
        battery = ROOT / "eval" / "smoke_eval_2.jsonl"
        expected = 2
        batch_size = 2
        estimate = 5
        name = "ranke1929-base-eval-smoke2"
        default_tag = "smoke2"

    if not battery.exists():
        raise FileNotFoundError(battery)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.run_tag or f"{default_tag}_{stamp}"
    output = ROOT / "experiments" / tag
    if output.exists():
        raise FileExistsError(output)

    run_parts = [
        str(VENV_PYTHON), str(GENERATOR),
        "--battery", str(battery),
        "--root", str(output),
        "--base", str(BASE),
        "--base-revision", BASE_REVISION,
        "--expected-prompts", str(expected),
        "--batch-size", str(batch_size),
    ]
    command = subprocess.list2cmdline(run_parts)
    router_parts = [
        sys.executable, str(ROUTER), "submit",
        "--name", name,
        "--needs-gpu", "--gpu-mem", "6",
        "--time-sensitive",
        "--est-minutes", str(estimate),
        "--est-rows", str(expected),
        "--purpose", "work",
        "--cmd", command,
    ]
    print(json.dumps({"output": str(output), "command": command}, indent=2), flush=True)
    return subprocess.call(router_parts)


if __name__ == "__main__":
    raise SystemExit(main())
