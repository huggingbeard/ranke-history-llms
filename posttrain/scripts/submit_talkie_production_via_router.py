from __future__ import annotations

import argparse
import base64
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
ROUTER = Path(r"C:\Users\hvoth\cluster\router\ch.py")
REMOTE = "/scratch/hvoth/ranke1929-talkie-sft"
PYTHON = "/scratch/hvoth/ranke-event120-20260912/.venv/bin/python"
CORPUS_SHA256 = "803d71395ad53355f61e04e249def5a1e2798c4e48d83a05d9af49d4a909ba04"
MODEL_REVISION = "8033675be6360ae0127fa75f941c12d52064f1dc"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpu-type", default="H100", choices=("A100", "H100", "H200"))
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--run-tag", default="prod_20260928")
    args = ap.parse_args()
    if args.shards < 1:
        ap.error("--shards must be at least 1")

    archive = ROOT / "stage_production_scripts.zip"
    sources = [
        ROOT / "scripts" / "generate_talkie_pairs.py",
        ROOT / "scripts" / "validate_talkie_pairs.py",
    ]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for source in sources:
            zf.write(source, source.name)
    payload = archive.read_bytes()
    archive_sha = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")

    tag = args.run_tag.replace("-", "_")
    task = "${SLURM_ARRAY_TASK_ID}"
    count = "${SLURM_ARRAY_TASK_COUNT}"
    remote_zip = f"{REMOTE}/stage_{tag}_{task}.zip"
    script_dir = f"{REMOTE}/scripts_{tag}_{task}"
    raw = f"{REMOTE}/outputs/talkie_{tag}_raw_shard_{task}.jsonl"
    accepted = f"{REMOTE}/outputs/talkie_{tag}_accepted_shard_{task}.jsonl"
    report = f"{REMOTE}/outputs/talkie_{tag}_report_shard_{task}.json"

    cmd = (
        f"mkdir -p {script_dir} {REMOTE}/outputs {REMOTE}/hf; "
        f"printf '%s' '{encoded}' | base64 -d > {remote_zip}; "
        f"echo '{archive_sha}  {remote_zip}' | sha256sum -c -; "
        f"unzip -oq {remote_zip} -d {script_dir}; "
        f"{PYTHON} -m py_compile {script_dir}/generate_talkie_pairs.py "
        f"{script_dir}/validate_talkie_pairs.py; "
        f"echo '{CORPUS_SHA256}  {REMOTE}/data/history_passages_v1.jsonl' | sha256sum -c -; "
        f"{PYTHON} {script_dir}/generate_talkie_pairs.py "
        f"--input {REMOTE}/data/history_passages_v1.jsonl --output {raw} "
        f"--variants 2 --num-shards {count} --shard-index {task} --seed 1929 "
        f"--model talkie-1930-13b-it --model-revision {MODEL_REVISION} "
        f"--device cuda --temperature 0.55 --top-p 0.90 --max-tokens 360 "
        f"--hf-home {REMOTE}/hf; "
        f"{PYTHON} {script_dir}/validate_talkie_pairs.py "
        f"--generated {raw} --sources {REMOTE}/data/history_passages_v1.jsonl "
        f"--accepted {accepted} --report {report}"
    )
    router_args = [
        sys.executable,
        str(ROUTER),
        "submit",
        "--name",
        f"talkie-qa-{tag}",
        "--needs-gpu",
        "--gpu-mem",
        "32",
        "--gpu-type",
        args.gpu_type,
        "--batch",
        "--est-minutes",
        "360",
        "--est-rows",
        "4154",
        "--purpose",
        "work",
        "--cpus",
        "1",
        "--mem",
        "64G",
        "--time",
        "12:00:00",
        "--array",
        f"0-{args.shards - 1}",
        "--cmd",
        cmd,
    ]
    if args.dry_run:
        router_args.append("--dry-run")
    print(
        f"archive_bytes={len(payload)} archive_sha256={archive_sha} "
        f"model_revision={MODEL_REVISION} shards={args.shards}",
        flush=True,
    )
    return subprocess.call(router_args)


if __name__ == "__main__":
    raise SystemExit(main())
