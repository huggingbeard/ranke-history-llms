from __future__ import annotations

"""Submit the 16-record Talkie run after the exact one-record gate passed.

The full job verifies the saved gate again before loading the model. Both runs
must use the same source archive, corpus, smoke file, model revision, and
generation settings.
"""

import argparse
import base64
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
ROUTER = Path(r"C:\Users\hvoth\cluster\router\ch.py")
REMOTE = "/scratch/hvoth/ranke1929-talkie-sft"
PYTHON = "/scratch/hvoth/ranke-event120-20260912/.venv/bin/python"
CORPUS_SHA256 = "803d71395ad53355f61e04e249def5a1e2798c4e48d83a05d9af49d4a909ba04"
SMOKE_SHA256 = "7f1b4b9bcd9adb2780569ce6dd9806439e400d51f24f099ad16591160144d505"
MODEL_REVISION = "8033675be6360ae0127fa75f941c12d52064f1dc"
MAX_TOKENS = 512
SOURCE_NAMES = (
    "generate_talkie_pairs.py",
    "validate_talkie_pairs.py",
    "verify_talkie_run.py",
)


def deterministic_archive() -> tuple[bytes, str, Path]:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in SOURCE_NAMES:
            source = ROOT / "scripts" / name
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 28, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zf.writestr(info, source.read_bytes())
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    archive = ROOT / "staging" / f"talkie_scripts_{digest}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(payload)
    return payload, digest, archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-type", default="A100", choices=("A100", "H100", "H200"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--gate-generated",
        required=True,
        help="Remote path to the generated JSONL from the passed one-record gate",
    )
    parser.add_argument("--host-mem", default="64G")
    args = parser.parse_args()

    if not args.gate_generated.startswith(f"{REMOTE}/outputs/"):
        parser.error(f"--gate-generated must be below {REMOTE}/outputs/")
    if not args.gate_generated.endswith("_raw.jsonl"):
        parser.error("--gate-generated must end in _raw.jsonl")

    payload, archive_sha, archive = deterministic_archive()
    encoded = base64.b64encode(payload).decode("ascii")
    remote_zip = f"{REMOTE}/staging/{archive.name}"
    script_dir = f"{REMOTE}/scripts/{archive_sha}"
    smoke = f"{REMOTE}/data/history_passages_smoke_stratified.jsonl"
    corpus = f"{REMOTE}/data/history_passages_v1.jsonl"
    tag = args.run_tag.replace("-", "_")
    stem = f"{REMOTE}/outputs/talkie_16_{tag}_{archive_sha[:12]}"
    generated = stem + "_raw.jsonl"
    accepted = stem + "_accepted.jsonl"
    report = stem + "_report.json"
    captures = generated + ".captures"
    gate_stem = args.gate_generated.removesuffix("_raw.jsonl")
    gate_accepted = gate_stem + "_accepted.jsonl"
    gate_report = gate_stem + "_report.json"

    setup = (
        f"mkdir -p {REMOTE}/staging {script_dir} {REMOTE}/outputs; "
        f"if [ ! -f {remote_zip} ]; then printf '%s' '{encoded}' | base64 -d > {remote_zip}; fi; "
        f"echo '{archive_sha}  {remote_zip}' | sha256sum -c -; "
        f"unzip -oq {remote_zip} -d {script_dir}; "
        f"{PYTHON} -m py_compile "
        + " ".join(f"{script_dir}/{name}" for name in SOURCE_NAMES)
        + "; "
        f"echo '{CORPUS_SHA256}  {corpus}' | sha256sum -c -; "
        f"echo '{SMOKE_SHA256}  {smoke}' | sha256sum -c -; "
        f"test \"$(wc -l < {smoke})\" -eq 16; "
        f"export HF_HOME={REMOTE}/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; "
        f"MODEL_REL=models--talkie-lm--talkie-1930-13b-it/snapshots/{MODEL_REVISION}; "
        f"if [ -d {REMOTE}/hf/$MODEL_REL ]; then CACHE_DIR={REMOTE}/hf; "
        f"elif [ -d {REMOTE}/hf/hub/$MODEL_REL ]; then CACHE_DIR={REMOTE}/hf/hub; "
        f"else echo MODEL_CACHE_MISSING >&2; exit 42; fi; "
        f"test -d \"$CACHE_DIR/$MODEL_REL\"; echo CACHE_DIR=$CACHE_DIR"
    )
    gate_verifier = (
        f"{PYTHON} {script_dir}/verify_talkie_run.py "
        f"--generated {args.gate_generated} --accepted {gate_accepted} "
        f"--report {gate_report} --expected-count 1 "
        f"--expected-input-sha256 {SMOKE_SHA256} --expected-code-sha256 {archive_sha} "
        f"--expected-max-tokens {MAX_TOKENS} --min-parsed 1 --min-accepted 1"
    )
    generator = (
        f"{PYTHON} {script_dir}/generate_talkie_pairs.py "
        f"--input {smoke} --output {generated} --limit 16 --variants 1 --seed 1929 "
        f"--model talkie-1930-13b-it --model-revision {MODEL_REVISION} "
        f"--device cuda --temperature 0.55 --top-p 0.90 --top-k 40 "
        f"--max-tokens {MAX_TOKENS} --hf-home {REMOTE}/hf --cache-dir \"$CACHE_DIR\" "
        f"--code-archive-sha256 {archive_sha}"
    )
    validator = (
        f"{PYTHON} {script_dir}/validate_talkie_pairs.py --generated {generated} "
        f"--sources {corpus} --accepted {accepted} --report {report}"
    )
    verifier = (
        f"{PYTHON} {script_dir}/verify_talkie_run.py --generated {generated} "
        f"--accepted {accepted} --report {report} --expected-count 16 "
        f"--expected-input-sha256 {SMOKE_SHA256} --expected-code-sha256 {archive_sha} "
        f"--expected-max-tokens {MAX_TOKENS} --min-parsed 1 --min-accepted 1"
    )
    command = (
        f"set -euo pipefail; {setup}; {gate_verifier}; "
        f"test ! -e {generated}; test ! -e {generated}.request.json; "
        f"test ! -e {generated}.receipt.json; test ! -e {captures}; "
        f"test ! -e {accepted}; test ! -e {report}; "
        f"{generator}; {validator}; {verifier}; echo TALKIE_SIXTEEN_RECORD_RUN_PASSED"
    )
    router_args = [
        sys.executable,
        str(ROUTER),
        "submit",
        "--name",
        f"talkie-exact-16-{args.run_tag}",
        "--needs-gpu",
        "--gpu-mem",
        "32",
        "--gpu-type",
        args.gpu_type,
        "--batch",
        "--est-minutes",
        "90",
        "--est-rows",
        "16",
        "--purpose",
        "work",
        "--cpus",
        "1",
        "--mem",
        args.host_mem,
        "--time",
        "02:00:00",
        "--cmd",
        command,
    ]
    if args.dry_run:
        router_args.append("--dry-run")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "archive_bytes": len(payload),
                "archive_sha256": archive_sha,
                "gate_generated": args.gate_generated,
                "model_revision": MODEL_REVISION,
                "smoke_sha256": SMOKE_SHA256,
                "max_tokens": MAX_TOKENS,
                "generated": generated,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return subprocess.call(router_args)


if __name__ == "__main__":
    raise SystemExit(main())
