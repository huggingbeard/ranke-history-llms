from __future__ import annotations

import base64
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
ARCHIVE = ROOT / "stage_scripts.zip"
REMOTE = "/scratch/hvoth/ranke1929-talkie-sft"
ROUTER = Path(r"C:\Users\hvoth\cluster\router\ch.py")


def main() -> int:
    sources = [
        ROOT / "scripts" / "generate_talkie_pairs.py",
        ROOT / "scripts" / "validate_talkie_pairs.py",
    ]
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for source in sources:
            zf.write(source, source.name)

    payload = ARCHIVE.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    remote_zip = f"{REMOTE}/stage_scripts.zip"
    command = (
        f"set -euo pipefail; mkdir -p {REMOTE}/scripts {REMOTE}/data/raw "
        f"{REMOTE}/logs {REMOTE}/outputs; "
        f"printf '%s' '{encoded}' | base64 -d > {remote_zip}; "
        f"echo '{digest}  {remote_zip}' | sha256sum -c -; "
        f"unzip -oq {remote_zip} -d {REMOTE}/scripts; "
        f"python -m py_compile {REMOTE}/scripts/generate_talkie_pairs.py "
        f"{REMOTE}/scripts/validate_talkie_pairs.py; echo STAGED_OK"
    )
    args = [
        sys.executable,
        str(ROUTER),
        "submit",
        "--name",
        "stage-talkie-scripts",
        "--no-gpu",
        "--est-minutes",
        "1",
        "--est-rows",
        "2",
        "--purpose",
        "cluster-rehearsal",
        "--cpus",
        "1",
        "--mem",
        "1G",
        "--time",
        "00:05:00",
        "--cmd",
        command,
    ]
    print(f"archive_bytes={len(payload)} base64_chars={len(encoded)} sha256={digest}")
    return subprocess.call(args)


if __name__ == "__main__":
    raise SystemExit(main())
