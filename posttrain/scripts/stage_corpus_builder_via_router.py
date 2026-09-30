from __future__ import annotations

import base64
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
SOURCE_ROOT = Path(r"G:\detritus\ranke_period_qa_scale_20260928")
ARCHIVE = ROOT / "stage_corpus_builder.zip"
REMOTE = "/scratch/hvoth/ranke1929-talkie-sft"
ROUTER = Path(r"C:\Users\hvoth\cluster\router\ch.py")


def main() -> int:
    members = [
        (SOURCE_ROOT / "scripts" / "build_history_passages.py", "scripts/build_history_passages.py"),
        (ROOT / "scripts" / "make_stratified_smoke.py", "scripts/make_stratified_smoke.py"),
        (SOURCE_ROOT / "sources" / "source_manifest.json", "sources/source_manifest.json"),
    ]
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for source, member in members:
            zf.write(source, member)
    payload = ARCHIVE.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    remote_zip = f"{REMOTE}/stage_corpus_builder.zip"
    command = (
        f"set -euo pipefail; mkdir -p {REMOTE}/scripts {REMOTE}/sources; "
        f"printf '%s' '{encoded}' | base64 -d > {remote_zip}; "
        f"echo '{digest}  {remote_zip}' | sha256sum -c -; "
        f"unzip -oq {remote_zip} -d {REMOTE}; "
        f"python -m py_compile {REMOTE}/scripts/build_history_passages.py "
        f"{REMOTE}/scripts/make_stratified_smoke.py; echo CORPUS_BUILDER_STAGED_OK"
    )
    args = [
        sys.executable, str(ROUTER), "submit",
        "--name", "stage-talkie-corpus", "--no-gpu",
        "--est-minutes", "1", "--est-rows", "3",
        "--purpose", "cluster-rehearsal", "--cpus", "1",
        "--mem", "1G", "--time", "00:05:00", "--cmd", command,
    ]
    print(f"archive_bytes={len(payload)} base64_chars={len(encoded)} sha256={digest}")
    return subprocess.call(args)


if __name__ == "__main__":
    raise SystemExit(main())
