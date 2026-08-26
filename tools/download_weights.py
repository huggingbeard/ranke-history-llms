"""
Download missing Ranke / Endeavour weights from the gated HF org uzh-echist-org.

Files used:
  - HF token: D:\\UZHechist Dropbox\\Joachim Voth\\claudecode\\api\\HF.txt
  - Destination: D:\\hf-models\\<model-name>
  - Manifest (per-model status): tools/weights_download_log.csv

Version: v1  (2026-08-26)

Only downloads models NOT already held locally, to avoid duplicates:
  * Ranke-4B-1913-1202-Instruct-WT   (the only Ranke-4B gap)
  * Endeavour-1913-1.7B  + 2 Instruct + 2 GRPO variants (predecessor 1.7B line)

snapshot_download resumes partial downloads, so re-running is safe.
"""
import csv
import time
from pathlib import Path

from huggingface_hub import snapshot_download

TOKEN_FILE = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt")
DEST = Path(r"D:\hf-models")
LOG = Path(__file__).with_name("weights_download_log.csv")

TOKEN = TOKEN_FILE.read_text(encoding="utf-8").strip()

MODELS = [
    "uzh-echist-org/Ranke-4B-1913-1202-Instruct-WT",
    "uzh-echist-org/Endeavour-1913-1.7B",
    "uzh-echist-org/Endeavour-1913-1.7B-1708-Instruct",
    "uzh-echist-org/Endeavour-1913-1.7B-1808-Instruct",
    "uzh-echist-org/Endeavour-1913-1.7B-2708-GRPO",
    "uzh-echist-org/Endeavour-1913-1.7B-2808-GRPO",
]


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def write_manifest(rows):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["repo", "local_dir", "status", "gb", "secs"])
        w.writeheader()
        w.writerows(rows)


def main():
    rows = []
    for repo in MODELS:
        name = repo.split("/")[-1]
        target = DEST / name
        t0 = time.time()
        print(f"--> {repo}", flush=True)
        try:
            snapshot_download(repo_id=repo, local_dir=str(target), token=TOKEN)
            gb = round(dir_size_gb(target), 2)
            rows.append({"repo": repo, "local_dir": str(target), "status": "ok",
                         "gb": gb, "secs": round(time.time() - t0)})
            print(f"OK  {repo} -> {target} ({gb} GB)", flush=True)
        except Exception as e:  # noqa: BLE001 - record and continue
            rows.append({"repo": repo, "local_dir": str(target),
                         "status": f"ERROR: {e}", "gb": "", "secs": round(time.time() - t0)})
            print(f"ERR {repo}: {e}", flush=True)
        write_manifest(rows)  # checkpoint after each model
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
