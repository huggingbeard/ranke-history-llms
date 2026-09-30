#!/usr/bin/env python3
"""Download one exact Hugging Face revision and print its snapshot path."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()
    path = Path(snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        cache_dir=args.cache_dir,
    ))
    if args.revision not in path.parts:
        raise RuntimeError(f"revision lock failed: {args.revision} not in {path}")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
