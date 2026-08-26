"""Inspect any HF dataset's schema from a live streamed record. Usage: python inspect_ds.py <repo> [config]"""
import sys, os, json
from datasets import load_dataset, get_dataset_config_names

repo = sys.argv[1]
cfg = sys.argv[2] if len(sys.argv) > 2 else None
tok = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip()
os.environ["HF_TOKEN"] = tok

try:
    print("configs:", get_dataset_config_names(repo, token=tok))
except Exception as e:
    print("configs: (could not list)", repr(e)[:200])

ds = load_dataset(repo, cfg, split="train", streaming=True, token=tok) if cfg \
    else load_dataset(repo, split="train", streaming=True, token=tok)
row = next(iter(ds))
print("\n--- FIELDS ---")
for k, v in row.items():
    print(f"  {k:<22} ({type(v).__name__:>6}): {str(v)[:150].replace(chr(10),' ')}")
