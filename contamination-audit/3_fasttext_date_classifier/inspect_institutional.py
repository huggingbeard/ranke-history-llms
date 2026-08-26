"""Look at the real structure of institutional/institutional-books-1.0 before building anything."""
import os, json
from datasets import load_dataset, get_dataset_config_names

REPO = "institutional/institutional-books-1.0"
tok = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip()
os.environ["HF_TOKEN"] = tok
os.environ["HUGGING_FACE_HUB_TOKEN"] = tok

print("configs:")
try:
    cfgs = get_dataset_config_names(REPO, token=tok)
    print("  ", cfgs)
except Exception as e:
    print("  could not list configs:", repr(e)[:300])
    cfgs = [None]

cfg = cfgs[0] if cfgs else None
print(f"\nstreaming first record (config={cfg}) ...")
try:
    ds = load_dataset(REPO, cfg, split="train", streaming=True, token=tok) if cfg \
        else load_dataset(REPO, split="train", streaming=True, token=tok)
    row = next(iter(ds))
    print("\n--- FIELDS ---")
    for k, v in row.items():
        s = str(v).replace("\n", " ")
        print(f"  {k:<24} ({type(v).__name__:>6}): {s[:160]}")
    # dump full small fields to a file for closer reading
    small = {k: (str(v)[:500]) for k, v in row.items()}
    json.dump(small, open(r"C:\Users\hvoth\timelock-contamination\inst_sample_record.json", "w", encoding="utf-8"), indent=2)
    print("\nwrote inst_sample_record.json")
except Exception as e:
    print("ERROR:", repr(e)[:600])
    print("\nIf this is a gating/401 error: open "
          "https://huggingface.co/datasets/institutional/institutional-books-1.0 "
          "while logged in as the HF.txt account and click 'Agree'.")
