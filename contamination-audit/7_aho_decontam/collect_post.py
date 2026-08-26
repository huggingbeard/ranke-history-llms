"""
collect_post.py — grow the POST-1945 pool: stream the WHOLE LoC-PD-Books and cache every
metadata-year>=1946 book (post candidate) plus a random pre-1945 reservoir, with text + lccn,
so we can OpenLibrary-confirm them and build a larger balanced ground-truth benchmark.
Dedupe by lccn against what features_v2 already holds. Resumable checkpoint.
"""
import os, sys, json, random
from pathlib import Path
import aho_scan as A

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
OUT = A.HERE / "post_pool.json"
random.seed(7)
have = set()
try:
    have = {r["id"] for r in json.loads((A.HERE/"features_v2.json").read_text(encoding="utf-8"))["modern"]}
except Exception: pass

def main():
    tok = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read_text().strip()
    os.environ["HF_TOKEN"]=tok; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
    from datasets import load_dataset
    ds=None
    for _ in range(6):
        try: ds=load_dataset("storytracer/LoC-PD-Books", split="train", streaming=True, token=tok); break
        except Exception as e: print("retry", repr(e)[:60], file=sys.stderr); import time; time.sleep(8)
    post, pre_res, n = [], [], 0
    seen = set(have)
    def save(): OUT.write_text(json.dumps({"post": post, "pre_reservoir": pre_res, "n": n}), encoding="utf-8")
    for rec in ds:
        n += 1
        yr, txt = rec.get("year"), A.doc_text(rec)
        if not isinstance(yr, int) or not txt: continue
        rid = str(rec.get("lccn") or rec.get("filename") or n)
        if yr >= 1946 and rid not in seen:
            seen.add(rid)
            post.append({"id": rid, "y": yr, "tx": A.normalize(txt)[:4000]})
        elif yr <= 1929 and len(pre_res) < 4000 and rid not in seen:
            pre_res.append({"id": rid, "y": yr, "tx": A.normalize(txt)[:4000]})
        if n % 10000 == 0:
            save(); print(f"  {n} streamed | new post {len(post)} | pre reservoir {len(pre_res)}", file=sys.stderr, flush=True)
    save()
    print(f"DONE streamed {n} | new post candidates {len(post)} | pre reservoir {len(pre_res)}")

if __name__ == "__main__":
    main()
