"""
build_features.py — the ground-truth feature set for a RECALL-FIRST contamination classifier.

Goal of the classifier: catch >=99.5% of post-1945 books, and subject to that keep as many
genuine old books as possible. To tune that we need a large labelled set with the cheap features.

Streams up to CAP books; keeps EVERY modern book (metadata year >= 1946) and up to N_OLD old
books (<= 1929). Per book we cache the lightweight signals only (no heavy model):
  year   : metadata year (the label)
  mk     : sorted list of RAW keyword/n-gram hits (boundary-checked, NO context guards, so guards
           can be evaluated offline)
  yc     : count of distinct post-1945 years printed in the text (stamp-stripped)
Checkpoints to features.json every CKPT books so a crash never loses the pull.
"""
import os, sys, json, re
from pathlib import Path
import ahocorasick, aho_scan as A

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
CAP, N_OLD, CKPT = 80000, 25000, 5000
OUT = A.HERE / "features.json"

MARKERS = [m["term"].lower() for m in json.loads((A.HERE/"markers_1945.json").read_text())["markers"]]
AUTO = ahocorasick.Automaton()
for t in MARKERS: AUTO.add_word(t, t)
AUTO.make_automaton()


def raw_hits(norm):
    out = set()
    for end, term in AUTO.iter(norm):
        s = end - len(term) + 1
        b = norm[s-1] if s > 0 else " "; a = norm[end+1] if end+1 < len(norm) else " "
        if not (b.isalnum() or a.isalnum()): out.add(term)
    return sorted(out)


def main():
    tok = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read_text().strip()
    os.environ["HF_HUB_DISABLE_SYMLINKS"]="1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
    os.environ["HF_TOKEN"]=tok
    from datasets import load_dataset
    ds = None
    for _ in range(6):
        try: ds = load_dataset("storytracer/LoC-PD-Books", split="train", streaming=True, token=tok); break
        except Exception as e: print("load retry:", repr(e)[:80], file=sys.stderr); import time; time.sleep(8)

    modern, old, n = [], [], 0
    def save():
        OUT.write_text(json.dumps({"markers": MARKERS, "n_streamed": n,
                                   "modern": modern, "old": old}))
    for rec in ds:
        if n >= CAP or (len(old) >= N_OLD and n >= CAP): break
        n += 1
        yr, txt = rec.get("year"), A.doc_text(rec)
        if not isinstance(yr, int) or not txt: continue
        is_mod = yr >= 1946; is_old = yr <= 1929
        if not (is_mod or (is_old and len(old) < N_OLD)): continue   # skip 1930-45 and surplus old
        norm = A.normalize(txt)
        rid = str(rec.get("lccn") or rec.get("filename") or rec.get("ocaid") or n)
        rowfeat = {"id": rid, "y": yr, "mk": raw_hits(norm),
                   "yc": len(set(A.POST1945.findall(norm)))}
        (modern if is_mod else old).append(rowfeat)
        if n % CKPT == 0:
            save(); print(f"  {n} streamed | modern={len(modern)} old={len(old)}", file=sys.stderr, flush=True)
    save()
    print(f"\nDONE: streamed {n} | modern (>=1946) = {len(modern)} | old (<=1929) = {len(old)}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
