"""
year_context.py — how much do stamp-stripping, date-context, and front-of-book position clean
up the year signal? Recall on modern books vs false-positives on old books, for each variant.

Variants (all count DISTINCT post-1945 years):
  raw            : any recent number, no cleaning            (old behaviour)
  strip          : any recent number, preservation/IA stamp stripped
  context        : only date-looking years (©, (1972), month, year-range), stamp stripped
  context+front  : date-looking years, only in the first 20% (where a copyright year sits)
"""
import os, sys
from pathlib import Path
import aho_scan as A

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
CAP, N_MOD, N_OLD = 9000, 150, 6000


def main():
    tok = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read_text().strip()
    os.environ["HF_HUB_DISABLE_SYMLINKS"]="1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
    os.environ["HF_TOKEN"]=tok
    from datasets import load_dataset
    ds = load_dataset("storytracer/LoC-PD-Books", split="train", streaming=True, token=tok)

    def variants(text):
        raw = len(set(A.POST1945.findall(" ".join(text.lower().split()))))   # no cleaning
        norm = A.normalize(text)                                              # IA + preservation stamp stripped
        return dict(raw=raw,
                    strip=len(A.post1945_years(norm)),
                    context=len(A.post1945_years(norm, require_context=True)),
                    cfront=len(A.post1945_years(norm, require_context=True, front_frac=0.2)))

    mod, old, n = [], [], 0
    for rec in ds:
        if n >= CAP or (len(mod) >= N_MOD and len(old) >= N_OLD): break
        n += 1
        yr, txt = rec.get("year"), A.doc_text(rec)
        if not isinstance(yr, int) or not txt: continue
        if yr >= 1946 and len(mod) < N_MOD: mod.append(variants(txt))
        elif yr <= 1945 and len(old) < N_OLD: old.append(variants(txt))
        if n % 3000 == 0: print(f"  streamed {n}: modern={len(mod)} old={len(old)}", file=sys.stderr, flush=True)

    print(f"\nmodern={len(mod)}  old={len(old)}\n")
    def pct(rows, key, k): return 100*sum(1 for r in rows if r[key] >= k)/max(len(rows),1)
    print(f"{'variant':<16}{'recall ≥1':>11}{'recall ≥2':>11}{'   |':>5}{'FP old ≥1':>11}{'FP old ≥2':>11}")
    for key, lab in [("raw","raw (no clean)"), ("strip","stamp-stripped"),
                     ("context","date-context"), ("cfront","context + front 20%")]:
        print(f"{lab:<16}{pct(mod,key,1):>10.1f}%{pct(mod,key,2):>10.1f}%{'   |':>5}"
              f"{pct(old,key,1):>10.1f}%{pct(old,key,2):>10.1f}%")
    print("\nrecall = % of modern books caught; FP = % of genuinely old books wrongly flagged.")


if __name__ == "__main__":
    main()
