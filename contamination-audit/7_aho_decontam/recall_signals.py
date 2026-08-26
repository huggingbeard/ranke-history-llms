"""
recall_signals.py — can we catch (nearly) ALL modern books? Compare CONTENT signals.

Goal: maximise recall — let no modern book through, even one whose date label is wrong.
We measure *content-only* recall: pretend we don't trust the year label and ask whether the
text itself betrays a modern book. Three signals:
  vocab      : >=1 validated post-1945 tripwire word
  intext_yr  : the text prints an explicit post-1945 year (1946-2029)  [>=1, and stricter >=2]
  union      : vocab OR intext_yr

Positives = books whose stated year >= 1946 (genuinely modern; here we use them to see if
CONTENT alone would catch them if the label lied). Negatives = books with year <= 1945, to
measure each signal's false-positive cost (an old book that trips the signal).
"""
import os, re, sys, time
from pathlib import Path
import aho_scan as A

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
N_MOD, N_OLD, CAP = 250, 800, 22000
POSTYEAR = re.compile(r"\b(19(?:4[6-9]|[5-9]\d)|20[0-2]\d)\b")   # 1946-1999, 2000-2029


def main():
    tok = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read_text().strip()
    os.environ["HF_HUB_DISABLE_SYMLINKS"]="1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
    os.environ["HF_TOKEN"]=tok
    from datasets import load_dataset
    m, ctx = A.load_markers(); auto = A.build_automaton(m)

    def signals(text):
        norm = A.normalize(text)
        vocab = {mk for mk, _ in A.scan_text(norm, auto, ctx)}
        years = set(POSTYEAR.findall(norm))
        return len(vocab), len(years)

    ds = None
    for attempt in range(6):   # HF tree endpoint 504s intermittently; retry
        try:
            ds = load_dataset("storytracer/LoC-PD-Books", split="train", streaming=True, token=tok); break
        except Exception as e:
            print(f"  load attempt {attempt+1} failed ({repr(e)[:90]}); retrying...", file=sys.stderr, flush=True)
            time.sleep(8)
    if ds is None:
        raise SystemExit("could not load dataset after retries")
    mod, old, n = [], [], 0
    for rec in ds:
        if n >= CAP or (len(mod) >= N_MOD and len(old) >= N_OLD): break
        n += 1
        yr, txt = rec.get("year"), A.doc_text(rec)
        if not isinstance(yr, int) or not txt: continue
        if yr >= 1946 and len(mod) < N_MOD: mod.append(signals(txt))
        elif yr <= 1945 and len(old) < N_OLD: old.append(signals(txt))
        if n % 4000 == 0: print(f"  streamed {n}: modern={len(mod)} old={len(old)}", file=sys.stderr, flush=True)

    def frac(rows, pred): return 100*sum(1 for r in rows if pred(r))/max(len(rows),1)
    print(f"\nmodern (year>=1946) sampled: {len(mod)}   |   old (year<=1945) sampled: {len(old)}\n")
    print(f"{'signal':<34}{'RECALL on modern':>18}{'false-pos on old':>18}")
    sigs = [
      ("vocabulary (>=1 marker)",          lambda r: r[0] >= 1),
      ("in-text year (>=1 post-1945)",     lambda r: r[1] >= 1),
      ("in-text year (>=2 distinct)",      lambda r: r[1] >= 2),
      ("vocab OR in-text-year(>=1)",       lambda r: r[0] >= 1 or r[1] >= 1),
      ("vocab OR in-text-year(>=2)",       lambda r: r[0] >= 1 or r[1] >= 2),
      ("vocab AND year both present",      lambda r: r[0] >= 1 and r[1] >= 1),
    ]
    for name, pred in sigs:
        print(f"{name:<34}{frac(mod,pred):>17.1f}%{frac(old,pred):>17.1f}%")
    print("\nreading: RECALL = % of modern books that signal would catch from CONTENT alone")
    print("(i.e. even if the date label were wrong); false-pos = % of genuinely old books it trips.")
    # distribution of in-text years among modern (how strong is the signal)
    import statistics
    yrs = [r[1] for r in mod]
    print(f"\nmodern books: median distinct post-1945 years in text = {statistics.median(yrs):.0f}; "
          f"share with 0 = {100*sum(1 for y in yrs if y==0)/len(yrs):.1f}%")


if __name__ == "__main__":
    main()
