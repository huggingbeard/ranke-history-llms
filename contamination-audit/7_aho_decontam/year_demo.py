"""
year_demo.py — show WHAT the year detector actually matches (date vs noise) on real books.

The detector is one regex (POSTYEAR). Here we run it on real LoC books and print, for old
books that trip it, the matched numbers IN CONTEXT — so you can see how many are real dates vs
OCR noise / non-date numbers. Also reports how often a matched year sits in a DATE context
(©, copyright, parentheses, "in 19xx", a year-range), i.e. how much a date-context filter helps.
"""
import os, re, sys
from pathlib import Path
import aho_scan as A

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
POSTYEAR = re.compile(r"\b(19(?:4[6-9]|[5-9]\d)|20[0-2]\d)\b")
# a "date-looking" use: year right after ©/copyright/(/c./in, or inside a year-range
DATECTX = re.compile(r"(©|copyright|cop\.|\bc\.|\bca\.|circa|\bin |\(|\bvol|–|—)\s*\(?\s*"
                     r"(19(?:4[6-9]|[5-9]\d)|20[0-2]\d)|"
                     r"(19(?:4[6-9]|[5-9]\d)|20[0-2]\d)\s*[-–—]\s*(19|20)\d\d")
CAP, N_OLD, N_MOD = 4000, 14, 4


def main():
    tok = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read_text().strip()
    os.environ["HF_HUB_DISABLE_SYMLINKS"]="1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
    os.environ["HF_TOKEN"]=tok
    from datasets import load_dataset
    ds = load_dataset("storytracer/LoC-PD-Books", split="train", streaming=True, token=tok)

    old_eg, mod_eg, n = [], [], 0
    old_with_yr = old_total = ctx_hits = match_total = 0
    for rec in ds:
        if n >= CAP: break
        n += 1
        yr, txt = rec.get("year"), A.doc_text(rec)
        if not isinstance(yr, int) or not txt: continue
        norm = A.normalize(txt)
        ms = list(POSTYEAR.finditer(norm))
        if yr <= 1945:
            old_total += 1
            if ms:
                old_with_yr += 1
                match_total += len({m.group(0) for m in ms})
                ctx_hits += len(DATECTX.findall(norm))
                if len(old_eg) < N_OLD:
                    distinct = sorted({m.group(0) for m in ms})
                    i = ms[0].start()
                    snip = norm[max(0,i-45):i+45]
                    old_eg.append((yr, str(rec.get("title"))[:34], distinct[:8], snip))
        elif yr >= 1946 and len(mod_eg) < N_MOD:
            distinct = sorted({m.group(0) for m in ms})
            mod_eg.append((yr, str(rec.get("title"))[:34], distinct[:10]))

    print(f"\nscanned {n} books | old (<=1945) with >=1 'recent year': {old_with_yr}/{old_total} "
          f"({100*old_with_yr/max(old_total,1):.0f}%)")
    print(f"of the matched numbers in old books, share that sit in a DATE context "
          f"(©/copyright/parens/range): {100*ctx_hits/max(match_total,1):.0f}%")
    print("\n--- OLD books that tripped the year detector: what number matched, in context ---")
    for yr, t, ds_, snip in old_eg:
        print(f"[{yr}] {t:<34} matched {ds_}")
        print(f"     ...{snip.strip()}...")
    print("\n--- MODERN books: the years their text prints ---")
    for yr, t, ds_ in mod_eg:
        print(f"[{yr}] {t:<34} {ds_}")


if __name__ == "__main__":
    main()
