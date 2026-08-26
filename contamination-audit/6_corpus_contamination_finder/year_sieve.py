"""
year_sieve.py — the COMPLETE contamination sieve: flag every document, and rank every SOURCE, that
contains a year AFTER its bucket's cutoff.

WHY THIS AND NOT JUST find_contamination.py:
  find_contamination.py greps for ENTITIES you listed (Truman, de Gaulle...). It only catches leaks you
  thought of. This catches ALL dated leakage with no list and no disambiguation: a document written
  before 1913 physically CANNOT contain "1964". So a post-cutoff year is a certain tell of a mis-dated /
  modern source — which is exactly what the date-spike test proved is happening (a 1913 model that knows
  "Civil Rights Act of 1964" read a MODERN document that slipped the date filter).

  Use BOTH: this sieve for everything dated (the bulk), find_contamination.py for undated entity leaks
  (e.g. "de Gaulle" with no year). Together they are comprehensive.

STRATEGY (matches find_contamination.py's philosophy): over-removing is cheap at 160B tokens, keeping
dirty docs is not. The real payload is the SOURCE RANKING at the end: a legitimately-old source has ~0%
of docs with post-cutoff years; a modern source has many. Purge the high-% sources WHOLESALE, fix the
date filter that let them in, then re-run to confirm ~0 flags before retraining.

FALSE POSITIVES: a bare 4-digit number ("1950 rifles") can look like a year. That noise is per-DOCUMENT;
the per-SOURCE aggregation smooths it out (one stray number won't make a source look modern). For a
high-precision doc list, pass --min-hits 2 (require >=2 distinct post-cutoff years in a doc).

CORPUS FORMAT: JSONL, one document per line; text field (default "text"), plus id and source fields.
Streams — never loads the whole corpus.

USAGE
  # per bucket (run once for each cutoff you retrain):
  python year_sieve.py --input corpus_1913.jsonl --cutoff 1913 --out flagged_years_1913.jsonl
  python year_sieve.py --input corpus_1929.jsonl --cutoff 1929 --min-hits 2
  # custom fields:
  python year_sieve.py --input corpus.jsonl --cutoff 1900 --text-field content --source-field url
"""
import argparse, json, re, sys, os
from collections import Counter

# modern-edition wrappers: a pre-1913 BODY can arrive with a modern copyright/scan header. These tells,
# alongside a post-cutoff year, make "this is a modern re-publication" near-certain.
MODERN_TELLS = re.compile(
    r"copyright\s*(?:©|\(c\))?\s*(?:19[3-9]\d|20\d\d)|all rights reserved|\bISBN\b|project gutenberg|"
    r"https?://|www\.|digiti[sz]ed|scanned by|\bOCR\b|\bwikipedia\b|retrieved (?:on )?\d|doi:\s*10\.", re.I)

FOURDIGIT = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def doc_text(rec, field):
    v = rec.get(field)
    if isinstance(v, str):
        return v
    parts = []
    def walk(x):
        if isinstance(x, str): parts.append(x)
        elif isinstance(x, dict):
            for vv in x.values(): walk(vv)
        elif isinstance(x, list):
            for vv in x: walk(vv)
    walk(rec)
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Flag docs/sources with post-cutoff years.")
    ap.add_argument("--input", required=True, help="corpus .jsonl (one doc per line)")
    ap.add_argument("--cutoff", type=int, required=True, help="bucket cutoff year (e.g. 1913)")
    ap.add_argument("--out", default=None, help="flagged docs jsonl (default flagged_years_<cutoff>.jsonl)")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--id-field", default="id")
    ap.add_argument("--source-field", default="source")
    ap.add_argument("--min-hits", type=int, default=1, help="flag a doc with >= this many DISTINCT post-cutoff years")
    ap.add_argument("--maxyear", type=int, default=2035, help="ignore 4-digit numbers above this (not plausible years)")
    ap.add_argument("--checkpoint", type=int, default=20000)
    args = ap.parse_args()
    out_path = args.out or f"flagged_years_{args.cutoff}.jsonl"

    n = flagged = 0
    by_source_flag, by_source_total, by_year, by_source_examples = Counter(), Counter(), Counter(), {}
    outf = open(out_path, "w", encoding="utf-8")
    for ln in open(args.input, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        n += 1
        rec = json.loads(ln)
        src = str(rec.get(args.source_field, "?"))
        by_source_total[src] += 1
        t = doc_text(rec, args.text_field)
        bad = sorted({int(y) for y in FOURDIGIT.findall(t) if args.cutoff < int(y) <= args.maxyear})
        if len(bad) >= args.min_hits:
            flagged += 1
            by_source_flag[src] += 1
            for y in bad:
                by_year[y] += 1
            i = t.find(str(bad[0]))
            snip = t[max(0, i - 90):i + 120].replace("\n", " ").strip()
            outf.write(json.dumps({"id": rec.get(args.id_field, n), "source": src,
                                   "post_cutoff_years": bad[:25],
                                   "modern_tells": bool(MODERN_TELLS.search(t[:5000])),
                                   "snippet": snip[:260]}, ensure_ascii=False) + "\n")
            by_source_examples.setdefault(src, snip[:160])
        if n % args.checkpoint == 0:
            outf.flush()
            print(f"  {n:,} scanned, {flagged:,} flagged ({100*flagged/n:.2f}%)", file=sys.stderr, flush=True)
    outf.close()

    print(f"\n==== year-sieve: {args.input}  (cutoff {args.cutoff}, min-hits {args.min_hits}) ====")
    print(f"docs scanned : {n:,}")
    print(f"docs flagged : {flagged:,}  ({100*flagged/max(n,1):.2f}%)")
    print("\nMOST-CONTAMINATED SOURCES  (flagged / total docs — PURGE the high-% ones wholesale):")
    print(f"   {'flagged/total':>18}  {'%':>6}  source")
    for s, c in by_source_flag.most_common(30):
        tot = by_source_total[s]
        print(f"   {c:>8,}/{tot:<8,}  {100*c/tot:5.1f}%  {s}")
    print("\nPOST-CUTOFF YEARS seen (which era leaked; huge spikes = a dated modern source):")
    for y, c in sorted(by_year.items()):
        bar = "#" * min(60, c * 60 // max(by_year.values()))
        print(f"   {y}  {c:>8,}  {bar}")
    print(f"\nflagged docs -> {out_path}")
    print("NEXT: 1) eyeball a few snippets per top source;  2) purge the high-% sources at the source level;\n"
          "      3) fix the date filter that admitted them;  4) re-run this sieve -> expect ~0 flags;\n"
          "      5) retrain;  6) re-run src/date_spike_full.py -> the leaked events must fall to the ~13% floor.")


if __name__ == "__main__":
    main()
