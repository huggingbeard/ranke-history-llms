"""
benchmark.py — the sample exercise: how fast can Aho-Corasick decontaminate LoC-PD-Books?

Streams a sample of REAL books from storytracer/LoC-PD-Books (genuine pre-1929 public-domain
OCR, so a near-zero hit rate is the expected clean baseline), injects a few synthetic
post-1945 "contaminated" docs to prove detection fires, then runs the SAME job two ways:

  (A) Aho-Corasick  — one pass, all 130 markers at once, O(text) regardless of #patterns
  (B) union regex   — re.compile("|".join(\bmarker\b ...)), the dependency-free baseline that
                      find_contamination.py / scan_sft.py currently use

and reports build time, throughput (MB/s, MChars/s, books/s), the A-vs-B speedup, the books
flagged (with their STATED metadata year, the metadata-vs-content cross-check), and an
extrapolation to the full ~48 GB / 140k-book dataset. Saves benchmark_results.json.

Usage:  python benchmark.py [--limit 500]
"""
import argparse, json, os, re, sys, time
from collections import Counter
from pathlib import Path

import aho_scan as A

HERE = Path(__file__).resolve().parent
try:  # OCR text contains non-cp1252 glyphs (e.g. ■); print them safely on Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Synthetic contaminated docs (these MUST be caught). Mixed with period-plausible filler.
INJECTED = [
    ("INJECT-coldwar-2001", 2001,
     "It was a fine morning in the old county town. " * 40 +
     "Historians now agree that the Cold War shaped the second half of the twentieth century, "
     "from the Berlin Wall to the collapse of the Soviet Union."),
    ("INJECT-tech-1998", 1998,
     "The merchant kept careful ledgers and rode out at dawn. " * 40 +
     "He uploaded the files over the internet using software on his personal computer, "
     "having first tested the transistor circuits with a laser."),
    ("INJECT-money-1890", 1890,  # decoy: contains 'lsd' as money + 'common market' as a square
     "Goods were sold for 3 lsd at the common market in the village square, "
     "and the players lowered the iron curtain when the stage caught fire. " * 8),
]


def union_regex(markers):
    # markers are already lowercased; text is lowercased before matching, so no re.I needed.
    return re.compile(r"\b(?:" + "|".join(re.escape(m) for m in markers) + r")\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500, help="how many real books to stream")
    ap.add_argument("--repo", default="storytracer/LoC-PD-Books")
    args = ap.parse_args()

    markers, contexts = A.load_markers()
    print(f"markers: {len(markers)}  | homograph context guards: {sorted(contexts)}")

    # ---- build both engines (timed) ----
    t = time.perf_counter(); auto = A.build_automaton(markers); t_build_ac = time.perf_counter() - t
    t = time.perf_counter(); rgx = union_regex(markers);        t_build_rx = time.perf_counter() - t
    print(f"build: aho-corasick {t_build_ac*1000:.1f} ms | union-regex {t_build_rx*1000:.1f} ms")

    # ---- stream real books ----
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from datasets import load_dataset
    tokpath = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt")
    tok = tokpath.read_text().strip() if tokpath.exists() else None
    if tok: os.environ["HF_TOKEN"] = tok
    print(f"streaming {args.limit} books from {args.repo} ...")
    ds = load_dataset(args.repo, split="train", streaming=True, token=tok)

    corpus = []  # (id, year, title, normalized_text)
    t = time.perf_counter()
    for i, rec in enumerate(ds):
        if i >= args.limit: break
        corpus.append((rec.get("lccn") or rec.get("filename") or i, rec.get("year"),
                       rec.get("title"), A.normalize(A.doc_text(rec))))
        if (i + 1) % 100 == 0: print(f"  pulled {i+1} ...")
    t_stream = time.perf_counter() - t
    for cid, yr, txt in INJECTED:
        corpus.append((cid, yr, "[SYNTHETIC INJECTED]", A.normalize(txt)))

    total_chars = sum(len(c[3]) for c in corpus)
    total_bytes = sum(len(c[3].encode("utf-8")) for c in corpus)
    real_books = len(corpus) - len(INJECTED)
    print(f"streamed {real_books} real books in {t_stream:.1f}s "
          f"(+{len(INJECTED)} injected); {total_bytes/1e6:.1f} MB normalized text\n")

    # ---- scan with Aho-Corasick (timed) ----
    t = time.perf_counter()
    flagged_ac = {}
    for cid, yr, title, norm in corpus:
        hits = A.scan_text(norm, auto, contexts)
        if hits:
            flagged_ac[cid] = (yr, title, sorted({h[0] for h in hits}), hits[0])
    t_ac = time.perf_counter() - t

    # ---- scan with union regex (timed) ----
    t = time.perf_counter()
    flagged_rx = {}
    for cid, yr, title, norm in corpus:
        found = []
        for m in rgx.finditer(norm):  # finditer (not findall): positions, to apply the
            term = m.group(0)         # SAME proximity context guard the AC path uses
            cx = contexts.get(term)
            if cx is None:
                found.append(term)
            else:
                lo, hi = max(0, m.start() - A.CONTEXT_WINDOW), min(len(norm), m.end() + A.CONTEXT_WINDOW)
                if cx.search(norm[lo:hi]):
                    found.append(term)
        if found:
            flagged_rx[cid] = sorted(set(found))
    t_rx = time.perf_counter() - t

    # ---- throughput ----
    mb = total_bytes / 1e6
    mchars = total_chars / 1e6
    ac_mbps, rx_mbps = mb / t_ac, mb / t_rx
    speedup = t_rx / t_ac
    print("=" * 70)
    print(f"{'engine':<16}{'time':>10}{'MB/s':>12}{'MChars/s':>12}{'books/s':>12}")
    print(f"{'Aho-Corasick':<16}{t_ac:>9.3f}s{ac_mbps:>12.1f}{mchars/t_ac:>12.1f}{len(corpus)/t_ac:>12.0f}")
    print(f"{'union regex':<16}{t_rx:>9.3f}s{rx_mbps:>12.1f}{mchars/t_rx:>12.1f}{len(corpus)/t_rx:>12.0f}")
    print(f"\nAho-Corasick is {speedup:.1f}x faster than the union-regex baseline.")

    # ---- agreement / correctness ----
    set_ac, set_rx = set(flagged_ac), set(flagged_rx)
    inj_ids = {c[0] for c in INJECTED}
    caught_inj = sorted(i for i in inj_ids if i in set_ac)
    decoy = "INJECT-money-1890"
    print(f"\ncorrectness:")
    print(f"  injected contaminated docs caught : {len(caught_inj)}/{len(inj_ids)-1 if decoy in inj_ids else len(inj_ids)} expected "
          f"({[i for i in caught_inj]})")
    print(f"  decoy (lsd=money / iron curtain=theatre) flagged? {'YES (BAD)' if decoy in set_ac else 'no (correct)'}")
    print(f"  AC vs regex flag the same docs?    {set_ac == set_rx}  "
          f"(AC={len(set_ac)}, regex={len(set_rx)})")

    # ---- flagged REAL books (metadata vs content cross-check) ----
    real_flags = [(cid, *flagged_ac[cid]) for cid in flagged_ac if cid not in inj_ids]
    norm_by_id = {c[0]: c[3] for c in corpus}
    marker_tally = Counter(m for _, _, _, terms, _ in real_flags for m in terms)
    print(f"\nflagged REAL books: {len(real_flags)}/{real_books} "
          f"({100*len(real_flags)/max(real_books,1):.2f}%)")
    print(f"markers firing on real books (tally): {dict(marker_tally.most_common())}")
    for cid, yr, title, terms, (mk, st) in real_flags:
        snip = A.snippet(norm_by_id[cid], st, mk)
        print(f"  [{cid} | yr={yr}] {str(title)[:42]:<42} {terms[:6]}")
        print(f"      ...{snip[:150]}...")

    # ---- extrapolation to the full dataset ----
    FULL_GB = 48.0   # ~8B words * ~6 bytes/word
    FULL_BOOKS = 140_000
    full_secs = FULL_GB * 1000 / ac_mbps
    print(f"\nextrapolation (single core, Aho-Corasick @ {ac_mbps:.0f} MB/s):")
    print(f"  full LoC-PD-Books ~{FULL_GB:.0f} GB / {FULL_BOOKS:,} books "
          f"-> ~{full_secs/60:.1f} min  (~{full_secs/3600:.2f} h)")
    print(f"  on 8 cores in parallel            -> ~{full_secs/60/8:.1f} min")

    out = {
        "n_markers": len(markers), "real_books": real_books, "injected": len(INJECTED),
        "total_MB": round(mb, 1), "total_Mchars": round(mchars, 1),
        "build_ms": {"aho_corasick": round(t_build_ac*1000, 2), "union_regex": round(t_build_rx*1000, 2)},
        "scan_seconds": {"aho_corasick": round(t_ac, 4), "union_regex": round(t_rx, 4)},
        "throughput_MBps": {"aho_corasick": round(ac_mbps, 1), "union_regex": round(rx_mbps, 1)},
        "books_per_sec_ac": round(len(corpus)/t_ac, 1),
        "speedup_ac_over_regex": round(speedup, 2),
        "injected_caught": caught_inj, "decoy_flagged": decoy in set_ac,
        "ac_regex_agree": set_ac == set_rx,
        "flagged_real_books": len(real_flags),
        "flagged_real_detail": [{"id": cid, "year": yr, "title": title, "markers": terms}
                                for cid, yr, title, terms, _ in real_flags],
        "extrapolation": {"full_GB": FULL_GB, "full_books": FULL_BOOKS,
                          "single_core_minutes": round(full_secs/60, 1),
                          "eight_core_minutes": round(full_secs/60/8, 1)},
    }
    (HERE / "benchmark_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved -> {HERE/'benchmark_results.json'}")


if __name__ == "__main__":
    main()
