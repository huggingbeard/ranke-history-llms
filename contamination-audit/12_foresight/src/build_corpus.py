# -*- coding: utf-8 -*-
"""Build a PRE-1913 fit corpus for the Jacobian lens, from public-domain period prose (Project Gutenberg). Guaranteed
free of any post-1913 content, so the fitted lens cannot import the 'forbidden' answers -- any 1917/executed readout in
the middle layers must then come from the MODEL, not the lens. Same corpus is used for both the 1913 and 1929 lenses so
the readout difference is the model. Writes ~180 chunks of ~110 words to results/jlens/fit_corpus_pre1913.json.
Run:  python src/build_corpus.py
"""
import os, re, json, urllib.request
import common

OUT = os.path.join(common.RESULTS, "jlens")
os.makedirs(OUT, exist_ok=True)
# id -> title (all first published well before 1913)
BOOKS = {3300: "Smith, Wealth of Nations (1776)", 1301: "Carlyle, The French Revolution (1837)",
         2612: "Macaulay, History of England (1848)", 1228: "Darwin, Origin of Species (1859)",
         731: "Gibbon, Decline and Fall (1776)", 147: "Bagehot, Lombard Street (1873)"}
PER_BOOK = 32                     # chunks to take from each, spread across the text
WORDS = 110

chunks = []
for bid, title in BOOKS.items():
    try:
        url = f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt"
        raw = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "ignore")
    except Exception as e:
        print(f"  skip {bid} ({title}): {e}"); continue
    body = raw
    if "*** START" in body:
        body = body.split("*** START", 1)[1]
        if "***" in body:
            body = body.split("***", 1)[1]              # skip the rest of the START marker line
    if "*** END" in body:
        body = body.split("*** END", 1)[0]
    body = re.sub(r"\s+", " ", body).strip()
    words = body.split()
    words = words[300:]                                  # skip front matter
    step = max(WORDS + 20, (len(words) - WORDS) // PER_BOOK)
    got = 0
    for start in range(0, len(words) - WORDS, step):
        c = " ".join(words[start:start + WORDS]).strip()
        if len(c) > 400:
            chunks.append(c); got += 1
        if got >= PER_BOOK:
            break
    print(f"  {title}: {got} chunks ({len(words)} words)")

json.dump(chunks, open(os.path.join(OUT, "fit_corpus_pre1913.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\ntotal {len(chunks)} pre-1913 chunks -> {os.path.join(OUT, 'fit_corpus_pre1913.json')}")
# sanity: confirm NO post-1913 year appears
bad = [c[:80] for c in chunks if re.search(r"\b(19[2-9]\d|191[4-9])\b", c)]
print(f"chunks containing a post-1913 year: {len(bad)}  (should be ~0)")
for b in bad[:5]:
    print("   !", b)
