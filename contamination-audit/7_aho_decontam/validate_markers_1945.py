"""
validate_markers_1945.py — build an empirically-validated post-1945 "tripwire" lexicon.

Goal: n-grams that were *effectively absent from books before 1945* but common after, so
that a book containing one cannot genuinely be pre-1945. We do NOT guess which terms are
anachronistic — we measure it against Google Books Ngrams (an independent, corpus-external
record of when language actually shifted). This is the same method as
period-classifier/src/ngram_sharpen.py, retargeted from a 1913 boundary to 1945.

For each candidate term we fetch its yearly frequency 1800-2019 and compute:
  pre_max  = max annual frequency in 1900-1944   (must be ~0 for a clean tripwire)
  pre_mean = mean annual frequency in 1900-1944
  post_mean= mean annual frequency in 1950-2000  (must be clearly present)
  ratio    = log10(post_mean / pre_mean)         (must be strongly positive)
  birth    = first year the term durably appears  (so the lexicon is reusable at other cutoffs)

KEEP a term as a tripwire iff pre_max < PRE_MAX_FLOOR and post_mean >= POST_FLOOR and
ratio >= RATIO_STRONG. Everything else is written to markers_rejected.json WITH THE REASON,
so we can see the method filtering out wrong guesses (penicillin, FBI, New Deal, Pearl
Harbor are included on purpose as negative controls — they predate or straddle 1945).

Output: markers_1945.json (kept) + markers_rejected.json (failed) + a printed summary.
"""
import json, time, urllib.parse
from pathlib import Path
import numpy as np
import requests

OUT = Path(__file__).resolve().parent
HEADERS = {"User-Agent": "decontam-research/1.0 (academic; joachimvoth@gmail.com)"}

# --- Ngram query config (reused from ngram_sharpen.py) -----------------------
CORPUS = "en-2019"
Y0, Y1 = 1800, 2019
BATCH = 8
# --- 1945 boundary + tripwire thresholds -------------------------------------
PRE_LO, PRE_HI   = 1900, 1944      # "before the cutoff"
POST_LO, POST_HI = 1950, 2000      # "clearly after"
EPS = 1e-12
# Gate on pre_MEAN, not pre_max: Google Books has many mis-dated volumes (modern books
# catalogued to the early 1900s), which inject single-year spikes of post-1945 vocabulary
# into the pre-1945 bucket. pre_max is dominated by that metadata noise; the pre-window
# MEAN is robust to it and reflects genuine pre-1945 usage. (pre_max is still recorded.)
PRE_MEAN_FLOOR = 1.5e-7  # avg annual freq in 1900-44 must be below this (effectively absent)
POST_FLOOR     = 1.0e-8  # post mean must be at least this (term really is used after 1945)
RATIO_STRONG   = 1.5     # log10(post/pre) >= 1.5  (~30x more common after the cutoff)

# --- Candidate tripwires, bucketed -------------------------------------------
# Seeded from ranke-contamination-audit CONFIRMED/MARKERS/BUCKETS plus a broad fresh sweep.
# Deliberately includes pre-1945 / WWII-era NEGATIVE CONTROLS so the validation visibly
# rejects them (marked # neg).
CANDIDATES = {
 "Tech/computing": [
    "transistor", "integrated circuit", "microchip", "semiconductor", "microprocessor",
    "personal computer", "electronic computer", "digital computer", "mainframe", "software",
    "internet", "world wide web", "website", "e-mail", "smartphone", "laser", "photocopier",
    "videotape", "compact disc", "floppy disk", "silicon valley", "artificial intelligence",
    "ballpoint pen", "nylon", "polyester", "teflon", "velcro", "tupperware", "microwave oven",
    "jet engine", "space shuttle", "credit card", "barcode", "fax machine",
 ],
 "Cold War / geopolitics": [
    "cold war", "iron curtain", "united nations", "security council", "nato", "warsaw pact",
    "marshall plan", "berlin wall", "berlin airlift", "korean war", "vietnam war",
    "cuban missile crisis", "domino theory", "third world", "superpower", "decolonization",
    "apartheid", "european union", "european community", "common market", "perestroika",
    "glasnost", "detente", "hydrogen bomb", "nuclear weapon", "intercontinental ballistic",
    "mccarthyism",
 ],
 "WWII (neg controls, 1939-45)": [
    "world war ii", "second world war", "pearl harbor", "d-day", "blitzkrieg", "atomic bomb",
    "hiroshima", "nagasaki", "holocaust", "auschwitz", "manhattan project", "lend-lease",
    "new deal", "great depression",   # neg: 1930s
 ],
 "Science/medicine": [
    "penicillin", "antibiotic", "polio vaccine", "double helix", "genetic code",
    "big bang theory", "plate tectonics", "continental drift", "radiocarbon", "chemotherapy",
    "organ transplant", "heart transplant", "pacemaker",
 ],
 "Culture/society": [
    "rock and roll", "rhythm and blues", "teenager", "baby boom", "supermarket",
    "shopping mall", "drive-in", "motel", "fast food", "television", "sitcom", "talk show",
    "science fiction", "comic book", "civil rights movement", "environmentalism",
    "globalization", "suburbia",
 ],
 "Orgs/brands": [
    "nasa", "cia", "kgb", "fbi", "unesco", "unicef", "world health organization",
    "world bank", "international monetary fund", "opec", "mcdonald's", "coca-cola",
    "disneyland", "volkswagen", "boeing 747",
 ],
 "People (post-1945)": [
    "de gaulle", "khrushchev", "brezhnev", "gorbachev", "mao zedong", "deng xiaoping",
    "ho chi minh", "fidel castro", "che guevara", "martin luther king", "john f. kennedy",
    "richard nixon", "ronald reagan", "margaret thatcher", "nelson mandela",
    "neil armstrong", "elvis presley", "the beatles",
 ],
 "Events": [
    "moon landing", "apollo 11", "space race", "cultural revolution", "great leap forward",
    "prague spring", "chernobyl", "watergate",
 ],
 "Added (space/cold-war/sci)": [
    "astronaut", "cosmonaut", "spacecraft", "space station", "nuclear power",
    "nuclear reactor", "supersonic", "jet airliner", "napalm", "ddt", "lsd", "genocide",
    "civil rights act", "khmer rouge", "viet cong", "maoism", "denazification", "email",
    "jumbo jet", "nuclear submarine", "third world", "marshall plan", "korean war",
 ],
}

# Context-disambiguation regexes for hook-prone surface forms (ported from
# find_contamination.py). Attached to a kept marker so aho_scan.py can require context
# before flagging. Only matters for terms that also survive the frequency test.
CONTEXT = {
    "israel": r"1948|state of israel|israeli|zionis|ben-gurion",
    "normandy": r"1944|d-day|allied|landing|overlord|eisenhower",
    "hiroshima": r"atomic|bomb|1945|nuclear",
    "nagasaki": r"atomic|bomb|1945|nuclear",
    "pearl harbor": r"1941|attack|japanese|bomb",
    "holocaust": r"jewish|nazi|jews|genocide|six million",
    "mao zedong": None,
    "de gaulle": r"general|france|french|free french|president|1940|194\d|195\d|196\d",
    "vichy": r"france|regime|government|p.tain|occupied|1940|nazi",
}


def fetch_ngrams(phrases):
    """Batched query to the Ngram Viewer JSON endpoint (case-insensitive)."""
    content = ",".join(phrases)
    url = "https://books.google.com/ngrams/json?" + urllib.parse.urlencode({
        "content": content, "year_start": Y0, "year_end": Y1,
        "corpus": CORPUS, "smoothing": 0, "case_insensitive": "true",
    })
    for _ in range(4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=45)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(2.5)
    return []


def win(ts, lo, hi, agg):
    a, b = lo - Y0, hi - Y0 + 1
    seg = ts[a:b]
    if not seg:
        return 0.0
    return float(agg(seg))


def birth_year(ts, post_mean):
    """First year the term *durably* appears: a sustained 5-year average above
    max(POST_FLOOR, 10% of post mean). The 5-year window rejects single-year spikes
    from mis-dated volumes (which otherwise produce nonsense births like 1865)."""
    thresh = max(POST_FLOOR, 0.10 * post_mean)
    for y in range(1850, Y1 - 4):
        i = y - Y0
        seg = ts[i:i + 5]
        if seg and float(np.mean(seg)) >= thresh:
            return y
    return None


def main():
    # flatten candidates, dedupe, remember bucket
    bucket_of, order = {}, []
    for bucket, terms in CANDIDATES.items():
        for t in terms:
            t = t.strip()
            if t.lower() not in bucket_of:
                bucket_of[t.lower()] = bucket
                order.append(t)
    print(f"Validating {len(order)} candidate tripwires against Google Ngrams "
          f"({CORPUS}, pre={PRE_LO}-{PRE_HI}, post={POST_LO}-{POST_HI})...\n")

    # fetch all timeseries (batched, polite delay — no parallelism by design)
    series = {}
    for i in range(0, len(order), BATCH):
        batch = order[i:i + BATCH]
        for rec in fetch_ngrams(batch):
            key = rec["ngram"].split(" (")[0].strip().lower()  # collapse "X (All)" + case variants
            ts = rec.get("timeseries", [])
            if key not in series or sum(ts) > sum(series[key] or [0]):
                series[key] = ts
        print(f"  fetched {min(i+BATCH, len(order))}/{len(order)}")
        time.sleep(1.3)

    kept, rejected = [], []
    for term in order:
        ts = series.get(term.lower())
        bucket = bucket_of[term.lower()]
        if not ts:
            rejected.append({"term": term, "bucket": bucket, "reason": "absent from Google Books"})
            continue
        pre_max  = win(ts, PRE_LO, PRE_HI, np.max)
        pre_mean = win(ts, PRE_LO, PRE_HI, np.mean)
        post_mean = win(ts, POST_LO, POST_HI, np.mean)
        ratio = float(np.log10((post_mean + EPS) / (pre_mean + EPS)))
        stat = {"term": term, "bucket": bucket,
                "pre_max": pre_max, "pre_mean": pre_mean, "post_mean": post_mean,
                "log10_post_over_pre": round(ratio, 2), "birth_year": birth_year(ts, post_mean)}
        # decide
        if post_mean < POST_FLOOR:
            stat["reason"] = "too rare after 1945 (post_mean below floor)"
            rejected.append(stat)
        elif pre_mean >= PRE_MEAN_FLOOR:
            stat["reason"] = f"present before 1945 (pre_mean={pre_mean:.1e} >= {PRE_MEAN_FLOOR:.0e})"
            rejected.append(stat)
        elif ratio < RATIO_STRONG:
            stat["reason"] = f"weak shift (log10 ratio {ratio:.2f} < {RATIO_STRONG})"
            rejected.append(stat)
        else:
            if term.lower() in CONTEXT and CONTEXT[term.lower()]:
                stat["context"] = CONTEXT[term.lower()]
            kept.append(stat)

    kept.sort(key=lambda s: -s["log10_post_over_pre"])
    rejected.sort(key=lambda s: s.get("pre_mean", 1.0), reverse=True)

    # write outputs
    OUT.joinpath("markers_1945.json").write_text(json.dumps({
        "boundary": 1945, "corpus": CORPUS,
        "pre_window": [PRE_LO, PRE_HI], "post_window": [POST_LO, POST_HI],
        "thresholds": {"pre_mean_floor": PRE_MEAN_FLOOR, "post_floor": POST_FLOOR,
                       "ratio_strong": RATIO_STRONG},
        "n_candidates": len(order), "n_kept": len(kept),
        "markers": kept,
    }, indent=2))
    OUT.joinpath("markers_rejected.json").write_text(json.dumps(rejected, indent=2))

    # summary
    print(f"\n==== validated post-1945 tripwires ====")
    print(f"candidates : {len(order)}")
    print(f"KEPT       : {len(kept)}")
    print(f"rejected   : {len(rejected)}\n")
    print(f"{'term':<28}{'bucket':<28}{'birth':>6}{'pre_mean':>11}{'post_mean':>11}{'log10':>8}")
    for s in kept:
        print(f"{s['term']:<28}{s['bucket']:<28}{str(s['birth_year']):>6}"
              f"{s['pre_mean']:>11.1e}{s['post_mean']:>11.1e}{s['log10_post_over_pre']:>8.2f}")
    print(f"\n--- rejected (top by pre-1945 presence; should include the neg controls) ---")
    for s in rejected[:22]:
        pm = s.get("pre_mean")
        pm = f"{pm:.1e}" if isinstance(pm, float) else "  n/a"
        print(f"  {s['term']:<26} pre_mean={pm:<10} {s['reason']}")
    print(f"\nsaved -> {OUT/'markers_1945.json'}  and  markers_rejected.json")


if __name__ == "__main__":
    main()
