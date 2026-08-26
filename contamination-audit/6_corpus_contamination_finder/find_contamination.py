"""
find_contamination.py  —  grep a (pre-cutoff) training corpus for post-cutoff leakage.

This is the tool to point at the 160B PRETRAINING corpus on server 2. It streams the
corpus, flags documents that contain unambiguous post-1929 entities, and — crucially —
uses CONTEXT DISAMBIGUATION so it does NOT drown in the false positives that fooled
the first keyword pass (Normandy=1066, Israel=the Bible, Kennedy=a 19th-c. surname,
Nagasaki=an old port). It then traces flagged docs back to their source so you can
pull the contaminating source wholesale.

LESSONS BAKED IN (read these, they cost us weeks):
  * A bare keyword count LIES. "Normandy" is a medieval duchy; "Israel" is biblical;
    "Reagan" was a Confederate cabinet member. Always require disambiguating context
    for hook-prone terms, and ALWAYS eyeball the snippet before believing a hit.
  * The confirmed leaks in this model are 20th-c. POLITICAL history (US presidents
    through the 1980s). The CONFIRMED list below is the high-priority grep set.
  * At 160B tokens, over-removing is cheap, keeping dirty docs is not. Flag a whole
    document (and ideally its whole source) on a single solid hit.

USAGE
  # full sweep, write flagged docs + a summary:
  python find_contamination.py --input corpus.jsonl --out flagged.jsonl
  # just grep one term with context (quick look):
  python find_contamination.py --input corpus.jsonl --grep "truman"
  # custom fields / cutoff:
  python find_contamination.py --input corpus.jsonl --text-field content --id-field doc_id --source-field source

CORPUS FORMAT: JSONL, one document per line, with a text field (default "text"),
optional id and source/url fields. Streams — never loads the whole corpus.

(For a 160B-scale run, regex is fine but pyahocorasick is ~10x faster if you have it;
this file stays dependency-free on purpose.)
"""
import argparse, json, re, sys, os
from collections import Counter, defaultdict

# ---- CONFIRMED leaks in Ranke-4B-1929 (highest priority) ---------------------
# proven present in the base model by rank-probe; these are the grep set most likely
# to pull the contaminating source.
CONFIRMED = [
    (r"\btruman\b", r"harry|president|1945|atomic|doctrine"),
    (r"\beisenhower\b", r"dwight|president|general|allied|194\d|195\d"),
    (r"\bkennedy\b", r"john f|j\. ?f\.|jfk|president|assassinat|196\d|cuban"),
    (r"\bnixon\b", r"richard|president|watergate|196\d|197\d"),
    (r"\breagan\b", r"ronald|president|198\d"),
    (r"ho chi minh", None),
    (r"\bmuhammad ali\b", None),
]

# ---- broad disambiguated sweep, grouped by topic -----------------------------
# (bucket, marker_regex, context_regex_or_None).  None = unambiguous, count alone.
MARKERS = [
 ("US politics", r"lyndon johnson", None), ("US politics", r"franklin d\.? roosevelt", r"president|new deal|193\d|194\d"),
 ("US politics", r"\bfdr\b", None), ("US politics", r"martin luther king", None), ("US politics", r"i have a dream", None),
 ("US politics", r"\bnasa\b", None), ("US politics", r"watergate", None),
 ("WWII", r"second world war", None), ("WWII", r"world war (?:ii|two|2)", None), ("WWII", r"\bwwii\b", None),
 ("WWII", r"pearl harbor", r"1941|attack|japanese|bomb"), ("WWII", r"\bnormandy\b", r"1944|d-day|allied|landing|overlord|eisenhower"),
 ("WWII", r"hiroshima", r"atomic|bomb|1945|nuclear"), ("WWII", r"nagasaki", r"atomic|bomb|1945|nuclear"),
 ("WWII", r"atomic bomb", None), ("WWII", r"auschwitz", None), ("WWII", r"holocaust", r"jewish|nazi|jews|genocide|six million"),
 ("WWII", r"third reich", None), ("WWII", r"blitzkrieg", None), ("WWII", r"operation overlord", None),
 ("WWII", r"operation barbarossa", None), ("WWII", r"d-day", None), ("WWII", r"final solution", r"jewish|nazi|jews|exterminat"),
 ("Cold War", r"cold war", None), ("Cold War", r"korean war", None), ("Cold War", r"vietnam war", None),
 ("Cold War", r"cuban missile", None), ("Cold War", r"berlin wall", None), ("Cold War", r"iron curtain", None),
 ("Cold War", r"warsaw pact", None), ("Cold War", r"\bnato\b", None), ("Cold War", r"sputnik", None),
 ("Cold War", r"khrushchev", None), ("Cold War", r"brezhnev", None), ("Cold War", r"gorbachev", None),
 ("Cold War", r"perestroika", None), ("Cold War", r"glasnost", None), ("Cold War", r"marshall plan", None),
 ("France", r"de gaulle", None), ("France", r"\bvichy\b", r"france|regime|government|p.tain|occupied|1940|nazi"),
 ("France", r"fourth republic", None), ("France", r"fifth republic", None), ("France", r"maginot line", None),
 ("China/Asia", r"\bmao\b", r"zedong|tse-tung|china|chinese|communist|chairman"), ("China/Asia", r"deng xiaoping", None),
 ("China/Asia", r"cultural revolution", r"china|chinese|mao"), ("China/Asia", r"great leap forward", None),
 ("China/Asia", r"people's republic of china", None),
 ("Sci/Tech", r"watson and crick", None), ("Sci/Tech", r"double helix", None), ("Sci/Tech", r"\btransistor\b", None),
 ("Sci/Tech", r"jet engine", None), ("Sci/Tech", r"manhattan project", None), ("Sci/Tech", r"hydrogen bomb", None),
 ("Sci/Tech", r"space shuttle", None), ("Sci/Tech", r"\binternet\b", None), ("Sci/Tech", r"world wide web", None),
 ("Sci/Tech", r"polio vaccine", None),
 ("Intl orgs", r"united nations", None), ("Intl orgs", r"european union", None), ("Intl orgs", r"bretton woods", None),
 ("Other 20c", r"moon landing", None), ("Other 20c", r"apollo 11", None), ("Other 20c", r"\bisrael\b", r"1948|state of israel|israeli|zionis|ben-gurion"),
 ("Other 20c", r"apartheid", None), ("Other 20c", r"chernobyl", None), ("Other 20c", r"world trade center", None),
]


def compile_set():
    out = []
    for bucket, mk, cx in MARKERS:
        out.append((bucket, re.compile(mk, re.I), re.compile(cx, re.I) if cx else None))
    conf = [(re.compile(mk, re.I), re.compile(cx, re.I) if cx else None) for mk, cx in CONFIRMED]
    return out, conf


def doc_text(rec, field):
    v = rec.get(field)
    if isinstance(v, str):
        return v
    # fall back: concatenate all string content (handles chat/messages formats)
    parts = []
    def walk(x):
        if isinstance(x, str): parts.append(x)
        elif isinstance(x, dict):
            for vv in x.values(): walk(vv)
        elif isinstance(x, list):
            for vv in x: walk(vv)
    walk(rec)
    return " ".join(parts)


def grep_mode(args):
    pat = re.compile(r"\b" + re.escape(args.grep) + r"\b", re.I)
    n = hits = 0
    for line in open(args.input, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        n += 1
        rec = json.loads(line)
        t = doc_text(rec, args.text_field)
        for m in pat.finditer(t):
            hits += 1
            i = m.start()
            did = rec.get(args.id_field, n)
            print(f"[doc {did}] ...{t[max(0,i-110):i+130].strip()}...")
            if hits >= args.max_grep:
                print(f"(stopped at {args.max_grep} hits; raise --max-grep for more)"); return
    print(f"\nscanned {n} docs, {hits} occurrences of '{args.grep}'")


def main():
    ap = argparse.ArgumentParser(description="Find post-cutoff contamination in a corpus.")
    ap.add_argument("--input", required=True, help="corpus .jsonl (one doc per line)")
    ap.add_argument("--out", default="flagged.jsonl", help="where to write flagged docs")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--id-field", default="id")
    ap.add_argument("--source-field", default="source", help="field naming the doc's origin (to trace & purge)")
    ap.add_argument("--grep", default=None, help="quick mode: just show context for one term")
    ap.add_argument("--max-grep", type=int, default=200)
    ap.add_argument("--checkpoint", type=int, default=5000)
    args = ap.parse_args()

    if args.grep:
        return grep_mode(args)

    markers, confirmed = compile_set()
    # resume
    done = set()
    if os.path.exists(args.out):
        for ln in open(args.out, encoding="utf-8"):
            try: done.add(json.loads(ln).get("id"))
            except Exception: pass
        print(f"resume: {len(done)} docs already flagged", file=sys.stderr)

    out = open(args.out, "a", encoding="utf-8")
    n = flagged = 0
    by_entity, by_bucket, by_source = Counter(), Counter(), Counter()
    for ln in open(args.input, encoding="utf-8"):
        ln = ln.strip()
        if not ln: continue
        n += 1
        rec = json.loads(ln)
        did = rec.get(args.id_field, n)
        if did in done: continue
        t = doc_text(rec, args.text_field); tl = t.lower()
        ents, bks, conf_hit, snip = set(), set(), False, None
        for rgx, cx in confirmed:
            m = rgx.search(tl)
            if m and (cx is None or cx.search(tl)):
                conf_hit = True; ents.add(m.group(0).strip()); bks.add("US politics")
                if snip is None: i=m.start(); snip=t[max(0,i-90):i+140]
        for bucket, rgx, cx in markers:
            m = rgx.search(tl)
            if m and (cx is None or cx.search(tl)):
                ents.add(m.group(0).strip()); bks.add(bucket)
                if snip is None: i=m.start(); snip=t[max(0,i-90):i+140]
        if ents:
            flagged += 1
            src = rec.get(args.source_field, "?")
            for e in ents: by_entity[e]+=1
            for b in bks: by_bucket[b]+=1
            by_source[str(src)] += 1
            out.write(json.dumps({"id": did, "source": src, "confirmed": conf_hit,
                                  "entities": sorted(ents), "buckets": sorted(bks),
                                  "snippet": (snip or "").strip()[:260]}) + "\n")
        if n % args.checkpoint == 0:
            out.flush(); print(f"  {n} scanned, {flagged} flagged ...", file=sys.stderr, flush=True)
    out.close()

    print(f"\n==== {args.input} ====")
    print(f"docs scanned : {n}\nflagged      : {flagged}  ({100*flagged/max(n,1):.2f}%)")
    print("\nby topic bucket:");   [print(f"   {b:<13} {c}") for b,c in by_bucket.most_common()]
    print("\ntop entities (EYEBALL these snippets before trusting):"); [print(f"   {c:>7}  {e}") for e,c in by_entity.most_common(25)]
    print("\ntop sources to consider purging:"); [print(f"   {c:>7}  {s}") for s,c in by_source.most_common(15)]
    print(f"\nflagged docs -> {args.out}")
    print("NEXT: open flagged.jsonl, read snippets, trace the recurring 'source' values,\n"
          "and drop the whole contaminating source; then fix the date filter and retrain.")


if __name__ == "__main__":
    main()
