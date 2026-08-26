"""
aho_scan.py — fast post-1945 contamination scanner built on Aho-Corasick.

Architecture (the key design point):
  Aho-Corasick is a single-pass, multi-pattern substring matcher. It finds EVERY occurrence
  of ALL markers in one left-to-right pass over the text, in time O(len(text) + #hits) that is
  INDEPENDENT of the number of patterns. That makes it the ideal cheap first-pass filter: it
  rejects the 99.9% of clean text instantly. But a bare substring match LIES (see
  find_contamination.py: "Normandy"=1066, "Israel"=Bible). So every raw hit is then verified:
    1. WORD BOUNDARIES — the chars flanking the match must be non-alphanumeric, so "laser"
       does not fire inside another token and "nato" does not fire inside "donato".
    2. CONTEXT DISAMBIGUATION — for hook-prone markers we require a context regex to also
       appear in the document (ported from find_contamination.py). This runs ONLY on the rare
       documents that already produced a candidate hit, so it costs almost nothing.

Text is lowercased and whitespace-collapsed before matching, so OCR line breaks
("cold\nwar") and double spaces still match — important for real book OCR.

Usage:
  # scan real books streamed from HuggingFace, write flagged + summary (resume-safe):
  python aho_scan.py --repo storytracer/LoC-PD-Books --limit 2000 --out flagged_sample.jsonl
  # the engine (build_automaton / scan_text) is also imported by benchmark.py.
"""
import argparse, json, os, re, sys, time
from collections import Counter
from pathlib import Path

import ahocorasick

HERE = Path(__file__).resolve().parent
WS = re.compile(r"\s+")
# Join words split by an OCR hyphen, killing word-split false positives like
# "agri-cultural revolution" and "virei-nato". Two cases:
#   ¬ (U+00AC not-sign) and ­ (U+00AD soft hyphen) are NEVER legitimate characters -- they
#     are pure OCR/typographic hyphens -- so join across them always (newline or just space).
#   ASCII "-" is ambiguous (real compounds), so only join it across an actual line break,
#     leaving within-line "pre- and post-war" alone.
DEHYPH_SOFT = re.compile(r"(\w)[¬­]\s*(\w)")        # OCR hyphen: join across any whitespace
DEHYPH_HARD = re.compile(r"(\w)-\s*\n\s*(\w)")       # ascii hyphen: only across a line break
CONTEXT_WINDOW = 200  # context disambiguation is checked within +-this many chars of a hit

# Markers that PASSED the Ngram frequency test but are unusable on OCR'd pre-1945 books:
#   civil rights act -> a genuine pre-1945 referent (Civil Rights Acts of 1866 / 1875)
#   motel            -> "hotel" mis-OCRs to "motel" (h->m) constantly, and "hotel" is
#                       everywhere in old books (e.g. "Eastman Motel" = the Eastman Hotel)
#   nasa             -> biggest homograph leak (Hebrew root "nasa'"=lift/stand; French "bois de la
#                       nasa"). investigate.py: space context appears ~equally in old (10%) and
#                       modern (13%) fires, so a guard CANNOT separate them -- drop, don't fake a fix.
EXCLUDE = {"civil rights act", "motel", "nasa"}

# Internet Archive / digitization boilerplate. These books are pre-1929, so ANY mention of
# "internet archive" / archive.org / a URL is a MODERN digitization stamp the scanner added,
# NOT the book's text. Strip it before scanning, or it dominates the false positives.
# Must be OCR-ROBUST: the stamp is itself OCR'd and badly garbled, e.g.
# "digitized by tine internet arciiive in 2011 witii funding from tine library of congress".
# So match the whole block from "digiti[sz]ed by" to "library of congress" (any garbling
# between), and also kill any "internet arc..." fragment (archive however misspelled).
IA_BOILER = re.compile(r"digiti[sz]ed by\b.{0,170}?library of congress", re.I | re.S)
URLS = re.compile(r"https?://\S+|www\.\S+|\S*archive\.org\S*", re.I)
IA_PHRASE = re.compile(r"\binternet\s+arc\w*", re.I)  # "internet archive/arciiive/arcinive..."
# Modern PRESERVATION stamp scanned onto old books: "treatment date: august 2005 ...
# preservationtechnologies a world leader in collections care". Its 2005 would otherwise count
# as a post-1945 'year'. Strip the year-bearing part (the company tokens carry no year).
PRESERV = re.compile(r"treatment\s*date\s*:?\s*[a-z]*\.?\s*20[0-2]\d", re.I)

# --- post-1945 year detection ------------------------------------------------
POST1945 = re.compile(r"\b(?:19(?:4[6-9]|[5-9]\d)|20[0-2]\d)\b")   # a recent-looking 4-digit number
ANYDATE  = re.compile(r"\b(?:1[6-9]\d\d|20[0-2]\d)\b")             # any year 1600-2029 (for ranges/lists)
# cues that mark a number as an actual DATE rather than a quantity/code/page number
_DATECUE = re.compile(r"©|copyright|\(c\)|cop\.|print|publish|\bed\.|edition|issued|revis|dated|"
                      r"\bin\b|\bby\b|\bca\.?|circa|\bc\.|\bvol|\bno\.|\bp{1,2}\.|"
                      r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", re.I)


# ----------------------------------------------------------------------------- engine
# A few VALIDATED markers are genuine homographs that would false-flag real pre-1945 books.
# Require disambiguating context (checked over the whole doc, so any genuine post-1945 use
# trips it; the innocent pre-1945 sense does not):
#   lsd           -> the drug, NOT "£sd" (pounds/shillings/pence) ubiquitous in old British texts
#   iron curtain  -> the Cold War, NOT a theatre fireproof safety curtain
#   common market -> the EEC, NOT an ordinary marketplace
# Context-disambiguation guards: a marker with a guard fires only if its topic regex also
# appears WITHIN +-CONTEXT_WINDOW chars of the hit. Each surface form here is a genuine
# homograph or OCR-collision found in real LoC-PD-Books OCR -- without the guard it false-flags
# innocent pre-1945 books. Alternatives are WORD-BOUNDED (\b) so "nation" does not match inside
# "emanation" (which would wrongly satisfy a guard).
CONTEXT_OVERRIDES = {
    # £sd money / Lazarus etc. vs the drug:
    "lsd": r"\bacid\b|\bdrug|hallucinogen|psychedelic|diethylamide|\btrip\b|hippie|\bdose\b|narcotic",
    # theatre fireproof curtain vs the Cold War:
    "iron curtain": r"soviet|stalin|communis|moscow|churchill|western|eastern|\beurope|cold war|\bbloc\b|berlin",
    # an ordinary marketplace / "common market value" vs the EEC:
    "common market": r"\beec\b|economic communit|\btreaty|brussels|tariff|member state|\brome\b|maastricht|\beuropean union",
    # literal/poetic/Kabbalah "the third world" vs the geopolitical Third World (~1952).
    # NB: bare "nation(s)" deliberately omitted -- it appears in biblical "scattered the
    # nations" text; genuine Third World usage always also has develop/countries/africa near.
    "third world": r"\bdevelop|\bcountr|\bpovert|cold war|nonalign|underdevelop|\baid\b|"
                   r"\bafrica|\basia|latin america|imperialis|colonial",
    # Italian "nato" (=born) / Spanish "vireinato" vs the alliance:
    "nato": r"atlantic|alliance|\btreaty|warsaw|brussels|\bpact\b|cold war|soviet|"
            r"defen[cs]e|military|\bmember|nuclear|\ballies\b",
    # IA boilerplate "internet arc.." / OCR of "internal" vs the actual Internet:
    "internet": r"\bweb\b|world wide web|online|\bnetwork|computer|digital|cyber|e-?mail|website|www|protocol|\bmodem",
    # OCR of "later" / "lazar" (leper) / Middle-English "leisure" vs the optical device:
    "laser": r"\bbeam|\blight|optic|\bruby\b|coherent|\bmaser|wavelength|radiation|surgery|infrared|photon|\bpulse|diode",
    # 1922 radio call-sign "KGB" vs the Soviet agency:
    "kgb": r"soviet|russian|secret police|intelligence|moscow|\bspy\b|espionage|state security|cheka|stasi",
    # descriptive-geometry "double helix" (two helices) vs the DNA structure (1953).
    # NB: "\bgene" omitted -- it matched "generated" ("helices, generated by"); use genetic/genes.
    "double helix": r"\bdna\b|nucleic|genetic|\bgenes?\b|molecul|watson|crick|\bstrand|base pair|chromosom",
    # an 1821 settler named Richard Nixon vs the president:
    "richard nixon": r"president|watergate|196\d|197\d|vietnam|kennedy|vice.?president|white house|impeach",
    # the English idiom "a great leap forward" vs Mao's policy (1958):
    "great leap forward": r"china|chinese|\bmao\b|commune|195\d|1960|famine|peasant|backyard",
    # a man named Martin Luther King (b.1882) in a genealogy vs the civil-rights leader:
    "martin luther king": r"civil rights|i have a dream|montgomery|selma|nonviolen|\bdream\b|reverend|196\d|assassinat|boycott|atlanta",
    # ships that "rock and roll" on the sea vs the music genre:
    "rock and roll": r"\bmusic|\bsong|\bband\b|elvis|guitar|\bdance|teenager|record|radio|rhythm",
    # "a european community" (literal) vs the EC/EEC:
    "european community": r"\beec\b|economic communit|\btreaty|brussels|member state|common market|maastricht|commission|single market",
    # French/Spanish "détente" (a pre-1945 diplomatic thaw, or Spanish "stop yourself") vs the
    # Cold-War thaw. investigate.py: 0% of old fires have Cold-War context, 57% of modern do.
    "detente": r"cold war|soviet|superpower|moscow|nixon|\barms\b|east-west|nuclear|washington|\brelations",
}

# REJECTED candidates re-admitted as guarded post-1945 markers, where investigate.py PROVED the
# post-1945 sense separates cleanly from the old sense (old fires ~0% with context; modern high):
#   holocaust -> the Nazi genocide. Old "holocaust" (a burnt offering / catastrophe) had 0% Nazi
#                context in 373 old fires; 32% of 78 modern fires do -> recall at zero old-book cost.
READMIT = {
    "holocaust": r"jew|nazi|genocid|hitler|six million|exterminat|auschwitz|concentration camp|third reich",
}


def load_markers(path=HERE / "markers_1945.json"):
    """Return (markers:list[str], contexts:dict[str,compiled_regex])."""
    data = json.loads(Path(path).read_text())
    markers, contexts = [], {}
    for m in data["markers"]:
        term = m["term"].lower()
        if term in EXCLUDE:
            continue
        markers.append(term)
        if m.get("context"):
            contexts[term] = re.compile(m["context"], re.I)
    for term, cx in CONTEXT_OVERRIDES.items():
        if term in markers:
            contexts[term] = re.compile(cx, re.I)
    for term, cx in READMIT.items():          # salvaged rejects: add the term AND its required guard
        if term not in markers:
            markers.append(term)
        contexts[term] = re.compile(cx, re.I)
    return markers, contexts


def build_automaton(markers):
    A = ahocorasick.Automaton()
    for term in markers:
        A.add_word(term, term)
    A.make_automaton()
    return A


def normalize(text):
    """Join line-break hyphenations, lowercase, collapse whitespace (robust to OCR line
    breaks), and strip digitization boilerplate (IA stamp / URLs) that would otherwise
    masquerade as 'internet' content.

    Speed: each regex pass over a whole book is expensive, so we (a) collapse whitespace with
    str.split()/join instead of a regex (C-level, ~10x faster than re.sub), and (b) gate each
    remaining regex behind a cheap substring pre-check, so the pass only runs when its trigger
    is actually present. Behaviour is identical to the all-passes version on any text."""
    if "¬" in text or "­" in text:        # OCR not-sign / soft hyphen (rare)
        text = DEHYPH_SOFT.sub(r"\1\2", text)
    if "-" in text:                                  # hyphen present -> maybe a line-break split
        text = DEHYPH_HARD.sub(r"\1\2", text)
    t = " ".join(text.lower().split())               # lowercase + collapse whitespace, fast
    if "digiti" in t or "internet arc" in t:         # IA digitization stamp
        t = IA_BOILER.sub(" ", t)
        t = IA_PHRASE.sub(" ", t)
    if "http" in t or "www." in t or "archive.org" in t:
        t = URLS.sub(" ", t)
    if "treatment date" in t:
        t = PRESERV.sub(" ", t)
    return t


def post1945_years(norm, require_context=False, front_frac=None):
    """Distinct post-1945 years printed in the text.
    require_context=True keeps only years that look like a real DATE -- preceded by a cue
      (©, copyright, a month, 'in', 'vol.'), inside parentheses, or part of a year list/range --
      rather than a stray number / page no. / catalog code.
    front_frac=0.2 restricts the search to the first 20% (front matter), where a copyright or
      publication year normally sits; a recent number deep in the body is rarely a date."""
    region = norm if not front_frac else norm[:max(2500, int(len(norm) * front_frac))]
    if not require_context:
        return set(POST1945.findall(region))
    out = set()
    for m in POST1945.finditer(region):
        s, e = m.start(), m.end()
        before = region[max(0, s - 18):s]; after = region[e:e + 10]
        in_parens = "(" in region[max(0, s - 4):s] and ")" in after[:5]
        near_year = ANYDATE.search(region[max(0, s - 8):s]) or ANYDATE.search(after[:8])
        if _DATECUE.search(before) or in_parens or near_year:
            out.add(m.group(0))
    return out


def _alnum(ch):
    return ch.isalnum()


def scan_text(norm, automaton, contexts):
    """Scan already-normalized text. Return list of (marker, start_idx) that pass
    word-boundary + context checks. Aho-Corasick yields (end_idx, marker)."""
    raw = []
    for end, term in automaton.iter(norm):
        start = end - len(term) + 1
        before = norm[start - 1] if start > 0 else " "
        after = norm[end + 1] if end + 1 < len(norm) else " "
        if _alnum(before) or _alnum(after):
            continue  # inside a larger token -> not a real word hit
        raw.append((term, start, end))
    if not raw:
        return []
    # context disambiguation (only for markers that carry a context regex), checked in a
    # window AROUND the hit -- not the whole doc, which is too loose (a book of sonnets has
    # "nation" *somewhere*, which would wrongly satisfy the "third world" guard).
    out = []
    for term, start, end in raw:
        cx = contexts.get(term)
        if cx is None:
            out.append((term, start)); continue
        lo, hi = max(0, start - CONTEXT_WINDOW), min(len(norm), end + 1 + CONTEXT_WINDOW)
        if cx.search(norm[lo:hi]):
            out.append((term, start))
    return out


def snippet(norm, start, term, pad=80):
    i, j = max(0, start - pad), min(len(norm), start + len(term) + pad)
    return ("..." + norm[i:j] + "...").strip()


# ----------------------------------------------------------------------------- HF streaming scan
def doc_text(rec):
    """LoC-PD-Books text field is 'full_text'; fall back to any large string field."""
    for f in ("full_text", "text", "content"):
        v = rec.get(f)
        if isinstance(v, str) and v:
            return v
    cand = [v for v in rec.values() if isinstance(v, str)]
    return max(cand, key=len) if cand else ""


def stream_scan(repo, limit, out_path, checkpoint=500):
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from datasets import load_dataset
    tokpath = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt")
    tok = tokpath.read_text().strip() if tokpath.exists() else None
    if tok:
        os.environ["HF_TOKEN"] = tok

    markers, contexts = load_markers()
    A = build_automaton(markers)
    print(f"loaded {len(markers)} markers, {len(contexts)} with context disambiguation")

    # resume
    done = set()
    if Path(out_path).exists():
        for ln in open(out_path, encoding="utf-8"):
            try: done.add(json.loads(ln).get("id"))
            except Exception: pass
        print(f"resume: {len(done)} books already flagged", file=sys.stderr)

    ds = load_dataset(repo, split="train", streaming=True, token=tok)
    out = open(out_path, "a", encoding="utf-8")
    n = flagged = errors = 0
    total_chars = 0
    t_norm = t_match = 0.0            # cumulative preprocessing time, Aho-Corasick match time
    by_marker, by_decade = Counter(), Counter()
    TOTAL_BOOKS = 140000             # for ETA / extrapolation
    REGEX_MBPS = 9.0                 # measured union-regex baseline (NOT run in full here)
    t0 = time.perf_counter()
    try:
        for rec in ds:
            if limit and n >= limit:
                break
            n += 1
            try:
                rid = rec.get("lccn") or rec.get("filename") or rec.get("ocaid") or n
                if rid in done:
                    continue
                ts = time.perf_counter()
                norm = normalize(doc_text(rec))
                tn = time.perf_counter()
                hits = scan_text(norm, A, contexts)
                t_match += time.perf_counter() - tn
                t_norm += tn - ts
                total_chars += len(norm)
                if hits:
                    flagged += 1
                    terms = sorted({t for t, _ in hits})
                    for t in terms: by_marker[t] += 1
                    yr = rec.get("year")
                    if isinstance(yr, int): by_decade[(yr // 10) * 10] += 1
                    t0_, s0_ = hits[0]
                    out.write(json.dumps({
                        "id": rid, "year": yr, "title": rec.get("title"),
                        "n_markers": len(terms), "markers": terms[:25],
                        "snippet": snippet(norm, s0_, t0_)[:300],
                    }) + "\n")
            except Exception as e:
                errors += 1
                if errors <= 5: print(f"  [record {n} error: {e!r}]", file=sys.stderr)
            if n % checkpoint == 0:
                out.flush()
                el = time.perf_counter() - t0
                mb = total_chars / 1e6
                rate = n / el if el else 0
                eta = (TOTAL_BOOKS - n) / rate / 60 if rate else 0
                tproc = t_norm + t_match
                pure = mb / tproc if tproc else 0
                mrate = mb / t_match if t_match else 0
                print(f"  {n}/{TOTAL_BOOKS} books | {flagged} flagged | {mb:,.0f} MB | "
                      f"wall {el/60:.1f}m | proc {tproc/60:.2f}m @ {pure:.0f} MB/s "
                      f"(match {mrate:.0f}) | ETA ~{eta:.0f}m", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"\n[stream ended early at book {n}: {e!r}]", file=sys.stderr)
    out.close()
    dt = time.perf_counter() - t0
    mb = total_chars / 1e6
    tproc = t_norm + t_match
    norm_rate = mb / t_norm if t_norm else 0
    match_rate = mb / t_match if t_match else 0
    regex_match_min = (mb / REGEX_MBPS) / 60
    print("\n" + "=" * 66)
    print(f"FULL SWEEP — {repo}")
    print("=" * 66)
    print(f"books scanned       : {n:,}   (errors skipped: {errors})")
    print(f"flagged             : {flagged:,}  ({100*flagged/max(n,1):.2f}%)")
    print(f"text processed      : {mb:,.0f} MB ({total_chars:,} chars)")
    print(f"total wall-clock    : {dt/60:.1f} min")
    print(f"  preprocessing     : {t_norm/60:.2f} min  @ {norm_rate:.0f} MB/s   (de-hyphenate / clean text)")
    print(f"  Aho-Corasick match: {t_match/60:.2f} min  @ {match_rate:.0f} MB/s   <-- the matcher")
    print(f"  streaming / IO    : {(dt-tproc)/60:.1f} min  (network download)")
    print(f"\ncompare vs the union-regex workflow (NOT run in full; from its measured "
          f"~{REGEX_MBPS:.0f} MB/s match rate):")
    print(f"  regex MATCH would take ~{regex_match_min:.0f} min on the same {mb:,.0f} MB")
    print(f"  -> Aho-Corasick MATCH is ~{(mb/REGEX_MBPS)/max(t_match,1e-9):.1f}x faster")
    print(f"     (preprocessing is shared by both workflows, so it's not the differentiator)")
    print("top markers:")
    for m, c in by_marker.most_common(20):
        print(f"   {c:>6}  {m}")
    if by_decade:
        print("flagged books by stated decade (metadata vs content cross-check):")
        for d, c in sorted(by_decade.items()):
            print(f"   {d}s  {c:>5}")
    print(f"flagged -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="storytracer/LoC-PD-Books")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--out", default=str(HERE / "flagged_sample.jsonl"))
    ap.add_argument("--checkpoint", type=int, default=500)
    args = ap.parse_args()
    stream_scan(args.repo, args.limit, args.out, args.checkpoint)


if __name__ == "__main__":
    main()
