"""
Disambiguated post-1929 contamination scan (v2): markers that are common as
pre-1929 place/surnames (Normandy, Israel, Kennedy, Nagasaki, Reagan, Vichy...)
require a CONTEXT term in the same text piece, killing the false positives that
swamped the first pass. Also counts the LITERAL facts the Instruct model emitted,
to tell leakage apart from confabulation.

Usage: python scan_sft2.py [repo ...]
"""
import os, re, sys, json
from collections import Counter
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from datasets import load_dataset

TOK = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip()
os.environ["HF_TOKEN"] = TOK

DEFAULT = [
    "uzh-echist-org/ranke-1929-sft-dataset",
    "uzh-echist-org/ranke-1929-sft-seedresponses",
    "uzh-echist-org/ranke-1913-sft-dataset",
    "uzh-echist-org/ranke-1913-sft-seedresponses",
    "uzh-echist-org/sft-conversations-1913",
    "uzh-echist-org/sft-bootstrap-1913",
    "uzh-echist-org/ranke-sft-seedquestions-1913",
]

# (bucket, marker_regex, context_regex_or_None)  -- None = unambiguous, count alone
M = [
    ("US politics", r"\btruman\b", r"harry|president|1945|1948|1949|doctrine|atomic"),
    ("US politics", r"\bkennedy\b", r"john f|j\. f\.|jfk|president|assassinat|196\d|cuban"),
    ("US politics", r"\beisenhower\b", r"dwight|president|general|allied|1944|1945|195\d"),
    ("US politics", r"\bnixon\b", r"richard|president|watergate|196\d|197\d"),
    ("US politics", r"\breagan\b", r"ronald|president|198\d"),
    ("US politics", r"lyndon johnson", None), ("US politics", r"franklin d\.? roosevelt", None),
    ("US politics", r"\bfdr\b", None), ("US politics", r"martin luther king", None),
    ("US politics", r"i have a dream", None), ("US politics", r"\bnasa\b", None),
    ("US politics", r"new deal", r"roosevelt|193\d|depression|fdr"),
    ("WWII", r"second world war", None), ("WWII", r"world war (?:ii|two|2)", None),
    ("WWII", r"\bwwii\b", None), ("WWII", r"pearl harbor", r"1941|attack|japanese|bomb"),
    ("WWII", r"\bnormandy\b", r"1944|d-day|allied|landing|invasion|second world war|nazi|overlord|eisenhower"),
    ("WWII", r"hiroshima", r"atomic|bomb|1945|nuclear"), ("WWII", r"nagasaki", r"atomic|bomb|1945|nuclear"),
    ("WWII", r"atomic bomb", None), ("WWII", r"atom bomb", None), ("WWII", r"auschwitz", None),
    ("WWII", r"holocaust", r"jewish|nazi|genocide|jews|six million|auschwitz"),
    ("WWII", r"third reich", None), ("WWII", r"blitzkrieg", None),
    ("WWII", r"operation barbarossa", None), ("WWII", r"battle of stalingrad", None),
    ("Cold War", r"cold war", None), ("Cold War", r"korean war", None), ("Cold War", r"vietnam war", None),
    ("Cold War", r"cuban missile", None), ("Cold War", r"berlin wall", None), ("Cold War", r"iron curtain", None),
    ("Cold War", r"warsaw pact", None), ("Cold War", r"\bnato\b", None), ("Cold War", r"sputnik", None),
    ("Cold War", r"khrushchev", None), ("Cold War", r"brezhnev", None), ("Cold War", r"gorbachev", None),
    ("Cold War", r"perestroika", None), ("Cold War", r"glasnost", None), ("Cold War", r"marshall plan", None),
    ("France", r"de gaulle", None), ("France", r"\bvichy\b", r"france|regime|government|p.tain|occupied|1940|nazi"),
    ("France", r"fourth republic", None), ("France", r"fifth republic", None),
    ("France", r"free french", None), ("France", r"maginot line", None),
    ("France", r"mitterrand", None), ("France", r"pompidou", None),
    ("China/Asia", r"\bmao\b", r"zedong|tse-tung|china|chinese|communist|chairman"),
    ("China/Asia", r"deng xiaoping", None), ("China/Asia", r"cultural revolution", r"china|chinese|mao"),
    ("China/Asia", r"great leap forward", None), ("China/Asia", r"people's republic of china", None),
    ("China/Asia", r"ho chi minh", None),
    ("Sci/Tech", r"watson and crick", None), ("Sci/Tech", r"double helix", None),
    ("Sci/Tech", r"transistor", None), ("Sci/Tech", r"jet engine", None),
    ("Sci/Tech", r"manhattan project", None), ("Sci/Tech", r"hydrogen bomb", None),
    ("Sci/Tech", r"space shuttle", None), ("Sci/Tech", r"personal computer", None),
    ("Sci/Tech", r"world wide web", None), ("Sci/Tech", r"\binternet\b", None), ("Sci/Tech", r"polio vaccine", None),
    ("Intl orgs", r"united nations", None), ("Intl orgs", r"european union", None),
    ("Intl orgs", r"bretton woods", None), ("Intl orgs", r"\bimf\b", None),
    ("Other 20c", r"moon landing", None), ("Other 20c", r"apollo 11", None),
    ("Other 20c", r"\bisrael\b", r"1948|state of israel|israeli|zionis|ben-gurion|netanyahu|six-day|arab"),
    ("Other 20c", r"apartheid", None), ("Other 20c", r"world trade center", None), ("Other 20c", r"chernobyl", None),
]
COMP = [(b, re.compile(p, re.I), (re.compile(c, re.I) if c else None)) for b, p, c in M]

# literal facts the Instruct model actually produced -> are they in the data at all?
EXACT = ["john f. kennedy", "john fitzgerald kennedy", "harry s. truman", "harry truman",
         "president kennedy", "president truman", "president nixon", "second world war", "world war ii"]


def pieces(rec):
    out = []
    for k, v in rec.items():
        if isinstance(v, str):
            out.append((k, v))
        elif isinstance(v, list):
            for it in v:
                if isinstance(it, dict) and "content" in it:
                    out.append(("role:" + str(it.get("role", "?")), str(it.get("content", ""))))
                elif isinstance(it, str):
                    out.append((k, it))
        elif isinstance(v, dict) and "content" in v:
            out.append(("role:" + str(v.get("role", "?")), str(v["content"])))
    if not out:
        out.append(("_record", json.dumps(rec, default=str, ensure_ascii=False)))
    return out


def scan_one(repo):
    print("\n" + "#" * 76 + f"\n# {repo}\n" + "#" * 76, flush=True)
    try:
        ds = load_dataset(repo, split="train", streaming=True, token=TOK)
    except Exception as e:
        print("  could not load:", repr(e)[:160]); return None
    n = contaminated = 0
    bucket_ex, loc_ex, marker_ct, exact_ct = Counter(), Counter(), Counter(), Counter()
    samples = []
    for rec in ds:
        n += 1
        ps = pieces(rec)
        ex_b, ex_l, ex_m, snip = set(), set(), set(), None
        for label, text in ps:
            tl = text.lower()
            for b, mrx, crx in COMP:
                mt = mrx.search(tl)
                if mt and (crx is None or crx.search(tl)):
                    ex_b.add(b); ex_l.add(label); ex_m.add(mt.group(0).strip())
                    if snip is None:
                        i = mt.start()
                        snip = (b, mt.group(0).strip(), label, text[max(0, i-90):i+140])
            for e in EXACT:
                if e in tl:
                    exact_ct[e] += 1
        if ex_b:
            contaminated += 1
            for b in ex_b: bucket_ex[b] += 1
            for l in ex_l: loc_ex[l] += 1
            for m in ex_m: marker_ct[m] += 1
            if len(samples) < 10: samples.append(snip)
        if n % 10000 == 0:
            print(f"  {n} scanned, {contaminated} flagged ...", flush=True)
    pct = 100 * contaminated / max(n, 1)
    print(f"\n  TRUE contaminated: {contaminated}/{n}  ({pct:.2f}%)")
    if bucket_ex:
        print("  by topic:", "  ".join(f"{b}:{c}" for b, c in bucket_ex.most_common()))
        print("  by location:", "  ".join(f"{l}:{c}" for l, c in loc_ex.most_common()))
        print("  markers:", "  ".join(f"{m}:{c}" for m, c in marker_ct.most_common(12)))
    print("  LITERAL model-output facts present:",
          ("  ".join(f"{e}:{c}" for e, c in exact_ct.most_common()) if exact_ct else "NONE"))
    for s in samples:
        if s:
            b, m, loc, t = s
            print(f"     [{b}|{m}|{loc}] ...{t.strip()[:170]}...")
    return dict(repo=repo, n=n, contaminated=contaminated, pct=pct,
                buckets=dict(bucket_ex), exact=dict(exact_ct))


def main():
    repos = sys.argv[1:] or DEFAULT
    res = [r for r in (scan_one(r) for r in repos) if r]
    print("\n\n" + "=" * 76 + "\nCROSS-DATASET SUMMARY (disambiguated)\n" + "=" * 76)
    for r in res:
        top = ", ".join(f"{b} {c}" for b, c in sorted(r["buckets"].items(), key=lambda x: -x[1])[:3]) or "clean"
        ex = sum(r["exact"].values())
        print(f"  {r['repo'].split('/')[-1]:<32} {r['contaminated']:>5}/{r['n']:<6} ({r['pct']:4.2f}%) "
              f"| {top} | literal-facts:{ex}")


if __name__ == "__main__":
    main()
