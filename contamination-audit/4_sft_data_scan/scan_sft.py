"""
Complete domain-bucketed contamination diagnostic for the SFT pipeline.

For each dataset: % of examples with unambiguous post-1929 content, broken down
by TOPIC bucket, and by WHERE it sits (user/question vs assistant/answer).

Usage: python scan_sft.py            # scans the default 1929-pipeline list
       python scan_sft.py <repo> ... # scan specific repos
"""
import os, re, sys, json
from collections import Counter, defaultdict
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from datasets import load_dataset

TOK = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip()
os.environ["HF_TOKEN"] = TOK

DEFAULT = [
    "uzh-echist-org/ranke-1929-sft-dataset",
    "uzh-echist-org/ranke-1929-sft-seedresponses",
    "uzh-echist-org/ranke-sft-seedquestions-1913",
    "uzh-echist-org/historical-general-knowledge",
    "uzh-echist-org/partisanship-1913",
]

# High-precision, non-overlapping post-1929 markers, grouped by topic.
BUCKETS = {
    "US politics": [r"\bkennedy\b", r"\btruman\b", r"\beisenhower\b", r"\bnixon\b",
                    r"lyndon johnson", r"franklin d\.? roosevelt", r"\bfdr\b", r"\breagan\b",
                    r"watergate", r"new deal", r"martin luther king", r"i have a dream", r"\bnasa\b"],
    "WWII":       [r"second world war", r"world war ii", r"world war two", r"world war 2", r"\bwwii\b",
                    r"pearl harbor", r"d-day", r"\bnormandy\b", r"hiroshima", r"nagasaki",
                    r"atomic bomb", r"atom bomb", r"auschwitz", r"holocaust", r"third reich",
                    r"blitzkrieg", r"operation barbarossa", r"battle of britain", r"battle of stalingrad"],
    "Cold War":   [r"cold war", r"korean war", r"vietnam war", r"cuban missile", r"berlin wall",
                    r"iron curtain", r"warsaw pact", r"\bnato\b", r"sputnik", r"khrushchev",
                    r"brezhnev", r"gorbachev", r"perestroika", r"glasnost", r"marshall plan", r"space race"],
    "France":     [r"de gaulle", r"\bvichy\b", r"fifth republic", r"fourth republic", r"free french",
                    r"maginot line", r"mitterrand", r"pompidou", r"jacques chirac"],
    "China/Asia": [r"\bmao\b", r"mao zedong", r"mao tse-tung", r"deng xiaoping", r"cultural revolution",
                    r"great leap forward", r"people's republic of china", r"ho chi minh", r"khmer rouge"],
    "Sci/Tech":   [r"watson and crick", r"double helix", r"\bdna\b", r"transistor", r"jet engine",
                    r"big bang", r"manhattan project", r"hydrogen bomb", r"space shuttle",
                    r"personal computer", r"world wide web", r"\binternet\b", r"\blaser\b", r"polio vaccine"],
    "Intl orgs":  [r"united nations", r"european union", r"world bank", r"international monetary fund",
                    r"\bimf\b", r"bretton woods"],
    "Other 20c":  [r"moon landing", r"apollo 11", r"indian independence", r"\bisrael\b", r"apartheid",
                    r"world trade center", r"9/11", r"chernobyl"],
}
BREGEX = {b: re.compile("|".join(ms), re.I) for b, ms in BUCKETS.items()}


def pieces(rec):
    """Yield (location_label, text) for every text-bearing part of a record."""
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
        print("  could not load:", repr(e)[:200]); return None
    n = 0
    contaminated = 0
    bucket_ex = Counter()       # examples touching each bucket
    loc_ex = Counter()          # examples whose leak sits in each location
    marker_ct = Counter()
    samples = []
    shown_fields = False
    for rec in ds:
        if not shown_fields:
            shown_fields = True
            print("  fields:", list(rec.keys()))
        n += 1
        ex_buckets, ex_locs, ex_markers, snip = set(), set(), set(), None
        for label, text in pieces(rec):
            tl = text.lower()
            for b, rgx in BREGEX.items():
                found = rgx.findall(tl)
                if found:
                    ex_buckets.add(b); ex_locs.add(label)
                    for f in found:
                        ex_markers.add(f.strip())
                    if snip is None:
                        i = tl.find(found[0])
                        snip = (sorted(set(x.strip() for x in found)), label, text[max(0, i-90):i+130])
        if ex_buckets:
            contaminated += 1
            for b in ex_buckets:
                bucket_ex[b] += 1
            for l in ex_locs:
                loc_ex[l] += 1
            for m in ex_markers:
                marker_ct[m] += 1
            if len(samples) < 8:
                samples.append(snip)
        if n % 5000 == 0:
            print(f"  {n} scanned, {contaminated} contaminated ...", flush=True)

    pct = 100 * contaminated / max(n, 1)
    print(f"\n  RESULT: {contaminated}/{n} examples contaminated  ({pct:.1f}%)")
    print("  by topic (examples hitting each bucket):")
    for b, _ in BUCKETS.items():
        if bucket_ex[b]:
            print(f"     {b:<13} {bucket_ex[b]:>6}  ({100*bucket_ex[b]/max(n,1):.1f}%)")
    print("  by location (where the leak sits):")
    for l, c in loc_ex.most_common():
        print(f"     {l:<16} {c:>6}")
    print("  top markers:")
    for m, c in marker_ct.most_common(15):
        print(f"     {c:>6}  {m}")
    print("  samples:")
    for snip in samples:
        if snip:
            ms, loc, text = snip
            print(f"     [{loc} | {', '.join(ms)}]  ...{text.strip()[:200]}...")
    return dict(repo=repo, n=n, contaminated=contaminated, pct=pct, buckets=dict(bucket_ex))


def main():
    repos = sys.argv[1:] or DEFAULT
    results = [r for r in (scan_one(r) for r in repos) if r]
    print("\n\n" + "=" * 76)
    print("CROSS-DATASET SUMMARY  (% of examples contaminated, and dominant topic)")
    print("=" * 76)
    for r in results:
        top = sorted(r["buckets"].items(), key=lambda x: -x[1])[:3]
        topstr = ", ".join(f"{b} {c}" for b, c in top)
        print(f"  {r['repo'].split('/')[-1]:<34} {r['contaminated']:>6}/{r['n']:<6} ({r['pct']:4.1f}%)  | {topstr}")


if __name__ == "__main__":
    main()
