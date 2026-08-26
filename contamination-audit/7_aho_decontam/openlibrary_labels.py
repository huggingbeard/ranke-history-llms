"""
openlibrary_labels.py — CLEAN ground truth from an external authority.

The dataset's own `year` is a cataloging/scan/reprint date and is wrong for many books (proven:
18/29 residual 'misses' are old books mislabeled modern). OpenLibrary's WORK-level
`first_publish_year` is the true content-era publication date (a 1966 reprint of an 1866 book
resolves to 1866). We match every book by its LCCN (exact, self-verifying: OpenLibrary echoes the
LCCN back), and record the authoritative first-publication year alongside the dataset metadata.

Politeness (this hits a free nonprofit API): SEQUENTIAL only (no parallelism), descriptive
User-Agent, ~2 req/s, exponential backoff on 429/errors, checkpoint every 100, fully resumable.

Output ol_labels.json: {lccn: {meta_year, ol_year, ol_title, how}}
  how = 'lccn-exact' (our lccn in the returned record) | 'lccn-loose' (top hit, lccn not echoed)
        | 'nomatch' | 'httpNNN'/'err'
"""
import json, sys, time, urllib.request, urllib.parse, urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
OUT = HERE / "ol_labels.json"
UA = {"User-Agent": "ranke-contamination-research/1.0 (academic use; joachimvoth@gmail.com)"}
SLEEP = 0.5

d = json.loads((HERE / "features_v2.json").read_text(encoding="utf-8"))
ROWS = d["modern"] + d["old"]                       # each has id (=lccn) and y (=metadata year)


def ol(lccn):
    q = urllib.parse.urlencode({"q": f"lccn:{lccn}", "fields": "first_publish_year,lccn,title", "limit": 3})
    url = "https://openlibrary.org/search.json?" + q
    for attempt in range(5):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45))
            docs = r.get("docs", [])
            for x in docs:
                if lccn in (x.get("lccn") or []):
                    return x.get("first_publish_year"), x.get("title"), "lccn-exact"
            if docs:
                return docs[0].get("first_publish_year"), docs[0].get("title"), "lccn-loose"
            return None, None, "nomatch"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1)); continue
            return None, None, f"http{e.code}"
        except Exception:
            time.sleep(2 * (attempt + 1)); continue
    return None, None, "err"


def main():
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    todo = [r for r in ROWS if r["id"] not in done]
    print(f"{len(ROWS)} books | already done {len(done)} | to do {len(todo)}", flush=True)
    for i, r in enumerate(todo):
        y, t, how = ol(r["id"]); time.sleep(SLEEP)
        done[r["id"]] = {"meta_year": r["y"], "ol_year": y, "ol_title": t, "how": how}
        if (i + 1) % 100 == 0:
            OUT.write_text(json.dumps(done), encoding="utf-8")
            m = sum(1 for v in done.values() if v["ol_year"] is not None)
            cross = sum(1 for v in done.values() if v["ol_year"] is not None
                        and (v["ol_year"] <= 1945) != (v["meta_year"] <= 1945))
            print(f"  {len(done)}/{len(ROWS)} | matched {m} | metadata crosses-1945 vs OL: {cross}",
                  file=sys.stderr, flush=True)
    OUT.write_text(json.dumps(done), encoding="utf-8")
    m = sum(1 for v in done.values() if v["ol_year"] is not None)
    print(f"\nDONE: {len(done)} looked up | matched {m} ({100*m/len(done):.1f}%)")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
