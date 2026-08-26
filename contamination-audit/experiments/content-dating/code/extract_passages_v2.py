"""Extract passages for the content-dating experiment (random + grep-directed, 4 lengths).

v2 (2026-07-22) — offset-bug fix + certified-negatives filter, run on the office 4090 box.
Files used:
  manifests:  <this repo dir>\\manifest_negatives.csv, manifest_positives.csv
  certification: D:\\...\\content-dating\\intermediates\\certify_negatives_full.csv (filters negatives)
  corpus:     D:\\...\\temp-guohui\\decontamination\\datasets\\history-llms\\sample-10gb\\data
  positives:  D:\\...\\content-dating\\positives_text\\<barcode>.txt
  markers:    <this repo dir>\\markers_1945.json  (copied from Dropbox code\\; hash-matches the
              anachronism-decontaminator repo copy)
  output:     D:\\...\\content-dating\\passages\\passages_neg.parquet / passages_pos.parquet

Changes vs v1 (documented for the PI):
  - BUG FIX: v1 found grep centers on a whitespace-collapsed lowercase string but sliced the ORIGINAL
    text with those offsets — drift of hundreds of chars by mid-book, so short grep passages could
    miss the marker they were aimed at. v2 normalizes ONCE (single-space collapse) and does all
    center-finding and slicing on that same normalized string. (v1 flattened newlines in passages
    anyway, so models see the same kind of text.)
  - grep centers: v1 took the FIRST 16 hits (start-of-book bias); v2 spreads evenly across ALL hits.
  - negatives are restricted to certify_negatives_full.csv verdict == 'certified_clean'.
  - adds a 'center' column (position in the normalized text) for auditability.
Same seed (1929), same lengths {250,500,1000,2000}, same N=16, same marker DROP list as v1.
"""
import os, json
import random
import pandas as pd, pyarrow.parquet as pq, ahocorasick

REPO = os.path.dirname(os.path.abspath(__file__))
DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
CORPUS = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\temp-guohui\decontamination\datasets\history-llms\sample-10gb\data"
CERT = os.path.join(DROPBOX, "intermediates", "certify_negatives_full.csv")
LENGTHS = [250, 500, 1000, 2000]; N = 16; SEED = 1929
MK = os.path.join(REPO, "markers_1945.json")
DROP = {"internet", "nato", "lsd", "common market", "european community", "third world", "nasa", "opec", "laser", "motel"}
markers = [m["term"].lower() for m in json.load(open(MK))["markers"] if m["term"].lower() not in DROP]
A = ahocorasick.Automaton()
for m in markers: A.add_word(m, m)
A.make_automaton()

def preload_corpus_texts(df):
    """One read per shard instead of one per book (32 shard reads, not 918)."""
    txt = {}
    sub = df[df.origin == "local_corpus"] if "shard" in df.columns else df.iloc[0:0]
    if len(sub) == 0: return txt
    for sh, g in sub.groupby("shard"):
        t = pq.read_table(os.path.join(CORPUS, sh), columns=["id", "text"])
        d = dict(zip(t.column("id").to_pylist(), t.column("text").to_pylist()))
        for i in g.id: txt[i] = d.get(i, "")
    return txt

def load_text(row, corpus_cache):
    if row.origin == "local_corpus":
        return corpus_cache.get(row.id, "")
    p = os.path.join(DROPBOX, "positives_text", row.id + ".txt")
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""

def rng(bid): return random.Random(f"{bid}-{SEED}")

def random_centers(norm, r):
    lo, hi = int(len(norm) * 0.05), int(len(norm) * 0.95)
    return [r.randint(lo, hi) for _ in range(N)] if hi > lo else [len(norm) // 2] * N

def grep_centers(low):
    hits = [end - len(term) // 2 for end, term in A.iter(low)]
    if not hits: return []
    if len(hits) <= N: return hits
    step = len(hits) / N
    return [hits[int(i * step)] for i in range(N)]   # spread across ALL hits, not first-16

def passages(norm, centers, L):
    h = L // 2; out = []
    for c in centers[:N]:
        s = max(0, c - h)
        out.append((c, norm[s:s + L].strip()))
    return out

def run(manifest, out_name, id_filter=None):
    df = pd.read_csv(os.path.join(REPO, manifest), dtype={"id": str})
    if id_filter is not None:
        before = len(df); df = df[df.id.isin(id_filter)]
        print(f"{manifest}: {before} -> {len(df)} after certification filter", flush=True)
    corpus_cache = preload_corpus_texts(df)
    rows = []
    for k, (_, row) in enumerate(df.iterrows()):
        tx = load_text(row, corpus_cache)
        if len(tx) < 1000: continue
        norm = " ".join(tx.split())
        low = norm.lower()
        r = rng(row.id)
        rc = random_centers(norm, r)
        gc = grep_centers(low) or rc
        for L in LENGTHS:
            for strat, centers in (("random", rc), ("grep", gc)):
                for j, (c, ps) in enumerate(passages(norm, centers, L)):
                    rows.append({"id": row.id, "label": int(row.label), "length": L,
                                 "strategy": strat, "idx": j, "center": c, "passage": ps})
        if (k + 1) % 100 == 0: print(f"  {k + 1}/{len(df)} books", flush=True)
    pd.DataFrame(rows).to_parquet(os.path.join(DROPBOX, "passages", out_name))
    print(f"{manifest}: wrote {len(rows)} passages -> {out_name}", flush=True)

if __name__ == "__main__":
    os.makedirs(os.path.join(DROPBOX, "passages"), exist_ok=True)
    cert = pd.read_csv(CERT, dtype={"id": str})
    certified = set(cert[cert.verdict == "certified_clean"].id)
    print(f"certification: {len(certified)} certified clean of {len(cert)}", flush=True)
    run("manifest_negatives.csv", "passages_neg.parquet", id_filter=certified)
    run("manifest_positives.csv", "passages_pos.parquet")
