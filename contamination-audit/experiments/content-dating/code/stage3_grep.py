"""Stage 3, part 1: exhaustive grep (markers + year-in-context sieve) with hit-context extraction.

v1 (2026-07-24), PI-approved third pipeline leg. Panels:
  miss_C / miss_B:   in-sample positives missed by frozen stage 2 (OR-3/3)
  miss_oos:          OOS moderns + plants missed by stage 2 (numeric-safe filter — fixes the
                     dtype artifact in the chat validation)
  clean_unscreened:  300 corpus books passing year/stage2 screens but NOT the marker screen
                     (the certified negatives were marker-screened at selection, so they cannot
                     measure stage-3 false-removal load; this sample can)
  neg_certified:     the 918 certified negatives (comparison row)
Sieve: full markers_1945 list (homograph-prone terms INCLUDED — adjudication handles them) +
  post-1929 year within a dating context (in/year/copyright/published/printed/during/since/of/(c)).
Output: intermediates\\stage3_hits.parquet — one row per (panel, book) with up to 6 hit contexts
  (+-800 chars each, marker hits first, then year hits), n_marker_hits, n_year_hits.
"""
import os, re, json
import pandas as pd
import ahocorasick
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

REPO = os.path.dirname(os.path.abspath(__file__))
DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
OUT = os.path.join(DROPBOX, "outputs")
I = os.path.join(DROPBOX, "intermediates")
CORPUS = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\temp-guohui\decontamination\datasets\history-llms\sample-10gb\data"
CTX = 800
MAX_CTX = 6

markers = [m["term"].lower() for m in json.load(open(os.path.join(REPO, "markers_1945.json")))["markers"]]
p1929 = os.path.join(REPO, "markers_1929.json")
if os.path.exists(p1929):  # v3: + 447 null-corpus-validated 1929-cutoff terms (56 curated, 391 mined)
    markers += [m["term"].lower() for m in json.load(open(p1929))["markers"]]
markers = sorted(set(markers))
A = ahocorasick.Automaton()
for m in markers: A.add_word(m, m)
A.make_automaton()
YEAR_CTX = re.compile(r"(copyright|published|printed|in the year|edition of|©)\s{0,3}(19[3-9]\d|20[0-2]\d)(?!\d)", re.I)
WORD = re.compile(r"[a-z0-9]")

def s2_flags(parq_35, parq_go):
    def votes(f):
        d = pd.read_parquet(f)
        s = d[(d.length == 2000) & (d.strategy == "random") & (d.idx < 8)]
        return s.groupby("id").agg(v=("pred", lambda x: int((x == 1).sum()))).v
    v35, vgo = votes(parq_35), votes(parq_go)
    idx = v35.index.union(vgo.index)
    return ((v35.reindex(idx).fillna(0) >= 3) | (vgo.reindex(idx).fillna(0) >= 3)).astype(int)

def corpus_texts(df):
    txt = {}
    for sh, g in df.groupby("shard"):
        t = pq.read_table(os.path.join(CORPUS, sh), columns=["id", "text"])
        t = t.filter(pc.is_in(t.column("id"), value_set=pa.array(list(g.id))))
        for i, tx in zip(t.column("id").to_pylist(), t.column("text").to_pylist()):
            txt[i] = tx or ""
    return txt

def file_text(bid):
    for d_ in ("positives_text", "plants_text"):
        p = os.path.join(DROPBOX, d_, bid + ".txt")
        if os.path.exists(p): return open(p, encoding="utf-8").read()
    return ""

def scan_book(tx):
    """Return (n_marker, n_year, contexts). Contexts on the normalized string."""
    norm = " ".join(tx.split())
    low = norm.lower()
    spans = []
    for end, term in A.iter(low):
        s = end - len(term) + 1
        # WORD BOUNDARIES: v1 matched substrings ("nato" inside "impersonato") — v2 requires
        # non-word characters on both sides
        if s > 0 and WORD.match(low[s - 1]): continue
        if end + 1 < len(low) and WORD.match(low[end + 1]): continue
        spans.append(("marker", term, s, end + 1))
    n_marker = len(spans)
    year_spans = [("year", m.group(2), m.start(), m.end()) for m in YEAR_CTX.finditer(norm)]
    n_year = len(year_spans)
    ordered = spans[:MAX_CTX] + year_spans[: max(0, MAX_CTX - min(len(spans), MAX_CTX))]
    ctxs = []
    for kind, term, s, e in ordered[:MAX_CTX]:
        a = max(0, s - CTX); b = min(len(norm), e + CTX)
        # RAW excerpt, no tag: v1's [marker:X] labels induced the adjudicator to "see" the term
        # even when the match was spurious — it must find the evidence in the text itself
        ctxs.append(f"...{norm[a:b]}...")
    return n_marker, n_year, ctxs

def main():
    s2 = s2_flags(os.path.join(OUT, "judg_Qwen36-35B-A3B-BF16-A100.parquet"),
                  os.path.join(OUT, "judg_openai__gpt-oss-20b.parquet"))
    s2o = s2_flags(os.path.join(OUT, "judg_Qwen36-35B-A3B-BF16-A100.oos.parquet"),
                   os.path.join(OUT, "judg_openai__gpt-oss-20b.oos.parquet"))
    mc = pd.read_csv(os.path.join(REPO, "manifest_contamC.csv"), dtype={"id": str})
    cmod = mc[mc.cls == "contam_modern"]
    b = pd.read_csv(os.path.join(REPO, "manifest_pos3046.csv"), dtype={"id": str})
    omod = pd.read_csv(os.path.join(REPO, "manifest_oos_moderns.csv"), dtype={"id": str})
    opl = pd.read_csv(os.path.join(REPO, "manifest_oos_plants.csv"), dtype=str)
    neg = pd.read_csv(os.path.join(REPO, "manifest_negatives.csv"), dtype={"id": str})

    panels = []
    panels += [("miss_C", i, None) for i in cmod.id if s2.get(i, 0) == 0]
    panels += [("miss_B", i, None) for i in b.id if s2.get(i, 0) == 0]
    panels += [("miss_oos_mod", i, None) for i in omod.id if s2o.get(i, 0) == 0]
    panels += [("miss_oos_plant", i, None) for i in opl.plant_id if s2o.get(i, 0) == 0]
    print(f"missed positives: {sum(1 for p in panels)} across panels", flush=True)

    # unscreened clean sample: year/stage2 screens only, NO marker screen, NO imprint screen
    meta = pd.read_parquet(os.path.join(I, "corpus_meta.parquet"))
    used = set(neg.id) | set(mc.id)
    for f in ("manifest_oos_negatives.csv",):
        used |= set(pd.read_csv(os.path.join(REPO, f), dtype={"id": str}).id)
    for n in ("gold_300_verdicts", "gold_1000_verdicts", "gold_1000b_verdicts"):
        used |= set(pd.read_csv(os.path.join(I, n + ".csv"), dtype={"id": str}).id)
    pool = meta[(meta.source == "institutional_books") & (meta.year <= 1929)
                & (meta.stage2_docyear <= 1929) & (~meta.id.isin(used))]
    uns = pool.sample(300, random_state=777)
    panels += [("clean_unscreened", r.id, r.shard) for r in uns.itertuples()]
    negs = neg.sample(300, random_state=777)
    panels += [("neg_certified", r.id, r.shard) for r in negs.itertuples()]

    need_corpus = pd.DataFrame([(i, sh) for _, i, sh in panels if sh is not None], columns=["id", "shard"])
    cshards = mc.set_index("id").shard.to_dict()
    for p_, i, sh in panels:
        if sh is None and i in cshards:
            need_corpus = pd.concat([need_corpus, pd.DataFrame([(i, cshards[i])], columns=["id", "shard"])])
    ctxt = corpus_texts(need_corpus.drop_duplicates("id"))

    rows = []
    for panel, bid, sh in panels:
        tx = ctxt.get(bid) or file_text(bid)
        if not tx: continue
        nm, ny, ctxs = scan_book(tx)
        rows.append(dict(panel=panel, id=bid, n_marker=nm, n_year=ny,
                         any_hit=int(nm + ny > 0), contexts="\n===\n".join(ctxs)))
    d = pd.DataFrame(rows)
    d.to_parquet(os.path.join(I, "stage3_hits.parquet"))
    print(d.groupby("panel").agg(n=("id", "count"), hit_rate=("any_hit", "mean"),
                                 med_markers=("n_marker", "median")).round(2).to_string(), flush=True)

if __name__ == "__main__":
    main()
