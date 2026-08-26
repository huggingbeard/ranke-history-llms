"""Build a 1929-cutoff marker list (option 2 of the PI's upgrade round) and validate it in-corpus.

v1 (2026-07-24). markers_1945 was built for a 1945 cutoff and deliberately excludes 1930-45
coinages — exactly the vocabulary of the binding 1930-46 stratum. Two candidate sources:
  A. curated 1930-1945 coinages / post-1929-prominent named entities (list below, each with a
     rough coinage rationale; multiword forms preferred where single words are homograph-risky);
  B. mined: word-boundary unigrams/bigrams with document frequency >= 5 in MINING moderns and
     0 in certified pre-1929 negatives. LEAKAGE GUARD: mining uses ONLY books the pipeline
     already catches (stage-2-flagged positives); the stage-3 miss panels stay disjoint for
     evaluation.
Validation (both sources): ZERO word-boundary hits across all 1,283 certified pre-1929 books
(918 in-sample + 365 OOS) — an in-corpus null test that catches OCR homographs directly, which
is how the 1945 list's "nato"/"talk show" failures arose.
Output: markers_1929.json (validated terms + provenance), stats printed.
"""
import os, re, json
from collections import Counter
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

REPO = os.path.dirname(os.path.abspath(__file__))
DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
I = os.path.join(DROPBOX, "intermediates")
CORPUS = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\temp-guohui\decontamination\datasets\history-llms\sample-10gb\data"

CURATED = [
    # Germany / WWII (coined or English-current only after 1929)
    "new deal", "third reich", "luftwaffe", "wehrmacht", "gestapo", "blitzkrieg", "panzer division",
    "nuremberg laws", "nuremberg rally", "anschluss", "sudeten crisis", "munich agreement",
    "kristallnacht", "maginot line", "vichy government", "vichy france", "quisling",
    "lend-lease", "pearl harbor attack", "d-day", "atomic bomb", "a-bomb", "manhattan project",
    "battle of britain", "dunkirk evacuation", "kamikaze",
    # science/technology coinages 1930-45
    "penicillin", "nylon", "radar", "sonar", "positron", "cyclotron", "plutonium", "neptunium",
    "sulfanilamide", "antibiotics", "streptomycin", "microfilm", "ballpoint pen", "jet propulsion",
    "jeep", "bazooka", "walkie-talkie", "television broadcast", "television set",
    # society/economy 1930s
    "social security act", "works progress administration", "civilian conservation corps",
    "keynesian", "gross national product", "supermarket", "drive-in", "jitterbug",
    "five-year plan", "collectivization", "gulag", "nkvd", "politburo member",
    "united nations organization", "flying fortress", "aircraft carrier task force",
]

WORD = re.compile(r"[a-z0-9]")
TOKEN = re.compile(r"[a-z][a-z\-']{3,}")

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

def norm(tx): return " ".join(tx.lower().split())

def wb_contains(low, term):
    i = low.find(term)
    while i >= 0:
        ok_l = i == 0 or not WORD.match(low[i - 1])
        j = i + len(term)
        ok_r = j >= len(low) or not WORD.match(low[j])
        if ok_l and ok_r: return True
        i = low.find(term, i + 1)
    return False

def main():
    # ---- load null corpus (all certified pre-1929 books) ----
    neg1 = pd.read_csv(os.path.join(REPO, "manifest_negatives.csv"), dtype={"id": str})
    neg2 = pd.read_csv(os.path.join(REPO, "manifest_oos_negatives.csv"), dtype={"id": str})
    negs = pd.concat([neg1[["id", "shard"]], neg2[["id", "shard"]]])
    ntxt = corpus_texts(negs)
    neg_lows = [norm(t) for t in ntxt.values() if len(t) > 2000]
    print(f"null corpus: {len(neg_lows)} certified pre-1929 books", flush=True)

    # ---- mining moderns: stage-2-CAUGHT positives only (leakage guard) ----
    def v(f):
        d = pd.read_parquet(os.path.join(DROPBOX, "outputs", f))
        s = d[(d.length == 2000) & (d.strategy == "random") & (d.idx < 8)]
        return s.groupby("id").agg(v=("pred", lambda x: int((x == 1).sum()))).v
    v35 = v("judg_Qwen36-35B-A3B-BF16-A100.parquet"); vgo = v("judg_openai__gpt-oss-20b.parquet")
    idx = v35.index.union(vgo.index)
    s2 = ((v35.reindex(idx).fillna(0) >= 3) | (vgo.reindex(idx).fillna(0) >= 3))
    caught = set(s2[s2].index)
    mc = pd.read_csv(os.path.join(REPO, "manifest_contamC.csv"), dtype={"id": str})
    b = pd.read_csv(os.path.join(REPO, "manifest_pos3046.csv"), dtype={"id": str})
    mine_ids = [i for i in list(mc[mc.cls == "contam_modern"].id) + list(b.id) if i in caught]
    ctxt = corpus_texts(mc[mc.id.isin(mine_ids)][["id", "shard"]])
    mod_lows = []
    for i in mine_ids:
        t = ctxt.get(i) or file_text(i)
        if len(t) > 2000: mod_lows.append(norm(t))
    print(f"mining moderns (stage-2-caught only): {len(mod_lows)} books", flush=True)

    # ---- mine unigrams + bigrams by document frequency ----
    df_mod, df_neg = Counter(), Counter()
    def doc_terms(low):
        toks = TOKEN.findall(low)
        return set(toks) | {f"{a} {b}" for a, b in zip(toks, toks[1:])}
    for low in mod_lows:
        for t in doc_terms(low): df_mod[t] += 1
    neg_term_docs = [doc_terms(low) for low in neg_lows]
    for s in neg_term_docs:
        for t in s: df_neg[t] += 1
    mined = [t for t, c in df_mod.items() if c >= 5 and df_neg.get(t, 0) == 0]
    mined = sorted(mined, key=lambda t: -df_mod[t])[:400]
    print(f"mined candidates (df_mod>=5, df_neg=0): {len(mined)}", flush=True)

    # ---- validate BOTH sources with word-boundary scan over the full null corpus ----
    old = {m["term"].lower() for m in json.load(open(os.path.join(REPO, "markers_1945.json")))["markers"]}
    cands = [(t, "curated") for t in CURATED] + [(t, "mined") for t in mined if t not in old]
    validated, rejected = [], []
    for term, src in cands:
        bad = sum(1 for low in neg_lows if wb_contains(low, term))
        if bad == 0:
            validated.append({"term": term, "source": src, "df_modern": df_mod.get(term, 0)})
        else:
            rejected.append((term, src, bad))
    print(f"validated: {len(validated)} ({sum(1 for v_ in validated if v_['source']=='curated')} curated, "
          f"{sum(1 for v_ in validated if v_['source']=='mined')} mined); rejected {len(rejected)}", flush=True)
    print("sample rejected:", [(t, n) for t, s, n in rejected[:8]], flush=True)
    print("top mined:", [v_["term"] for v_ in validated if v_["source"] == "mined"][:25], flush=True)
    json.dump({"cutoff": 1929, "validation": "zero word-boundary hits in 1,283 certified pre-1929 books",
               "markers": validated}, open(os.path.join(REPO, "markers_1929.json"), "w"), indent=1)
    print("-> markers_1929.json", flush=True)

if __name__ == "__main__":
    main()
