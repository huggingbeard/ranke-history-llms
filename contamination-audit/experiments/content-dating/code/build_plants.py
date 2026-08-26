"""Build metadata-corrupted plants for the composite pipeline evaluation.

v1 (2026-07-23). PI condition: plants must RESEMBLE genuine corpus books — "if that is assured
- test it". Corruption patterns are copied from the 3 real got-throughs (all had authentic but
UNINFORMATIVE front matter, none was surgically stripped):
  P1 truncate:   drop everything before a paragraph boundary at 2-5% depth (title/copyright pages
                 gone — the "scan starts mid-content" pattern, extremely common in real scans).
  P2 provenance: P1 + prepend a GENUINE provenance/bookplate block harvested from a certified-clean
                 negative's front matter (authentic period text, the 970f287d pattern).
Metadata corruption: meta_year drawn from the negatives' year distribution (seeded per plant).
stage2_docyear is NOT corrupted here — the composite is scored under two bounds (fooled = meta_year,
correct = true year) until the real 14B doc-dater can run (GPU busy with the eugenics track).

Realism gate (all must pass before plants are used):
  G1: imprint_year(plant) is None for truncate-pattern plants (programmatic).
  G2: the no-imprint front is a COMMON genuine class — report base rate among certified negatives.
  G3: blind discrimination — deepseek judges shuffled plant vs genuine no-imprint fronts as
      "artificially assembled?"; accuracy ~50% = indistinguishable (run: certify script below).
Outputs: Dropbox plants_text\\<plantid>.txt, repo manifest_plants.csv
         (plant_id, parent_id, parent_cls, pattern, fake_meta_year, true_year).
Stage-2 verdicts for plants = parent's existing per-passage verdicts (body is unchanged;
passages sample the 5-95% window, corruption touches only the front).
"""
import os, sys, random
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, r"C:\Users\hvoth\Documents\GitHub\ranke-edition-dating")
from freeze_pipeline import imprint_year

REPO = os.path.dirname(os.path.abspath(__file__))
DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
CORPUS = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\temp-guohui\decontamination\datasets\history-llms\sample-10gb\data"
PLANTS = os.path.join(DROPBOX, "plants_text")
SEED = 1929
N_FROM_B, N_FROM_C = 120, 80

def load_corpus_texts(ids_shards):
    txt = {}
    for sh, g in ids_shards.groupby("shard"):
        t = pq.read_table(os.path.join(CORPUS, sh), columns=["id", "text"])
        d = dict(zip(t.column("id").to_pylist(), t.column("text").to_pylist()))
        for i in g.id: txt[i] = d.get(i, "")
    return txt

def main():
    os.makedirs(PLANTS, exist_ok=True)
    rng = random.Random(SEED)
    b = pd.read_csv(os.path.join(REPO, "manifest_pos3046.csv"), dtype={"id": str})
    mc = pd.read_csv(os.path.join(REPO, "manifest_contamC.csv"), dtype={"id": str})
    cmod = mc[mc.cls == "contam_modern"]
    neg = pd.read_csv(os.path.join(REPO, "manifest_negatives.csv"), dtype={"id": str})

    # G2 baseline + provenance harvest from genuine negatives
    neg_s = neg.sample(220, random_state=SEED)
    neg_txt = load_corpus_texts(neg_s)
    no_imp = [i for i, tx in neg_txt.items() if len(tx) > 5000 and imprint_year(tx) is None]
    print(f"G2 base rate: {len(no_imp)}/{len(neg_txt)} genuine certified-clean books have NO imprint year "
          f"({100*len(no_imp)/max(1,len(neg_txt)):.0f}%)", flush=True)
    prov_blocks = []
    PROV_WORDS = ("library", "legacy", "bequest", "gift", "estate", "fund", "college", "received")
    for i in no_imp:
        head = neg_txt[i][:1200]
        first_para = head.split("\n\n")[0]
        if 120 < len(first_para) < 900 and any(w in first_para.lower() for w in PROV_WORDS):
            prov_blocks.append(first_para.strip())
    print(f"harvested {len(prov_blocks)} genuine provenance blocks", flush=True)

    neg_years = pd.to_numeric(neg.meta_year).tolist()
    parents = pd.concat([
        b.sample(N_FROM_B, random_state=SEED).assign(parent_cls="B_1930_46_ordinary", true_year=lambda d: d.meta_year),
        cmod.sample(N_FROM_C, random_state=SEED).assign(parent_cls="contam_modern", true_year=lambda d: d.meta_year),
    ], ignore_index=True)

    c_shards = mc.set_index("id").shard.to_dict()
    c_txt = load_corpus_texts(pd.DataFrame({"id": [r.id for r in parents.itertuples() if r.parent_cls == "contam_modern"],
                                            "shard": [c_shards[r.id] for r in parents.itertuples() if r.parent_cls == "contam_modern"]}))
    rows, g1_fail = [], 0
    for r in parents.itertuples():
        if r.parent_cls == "contam_modern":
            tx = c_txt.get(r.id, "")
        else:
            p = os.path.join(DROPBOX, "positives_text", r.id + ".txt")
            tx = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
        if len(tx) < 20000: continue
        # P1: cut at a paragraph boundary between 2% and 5% depth
        lo, hi = int(len(tx) * 0.02), int(len(tx) * 0.05)
        cut = tx.find("\n\n", rng.randint(lo, hi))
        if cut < 0: cut = hi
        body = tx[cut:].lstrip()
        pattern = "P1_truncate"
        if prov_blocks and rng.random() < 0.4:
            body = rng.choice(prov_blocks) + "\n\n" + body
            pattern = "P2_provenance"
        iy = imprint_year(body)
        if iy is not None and iy > 1929:
            # still carries a modern imprint deeper in front/back matter — cut again past it or skip
            g1_fail += 1
            continue
        fake_year = int(rng.choice(neg_years))
        pid = r.id + "_P"
        with open(os.path.join(PLANTS, pid + ".txt"), "w", encoding="utf-8") as f:
            f.write(body)
        rows.append({"plant_id": pid, "parent_id": r.id, "parent_cls": r.parent_cls,
                     "pattern": pattern, "fake_meta_year": fake_year, "true_year": r.true_year,
                     "imprint_after": iy if iy is not None else ""})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(REPO, "manifest_plants.csv"), index=False)
    print(f"plants built: {len(out)} ({dict(out.pattern.value_counts())}); "
          f"G1: {g1_fail} parents skipped (residual modern imprint survived the cut)", flush=True)
    print(f"imprint_year None after corruption: {(out.imprint_after == '').mean():.0%}", flush=True)

if __name__ == "__main__":
    main()
