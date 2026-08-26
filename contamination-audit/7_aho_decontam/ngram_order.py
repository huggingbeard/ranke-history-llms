"""
ngram_order.py — do longer n-grams give cleaner anachronism signal? (1- vs 2- vs 3-grams)

Honest train/test design (no peeking):
  TRAIN: mine candidate n-grams that appear in MANY modern-train docs and ZERO old-train docs.
  TEST : on held-out docs, measure recall (modern caught) and false-positives (old wrongly hit)
         as a function of how many distinct candidate n-grams a doc contains (count threshold).

Modern corpus = Wikipedia (abundant, post-1945). Old corpus = LoC-PD-Books year<=1929.
Also reports recall on a held-out set of REAL post-1945 LoC books (the actual target genre).

Expectation: higher n -> fewer false positives (a 3-word phrase rarely collides) but lower
coverage (rarer). The table/plot shows the precision/recall trade-off across n.
"""
import json, os, re, sys, time
from collections import Counter
from pathlib import Path
import aho_scan as A
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
N_WIKI, N_OLD, N_REAL, CAP = 2000, 2000, 150, 14000
NS = [1, 2, 3]
DF_MIN = 6          # candidate must appear in >= this many modern-train docs
TOK = re.compile(r"[a-z]+")


def grams(text, n):
    toks = [w for w in TOK.findall(A.normalize(text[:40000])) if len(w) > 1]
    if n == 1: return set(toks)
    return {" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)}


def main():
    tok = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read_text().strip()
    os.environ["HF_HUB_DISABLE_SYMLINKS"]="1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
    os.environ["HF_TOKEN"]=tok
    from datasets import load_dataset

    def load(repo, **kw):
        for _ in range(6):
            try: return load_dataset(repo, split="train", streaming=True, token=tok, **kw)
            except Exception as e:
                print(f"  load {repo} failed; retry", file=sys.stderr, flush=True); time.sleep(8)
        raise SystemExit("load failed")

    # gram-sets per doc, split into train/test by index parity
    print("streaming Wikipedia (modern) ...")
    wiki = [r["text"] for i, r in enumerate(load("wikimedia/wikipedia", name="20231101.en")) if i < N_WIKI and r.get("text")]
    print("streaming LoC (old + real-modern) ...")
    old, real = [], []
    for i, rec in enumerate(load("storytracer/LoC-PD-Books")):
        if i >= CAP or (len(old) >= N_OLD and len(real) >= N_REAL): break
        yr, txt = rec.get("year"), A.doc_text(rec)
        if not isinstance(yr, int) or not txt: continue
        if yr <= 1929 and len(old) < N_OLD: old.append(txt)
        elif yr >= 1946 and len(real) < N_REAL: real.append(txt)
    print(f"  wiki={len(wiki)} old={len(old)} real_modern={len(real)}")

    def split(lst): return lst[::2], lst[1::2]    # train, test
    wiki_tr, wiki_te = split(wiki); old_tr, old_te = split(old)

    rows = []
    for n in NS:
        gw_tr = [grams(t, n) for t in wiki_tr]; go_tr = [grams(t, n) for t in old_tr]
        gw_te = [grams(t, n) for t in wiki_te]; go_te = [grams(t, n) for t in old_te]
        gr_te = [grams(t, n) for t in real]
        dfm = Counter(); dfo = Counter()
        for s in gw_tr: dfm.update(s)
        for s in go_tr: dfo.update(s)
        cand = {g for g, c in dfm.items() if c >= DF_MIN and dfo.get(g, 0) == 0}
        def cnt(gs): return len(gs & cand)
        def frac(sets, k): return 100*sum(1 for s in sets if cnt(s) >= k)/max(len(sets), 1)
        rows.append(dict(n=n, n_candidates=len(cand),
            rec_wiki1=frac(gw_te,1), rec_wiki2=frac(gw_te,2),
            rec_real1=frac(gr_te,1), rec_real2=frac(gr_te,2),
            fp_old1=frac(go_te,1), fp_old2=frac(go_te,2),
            top=[g for g,_ in dfm.most_common(2000) if g in cand][:12]))

    print("\n" + "=" * 92)
    print("N-GRAM ORDER — recall (held-out) vs false-positives, by n and count-threshold")
    print("=" * 92)
    print(f"{'n':>2}{'#cand':>8}{'recall wiki >=1':>16}{'recall real >=1':>16}"
          f"{'FP old >=1':>12}{'recall real >=2':>16}{'FP old >=2':>12}")
    for r in rows:
        print(f"{r['n']:>2}{r['n_candidates']:>8}{r['rec_wiki1']:>15.1f}%{r['rec_real1']:>15.1f}%"
              f"{r['fp_old1']:>11.1f}%{r['rec_real2']:>15.1f}%{r['fp_old2']:>11.1f}%")
    for r in rows:
        print(f"\n top {r['n']}-gram markers: {', '.join(r['top'])}")

    # plot: recall(real, >=1) vs FP(old, >=1) per n
    plt.figure(figsize=(7,5.5))
    for r in rows:
        plt.scatter(r['fp_old1'], r['rec_real1'], s=80)
        plt.annotate(f"{r['n']}-gram (>=1)", (r['fp_old1'], r['rec_real1']),
                     textcoords="offset points", xytext=(7,0), fontsize=9)
        plt.scatter(r['fp_old2'], r['rec_real2'], s=50, marker='^')
        plt.annotate(f"{r['n']}-gram (>=2)", (r['fp_old2'], r['rec_real2']),
                     textcoords="offset points", xytext=(7,-10), fontsize=8, color="#777")
    plt.xlabel("false positives on old books (%)"); plt.ylabel("recall on real modern books (%)")
    plt.title("Cleaner vs broader: n-gram order trade-off\n(mined on train, measured on held-out)")
    plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(HERE/"ngram_order.png", dpi=130); plt.close()
    (HERE/"ngram_order.json").write_text(json.dumps(rows, indent=2))
    print("\nsaved -> ngram_order.json, ngram_order.png")


if __name__ == "__main__":
    main()
