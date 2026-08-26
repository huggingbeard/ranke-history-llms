"""
vocab_ablation.py — does a bigger giveaway-word list catch more impostors? (full-book dose)

Question: as we grow the lexicon from 1 word to all 128, how does the CATCH RATE (recall) on
whole post-1945 books change, and does PRECISION suffer? (Precision is already ~100%, so the
point is: extra vocab buys recall at no precision cost.)

Setup (whole-book contamination only):
  POSITIVES = whole modern documents -> real post-1945 LoC books (the actual threat) and whole
              Wikipedia articles (abundant cross-check).
  NEGATIVES = whole genuine pre-1929 LoC books.
  Each doc is scanned ONCE with the full 128-marker scanner -> the set of markers it contains.
  Ablation is then pure set logic: with the top-k words, a doc is "caught" iff its marker-set
  intersects the top-k set. Two orderings:
     frequency-ranked -> markers sorted by their post-1945 Ngram frequency (independent, the
                         order you'd actually deploy)
     greedy-optimal   -> repeatedly add the marker that catches the most still-missed books
                         (an oracle upper bound: "how few words could suffice")

Outputs: vocab_ablation.json + vocab_ablation.png. Caches scanned marker-sets so reruns are instant.
"""
import json, os, random, sys
from pathlib import Path
import numpy as np
import aho_scan as A
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
SEED = 20260629; random.seed(SEED)
N_NEG = 600; N_REAL = 150; N_WIKI = 500; SCAN_CAP = 12000
KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
CACHE = HERE / "vocab_ablation_cache.json"


def hf_token():
    p = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt")
    return p.read_text().strip() if p.exists() else os.getenv("HF_TOKEN")


def build_cache():
    tok = hf_token()
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    if tok: os.environ["HF_TOKEN"] = tok
    from datasets import load_dataset
    markers, contexts = A.load_markers(); auto = A.build_automaton(markers)
    def found(text):
        return sorted({m for m, _ in A.scan_text(A.normalize(text[:200000]), auto, contexts)})

    print("streaming LoC-PD-Books (negatives + real modern positives) ...")
    neg, real = [], []
    ds = load_dataset("storytracer/LoC-PD-Books", split="train", streaming=True, token=tok)
    n = 0
    for rec in ds:
        if n >= SCAN_CAP or (len(neg) >= N_NEG and len(real) >= N_REAL): break
        n += 1
        yr, txt = rec.get("year"), A.doc_text(rec)
        if not isinstance(yr, int) or not txt: continue
        if yr <= 1929 and len(neg) < N_NEG: neg.append(found(txt))
        elif yr >= 1946 and len(real) < N_REAL: real.append(found(txt))
        if n % 1000 == 0: print(f"  {n}: neg={len(neg)} real_modern={len(real)}")
    print(f"  negatives={len(neg)}, real-modern positives={len(real)}")

    print("streaming Wikipedia (modern positives) ...")
    wiki = []
    dw = load_dataset("wikimedia/wikipedia", name="20231101.en", split="train",
                      streaming=True, token=tok)
    for i, r in enumerate(dw):
        if len(wiki) >= N_WIKI: break
        if r.get("text"): wiki.append(found(r["text"]))
    print(f"  wikipedia positives={len(wiki)}")

    cache = {"neg": neg, "real": real, "wiki": wiki, "markers": markers}
    CACHE.write_text(json.dumps(cache))
    return cache


def main():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else build_cache()
    neg = [set(f) for f in cache["neg"]]
    real = [set(f) for f in cache["real"]]
    wiki = [set(f) for f in cache["wiki"]]
    deployed = set(cache["markers"])
    print(f"\nloaded: neg={len(neg)} real={len(real)} wiki={len(wiki)} markers={len(deployed)}")

    # --- orderings ---
    meta = {m["term"].lower(): m for m in json.loads((HERE / "markers_1945.json").read_text())["markers"]}
    ranked_freq = sorted(deployed, key=lambda t: -meta.get(t, {}).get("post_mean", 0.0))
    # greedy-optimal on the real-modern positives
    order_g, covered = [], [False] * len(real)
    pool = set(deployed)
    while pool and not all(covered):
        best, gain = None, -1
        for m in pool:
            g = sum(1 for i, f in enumerate(real) if not covered[i] and m in f)
            if g > gain: best, gain = m, g
        if gain <= 0: break
        order_g.append(best); pool.discard(best)
        covered = [c or (best in real[i]) for i, c in enumerate(covered)]
    order_g += [m for m in ranked_freq if m not in order_g]   # leftovers in freq order

    def recall(found_sets, topset):
        return np.mean([bool(f & topset) for f in found_sets]) if found_sets else 0.0

    rows = []
    for k in KS:
        tf, tg = set(ranked_freq[:k]), set(order_g[:k])
        rr_f, rr_g = recall(real, tf), recall(real, tg)
        rw_f = recall(wiki, tf)
        fp = sum(bool(f & tf) for f in neg)
        tp = sum(bool(f & tf) for f in real)
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rows.append(dict(k=k, recall_real_freq=rr_f, recall_real_greedy=rr_g,
                         recall_wiki_freq=rw_f, false_pos=fp,
                         false_accusation_rate=fp / max(len(neg), 1), precision_real=prec))

    # --- report ---
    print("\n" + "=" * 86)
    print("VOCABULARY ABLATION — full-book contamination: catch rate vs #giveaway words")
    print("=" * 86)
    print(f"{'#words':>7}{'recall real (freq)':>20}{'recall real (greedy)':>22}"
          f"{'recall wiki':>13}{'false-pos':>11}{'precision':>11}")
    for r in rows:
        print(f"{r['k']:>7}{r['recall_real_freq']*100:>19.1f}%{r['recall_real_greedy']*100:>21.1f}%"
              f"{r['recall_wiki_freq']*100:>12.1f}%{r['false_pos']:>11}{r['precision_real']*100:>10.1f}%")
    print(f"\ngreedy: {sum(1 for r in rows)} ; first 12 greedy-optimal words: "
          f"{', '.join(order_g[:12])}")
    print(f"top-12 by frequency: {', '.join(ranked_freq[:12])}")

    # --- plot ---
    fig, ax1 = plt.subplots(figsize=(8, 5.2))
    ks = [r["k"] for r in rows]
    ax1.plot(ks, [r["recall_real_greedy"]*100 for r in rows], "--o", color="#e67e22",
             label="catch rate, real books (greedy-optimal order)")
    ax1.plot(ks, [r["recall_real_freq"]*100 for r in rows], "-o", color="#c0392b",
             label="catch rate, real post-1945 books (frequency order)")
    ax1.plot(ks, [r["recall_wiki_freq"]*100 for r in rows], "-o", color="#2c3e50",
             label="catch rate, Wikipedia articles (frequency order)")
    ax1.plot(ks, [r["precision_real"]*100 for r in rows], "-s", color="#27ae60",
             label="precision (true catches / all flags)")
    ax1.set_xscale("log"); ax1.set_xlabel("number of giveaway words in the lexicon (log scale)")
    ax1.set_ylabel("percent"); ax1.set_ylim(-3, 103)
    ax1.set_title("More vocabulary → higher catch rate, with precision staying ~100%\n"
                  "(whole-book contamination)")
    ax1.grid(alpha=.3); ax1.legend(loc="center right", fontsize=9); fig.tight_layout()
    fig.savefig(HERE / "vocab_ablation.png", dpi=130); plt.close()

    (HERE / "vocab_ablation.json").write_text(json.dumps(
        {"seed": SEED, "n_neg": len(neg), "n_real": len(real), "n_wiki": len(wiki),
         "ks": KS, "rows": rows,
         "greedy_first12": order_g[:12], "freq_first12": ranked_freq[:12]}, indent=2))
    print("\nsaved -> vocab_ablation.json, vocab_ablation.png")


if __name__ == "__main__":
    main()
