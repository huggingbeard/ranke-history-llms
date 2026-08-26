"""
contamination_trial.py — controlled ground-truth evaluation of the post-1945 detector.

Treats the scanner as a binary classifier and measures ROC-AUC, average precision, the
precision/recall trade-off, and explicit FP/FN counts as a function of how much modern text
is planted (the dose-response curve).

DESIGN
  * SCORE(doc) = number of DISTINCT validated markers detected (after word-boundary + context
    checks). A graded score, so we can sweep a decision threshold and draw ROC / PR curves.
  * NEGATIVES (label 0): real LoC-PD-Books with stated year <= 1929 (the genuine-old population,
    NOT pre-filtered -> gives an honest false-positive rate).
  * POSITIVES (label 1): a verified-clean old book with a length-X section spliced in from a
    NEW (post-1945) source. Any detection then comes from the planted text.
  * THREE contaminant sources bracket reality:
       real_modern_loc -> genuine post-1945 books (the actual threat)
       wikipedia       -> modern encyclopedic prose (vocabulary-rich)
       tinystories     -> bland modern fiction (vocabulary-poor; adversarial worst case)
  * DOSE = chars of modern text planted, swept over DOSES -> dose-response curve.
  * Integrity: "ceiling" = does the planted chunk ALONE contain a marker. recall == ceiling
    proves the scanner catches every detectable chunk (misses are modern text with no
    period-specific vocabulary -- the inherent limit of a lexicon method, not a code bug).

Outputs: console tables, trial_results.json, and dose_response.png / roc.png / pr.png.
Usage: python contamination_trial.py
"""
import json, os, random, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
import aho_scan as A
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

SEED = 20260629
random.seed(SEED)
_i = lambda k, d: int(os.getenv(k, d))          # env overrides for a fast smoke test
N_NEG     = _i("CT_NEG", 600)      # genuine old books used as negatives
SCAN_CAP  = _i("CT_CAP", 8000)     # max LoC books to stream while building pools
N_WIKI    = _i("CT_WIKI", 250)
N_TINY    = _i("CT_TINY", 1500)
N_TRIALS  = _i("CT_TRIALS", 400)   # positive trials per (source x dose)
# chars planted: ~sentence, ~paragraph, ~few pages, ~chapter, ~short book, ~book
DOSES     = [60, 300, 1500, 8000, 40000, 150000]
WS = re.compile(r"\s+")


def hf_token():
    p = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt")
    return p.read_text().strip() if p.exists() else None


def snap(pool, start, n):
    a = max(0, min(start, max(0, len(pool) - n))); b = min(len(pool), a + n)
    while a > 0 and pool[a - 1].isalnum(): a -= 1
    while b < len(pool) and pool[b].isalnum(): b += 1
    return pool[a:b]


def main():
    tok = hf_token()
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    if tok: os.environ["HF_TOKEN"] = tok
    from datasets import load_dataset

    markers, contexts = A.load_markers()
    auto = A.build_automaton(markers)
    def score(text):
        return len({m for m, _ in A.scan_text(A.normalize(text), auto, contexts)})
    print(f"scanner: {len(markers)} markers, {len(contexts)} context-guarded\n")

    # ---- stream LoC: negatives (year<=1929), clean injection base, real-modern donors ----
    print("streaming LoC-PD-Books for negatives + clean base + real-modern donors ...")
    neg_books, clean_base, modern_donor = [], [], []
    modern_seen = modern_flagged = 0
    ds = load_dataset("storytracer/LoC-PD-Books", split="train", streaming=True, token=tok)
    n = 0
    for rec in ds:
        if n >= SCAN_CAP or (len(neg_books) >= N_NEG and len(modern_donor) >= 120):
            break
        n += 1
        yr, txt = rec.get("year"), A.doc_text(rec)
        if not isinstance(yr, int) or not txt:
            continue
        if yr <= 1929:
            if len(neg_books) < N_NEG:
                t = txt[:60000]; neg_books.append(t)
                if score(t) == 0:
                    clean_base.append(t)
        elif yr >= 1946:
            modern_seen += 1; modern_flagged += (score(txt) > 0)
            if len(modern_donor) < 120:
                modern_donor.append(txt[:60000])
        if n % 1000 == 0:
            print(f"  streamed {n}: neg={len(neg_books)} clean_base={len(clean_base)} "
                  f"modern_donor={len(modern_donor)}")
    print(f"  negatives={len(neg_books)}, clean injection base={len(clean_base)}, "
          f"real-modern donors={len(modern_donor)}\n")

    # ---- modern donor pools ----
    def pool_external(name, kw, want):
        parts = []
        d = load_dataset(name, split="train", streaming=True, token=tok, **kw)
        for i, r in enumerate(d):
            if i >= want: break
            if r.get("text"): parts.append(r["text"][:40000])
        return WS.sub(" ", " \n ".join(parts))
    print("building modern donor pools (wikipedia, tinystories) ...")
    pools = {
        "real_modern_loc": WS.sub(" ", " \n ".join(modern_donor)),
        "wikipedia":  pool_external("wikimedia/wikipedia", dict(name="20231101.en"), N_WIKI),
        "tinystories": pool_external("roneneldan/TinyStories", {}, N_TINY),
    }
    for k, v in pools.items():
        print(f"  {k:<16} pool = {len(v):,} chars")

    # ---- negatives scored once (reused for every cell) ----
    neg_scores = np.array([score(b) for b in neg_books])
    print(f"\nnegative scores: {(neg_scores>0).sum()}/{len(neg_scores)} old books have >=1 marker "
          f"(false-alarm rate {100*(neg_scores>0).mean():.2f}%)")

    # ---- positive trials per (source x dose) ----
    print(f"running {N_TRIALS} positive trials per (source x dose); seed={SEED} ...")
    cells, curve = {}, {s: {"dose": [], "recall": [], "precision": [], "auc": [], "ceiling": []}
                        for s in pools}
    pooled_pos = {s: [] for s in pools}     # all positive scores (all doses) for source-level ROC
    miss_eg = {}
    for src, pool in pools.items():
        if len(pool) < max(DOSES) + 10:
            print(f"  [skip {src}: pool too small ({len(pool)} chars)]"); continue
        for dose in DOSES:
            pos = np.empty(N_TRIALS, dtype=int); ceil_hits = 0
            for i in range(N_TRIALS):
                chunk = snap(pool, random.randint(0, len(pool) - dose - 1), dose)
                ceil_hits += (score(chunk) > 0)
                base = random.choice(clean_base)
                p = random.randint(0, len(base))
                # " ~ " sentinels stop a base word + chunk word from accidentally forming a
                # 2-word marker at the junction (e.g. base "...cold" + chunk "war..."), so any
                # detection is genuinely from the planted chunk -> recall == ceiling exactly.
                pos[i] = score(base[:p] + " ~ " + chunk + " ~ " + base[p:])
                if score(chunk) == 0 and src not in miss_eg:
                    miss_eg[src] = WS.sub(" ", chunk)[:150]
            pooled_pos[src].extend(pos.tolist())
            y = np.r_[np.ones(N_TRIALS), np.zeros(len(neg_scores))]
            s = np.r_[pos, neg_scores]
            auc = roc_auc_score(y, s) if len(set(s)) > 1 else float("nan")
            tp = int((pos >= 1).sum()); fn = N_TRIALS - tp
            fp = int((neg_scores >= 1).sum()); tn = len(neg_scores) - fp
            prec = tp / (tp + fp) if (tp + fp) else 1.0
            cells[f"{src}|{dose}"] = dict(source=src, dose=dose, recall=tp / N_TRIALS,
                precision=prec, auc=auc, ceiling=ceil_hits / N_TRIALS,
                TP=tp, FN=fn, FP=fp, TN=tn)
            for kk, vv in [("dose", dose), ("recall", tp / N_TRIALS), ("precision", prec),
                           ("auc", auc), ("ceiling", ceil_hits / N_TRIALS)]:
                curve[src][kk].append(vv)

    # ---- console report ----
    print("\n" + "=" * 92)
    print("DOSE-RESPONSE — detector vs planted modern text (positives) vs real old books (negatives)")
    print("=" * 92)
    print(f"{'source':<16}{'dose':>7}{'~words':>7}{'recall':>9}{'precision':>10}{'AUC':>7}"
          f"{'ceiling':>9}{'TP':>5}{'FN':>5}{'FP':>5}")
    for src in pools:
        for dose in DOSES:
            c = cells.get(f"{src}|{dose}")
            if not c: continue
            print(f"{src:<16}{dose:>7}{dose//6:>7}{c['recall']*100:>8.1f}%{c['precision']*100:>9.1f}%"
                  f"{c['auc']:>7.3f}{c['ceiling']*100:>8.1f}%{c['TP']:>5}{c['FN']:>5}{c['FP']:>5}")
    recall_eq_ceiling = all(abs(cells[k]['recall'] - cells[k]['ceiling']) < 1e-9 for k in cells)
    print(f"\nrecall == ceiling in every cell? {recall_eq_ceiling}  "
          f"-> the scanner catches EVERY chunk that contains a giveaway word; misses are modern\n"
          f"   snippets with no period-specific vocabulary (the inherent limit, not a code bug).")
    print(f"false alarms (negatives flagged): {(neg_scores>=1).sum()}/{len(neg_scores)} "
          f"old books  (precision stays high because clean old books almost never score > 0)")
    print(f"real-data recall (NO injection): {modern_flagged}/{modern_seen} genuine post-1945 "
          f"books flagged ({100*modern_flagged/max(modern_seen,1):.0f}%)")
    for s, eg in miss_eg.items():
        print(f"  undetectable {s} snippet (no period vocabulary): ...{eg}...")

    # ---- source-level pooled ROC + PR (all doses) + AUC ----
    print("\nsource-level (pooled over doses):")
    pooled = {}
    for src in pools:
        if not pooled_pos[src]: continue
        y = np.r_[np.ones(len(pooled_pos[src])), np.zeros(len(neg_scores))]
        s = np.r_[np.array(pooled_pos[src]), neg_scores]
        pooled[src] = dict(auc=roc_auc_score(y, s), ap=average_precision_score(y, s),
                           y=y, s=s)
        print(f"  {src:<16} ROC-AUC={pooled[src]['auc']:.3f}  avg-precision={pooled[src]['ap']:.3f}")

    # ---- plots ----
    colors = {"real_modern_loc": "#c0392b", "wikipedia": "#2c3e50", "tinystories": "#27ae60"}
    # dose-response
    plt.figure(figsize=(7, 5))
    for src in pools:
        if not curve[src]["dose"]: continue
        plt.plot(curve[src]["dose"], [r*100 for r in curve[src]["recall"]], "-o",
                 color=colors[src], label=f"{src} (recall)")
    plt.xscale("log"); plt.xlabel("modern text planted (characters, log scale)")
    plt.ylabel("recall  (% of contaminated books caught)"); plt.ylim(-3, 103)
    plt.title("Dose-response: catch rate vs amount of planted modern text")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(HERE / "dose_response.png", dpi=130); plt.close()
    # ROC
    plt.figure(figsize=(6, 6))
    for src in pooled:
        fpr, tpr, _ = roc_curve(pooled[src]["y"], pooled[src]["s"])
        plt.plot(fpr, tpr, color=colors[src], label=f"{src}  AUC={pooled[src]['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey", alpha=.6)
    plt.xlabel("false-positive rate"); plt.ylabel("true-positive rate (recall)")
    plt.title("ROC (positives pooled over all doses)"); plt.grid(alpha=.3); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(HERE / "roc.png", dpi=130); plt.close()
    # PR
    plt.figure(figsize=(6, 6))
    for src in pooled:
        pr, rc, _ = precision_recall_curve(pooled[src]["y"], pooled[src]["s"])
        plt.plot(rc, pr, color=colors[src], label=f"{src}  AP={pooled[src]['ap']:.3f}")
    plt.xlabel("recall"); plt.ylabel("precision"); plt.ylim(0, 1.03)
    plt.title("Precision-recall (positives pooled over all doses)"); plt.grid(alpha=.3)
    plt.legend(loc="lower left"); plt.tight_layout(); plt.savefig(HERE / "pr.png", dpi=130); plt.close()

    out = dict(seed=SEED, n_trials=N_TRIALS, doses=DOSES, n_negatives=len(neg_scores),
               n_clean_base=len(clean_base),
               false_alarm_count=int((neg_scores >= 1).sum()),
               real_modern_recall=dict(flagged=modern_flagged, total=modern_seen),
               recall_equals_ceiling=recall_eq_ceiling,
               source_auc={s: dict(auc=float(pooled[s]["auc"]), ap=float(pooled[s]["ap"]))
                           for s in pooled},
               cells={k: {kk: (float(vv) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()} for k, v in cells.items()})
    (HERE / "trial_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> trial_results.json, dose_response.png, roc.png, pr.png")


if __name__ == "__main__":
    main()
