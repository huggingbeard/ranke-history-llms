"""
confidence_score.py — a tunable confidence score for "let no modern book through".

Score per book (content only, ignoring the date label so it also catches MIS-dated books):
    confidence = (#distinct post-1945 tripwire words) + (#distinct post-1945 years printed)
A book is flagged if confidence >= T (or via the discrete policies below). Sweeping T trades
recall (modern books caught) against over-removal (genuinely old books wrongly flagged).

Positives = books with stated year >= 1946 (modern).  Negatives = year <= 1945 (old).
Outputs a recall-vs-over-removal table, the ROC over the score, and confidence_score.png.
Saves the raw per-book (n_vocab, n_years) so thresholds can be re-swept instantly.
"""
import json, os, re, sys, time
from pathlib import Path
import numpy as np, aho_scan as A
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
N_MOD, N_OLD, CAP = 250, 12000, 16000   # big OLD sample so rare word false-positives show up
POSTYEAR = re.compile(r"\b(19(?:4[6-9]|[5-9]\d)|20[0-2]\d)\b")


def main():
    tok = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read_text().strip()
    os.environ["HF_HUB_DISABLE_SYMLINKS"]="1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
    os.environ["HF_TOKEN"]=tok
    from datasets import load_dataset
    m, ctx = A.load_markers(); auto = A.build_automaton(m)

    def sig(text):
        norm = A.normalize(text)
        v = len({mk for mk, _ in A.scan_text(norm, auto, ctx)})
        y = len(set(POSTYEAR.findall(norm)))
        return v, y

    ds = None
    for attempt in range(6):
        try:
            ds = load_dataset("storytracer/LoC-PD-Books", split="train", streaming=True, token=tok); break
        except Exception as e:
            print(f"  load attempt {attempt+1} failed; retrying...", file=sys.stderr, flush=True); time.sleep(8)
    mod, old, n = [], [], 0
    for rec in ds:
        if n >= CAP or (len(mod) >= N_MOD and len(old) >= N_OLD): break
        n += 1
        yr, txt = rec.get("year"), A.doc_text(rec)
        if not isinstance(yr, int) or not txt: continue
        if yr >= 1946 and len(mod) < N_MOD: mod.append(sig(txt))
        elif yr <= 1945 and len(old) < N_OLD: old.append(sig(txt))
        if n % 5000 == 0: print(f"  streamed {n}: modern={len(mod)} old={len(old)}", file=sys.stderr, flush=True)
    (HERE / "confidence_raw.json").write_text(json.dumps({"modern": mod, "old": old}))
    print(f"\nmodern={len(mod)}  old={len(old)}")

    # ---- discrete policy table ----
    def rate(rows, pred): return 100*sum(1 for r in rows if pred(r))/max(len(rows),1)
    print(f"\n{'policy':<40}{'RECALL (modern)':>16}{'over-removal (old)':>20}")
    policies = [("vocabulary only (>=1 word)", lambda r: r[0]>=1)]
    for k in (1,2,3,4):
        policies.append((f"vocab OR >= {k} post-1945 year(s)", (lambda k: lambda r: r[0]>=1 or r[1]>=k)(k)))
    for name, pred in policies:
        print(f"{name:<40}{rate(mod,pred):>15.1f}%{rate(old,pred):>19.1f}%")

    # ---- continuous confidence score = vocab + years ----
    sc_mod = np.array([v+y for v,y in mod]); sc_old = np.array([v+y for v,y in old])
    y_true = np.r_[np.ones(len(sc_mod)), np.zeros(len(sc_old))]
    y_sc   = np.r_[sc_mod, sc_old]
    auc = roc_auc_score(y_true, y_sc)
    print(f"\nconfidence score = (#vocab words)+(#post-1945 years):  ROC-AUC = {auc:.3f}")
    print(f"{'threshold T (flag if score>=T)':<34}{'RECALL':>10}{'over-removal':>14}")
    for T in (1,2,3,4,5,6,8,10):
        rc = 100*(sc_mod>=T).mean(); fp = 100*(sc_old>=T).mean()
        print(f"  score >= {T:<27}{rc:>9.1f}%{fp:>13.1f}%")

    # ---- plot: recall vs over-removal as the dial turns ----
    fpr, tpr, thr = roc_curve(y_true, y_sc)
    plt.figure(figsize=(7,6))
    plt.plot(fpr*100, tpr*100, "-o", color="#2b5dab", ms=3)
    for T in (1,2,3,5):
        rc=100*(sc_mod>=T).mean(); fp=100*(sc_old>=T).mean()
        plt.annotate(f"T={T}", (fp, rc), textcoords="offset points", xytext=(6,-4), fontsize=9, color="#b32424")
        plt.plot([fp],[rc],"s",color="#b32424",ms=6)
    plt.xlabel("over-removal: % of genuinely OLD books wrongly flagged")
    plt.ylabel("recall: % of MODERN books caught (from content alone)")
    plt.title(f"Confidence-score dial: recall vs over-removal  (AUC={auc:.3f})\n"
              "score = #post-1945 words + #distinct post-1945 years")
    plt.grid(alpha=.3); plt.xlim(-1,40); plt.ylim(0,101); plt.tight_layout()
    plt.savefig(HERE/"confidence_score.png", dpi=130); plt.close()
    print("\nsaved -> confidence_score.png, confidence_raw.json")


if __name__ == "__main__":
    main()
