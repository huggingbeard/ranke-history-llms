"""
confusion_matrix.py — 2x2 of the detector vs the books' own year labels (the 20% sweep).

Ground truth (imperfect): each book's stated `year`.  actual MODERN = year >= 1946 ;
actual OLD = year <= 1945.  Predicted MODERN = the scanner flagged it (>=1 marker).

We already have the flagged ids from the 28k sweep (flagged_20pct.jsonl). To fill the
not-flagged cells we only need every book's year, so we stream the same 28k books reading
just id+year (no scanning) and cross-reference the flags. Fast.

CAVEAT printed with the result: `year` is not perfect truth. A flagged book whose label says
1915 may be a genuinely MIS-dated modern book (a real catch), not a false alarm. So the
"false positive" cell mixes true mistakes (e.g. NASA = Hebrew root) with genuine mislabels.
"""
import json, os, sys
from pathlib import Path
import aho_scan as A

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
N = 28000
HERE = Path(__file__).resolve().parent


def main():
    tok = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read_text().strip()
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_TOKEN"] = tok
    from datasets import load_dataset

    flagged = {str(json.loads(l)["id"]) for l in open(HERE / "flagged_20pct.jsonl", encoding="utf-8")}
    print(f"flagged ids loaded: {len(flagged)}")

    ds = load_dataset("storytracer/LoC-PD-Books", split="train", streaming=True, token=tok)
    # cells[(actual, predicted)] ; actual in {modern,old}; predicted in {flag,clean}
    cells = {("modern", "flag"): 0, ("modern", "clean"): 0,
             ("old", "flag"): 0, ("old", "clean"): 0}
    no_year = 0
    fn_examples, n = [], 0
    for rec in ds:
        if n >= N: break
        n += 1
        rid = str(rec.get("lccn") or rec.get("filename") or rec.get("ocaid") or n)
        yr = rec.get("year")
        if not isinstance(yr, int):
            no_year += 1; continue
        actual = "modern" if yr >= 1946 else "old"
        pred = "flag" if rid in flagged else "clean"
        cells[(actual, pred)] += 1
        if actual == "modern" and pred == "clean" and len(fn_examples) < 12:
            fn_examples.append((yr, str(rec.get("title"))[:50]))
        if n % 4000 == 0:
            print(f"  tallied {n} ...", file=sys.stderr, flush=True)

    TP = cells[("modern", "flag")]; FN = cells[("modern", "clean")]
    FP = cells[("old", "flag")];    TN = cells[("old", "clean")]
    modern = TP + FN; old = FP + TN
    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    rec_ = TP / modern if modern else float("nan")
    spec = TN / old if old else float("nan")
    f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else float("nan")

    print("\n" + "=" * 60)
    print(f"2x2 — detector vs stated year  ({n:,} books; {no_year:,} had no year, excluded)")
    print("=" * 60)
    print(f"{'':>22}{'FLAGGED (pred modern)':>24}{'not flagged (pred old)':>24}")
    print(f"{'actual MODERN (>=1946)':>22}{('TP = '+format(TP,',')):>24}{('FN = '+format(FN,',')):>24}")
    print(f"{'actual OLD (<=1945)':>22}{('FP = '+format(FP,',')):>24}{('TN = '+format(TN,',')):>24}")
    print(f"\ncolumn totals: flagged={TP+FP:,}   not-flagged={FN+TN:,}")
    print(f"row totals:    modern={modern:,}   old={old:,}")
    print("\n--- scores (against year-as-truth) ---")
    print(f"recall  (modern books caught)         : {rec_*100:5.1f}%   ({TP}/{modern})")
    print(f"precision (flags that are truly >=1946): {prec*100:5.1f}%   ({TP}/{TP+FP})")
    print(f"specificity (old books left alone)    : {spec*100:5.2f}%   ({TN}/{old})")
    print(f"false-alarm rate on old books         : {100*FP/old:5.2f}%   ({FP}/{old})")
    print(f"F1                                    : {f1*100:5.1f}%")
    print("\nFN examples — modern books we MISSED (year says >=1946, scanner clean):")
    for yr, t in fn_examples:
        print(f"   [{yr}] {t}")
    print("\nCAVEAT: 'year' is imperfect truth. Some FP-cell books are genuinely MIS-dated")
    print("modern books (real catches), not mistakes; and some FN-cell books are short/plain")
    print("modern texts with no period vocabulary (undetectable by words alone).")

    (HERE / "confusion_matrix.json").write_text(json.dumps(
        {"n": n, "no_year": no_year, "TP": TP, "FN": FN, "FP": FP, "TN": TN,
         "recall": rec_, "precision": prec, "specificity": spec, "f1": f1}, indent=2))
    print(f"\nsaved -> confusion_matrix.json")


if __name__ == "__main__":
    main()
