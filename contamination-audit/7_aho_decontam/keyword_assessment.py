"""
keyword_assessment.py — a structured quality assessment of the keyword list itself.

We had no precision/recall/F1 for the list-building step. This measures it against a practical,
measurable ground truth: a marker is a GOOD tripwire if it fires on ~no genuinely-old books.

  Run every KEPT marker (with its guards) and every REJECTED candidate over N old books
  (stated year <= 1929). Per marker: false-positive rate = fraction of old books it fires on.
  clean  = fires on < 0.05% of old books ;  leaky = fires more.

Keep/reject decision, scored against "is it clean on real old books?":
  TP kept&clean   FP kept&leaky   FN rejected&clean   TN rejected&leaky  ->  precision, recall, F1.
Also: overall specificity of the deployed list (share of old books it leaves alone).
"""
import os, re, sys, json
from collections import Counter
from pathlib import Path
import ahocorasick, aho_scan as A

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
N_OLD, CAP = 25000, 30000   # large enough to actually surface the rare homograph leaks
CLEAN = 0.0005   # <0.05% of old books = clean tripwire


def main():
    tok = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read_text().strip()
    os.environ["HF_HUB_DISABLE_SYMLINKS"]="1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
    os.environ["HF_TOKEN"]=tok
    from datasets import load_dataset

    kept, contexts = A.load_markers(); autoK = A.build_automaton(kept)
    rej = [r["term"].lower() for r in json.loads((A.HERE/"markers_rejected.json").read_text())
           if r["term"].lower() not in getattr(A, "READMIT", {})]   # re-admitted terms are no longer rejected
    autoR = ahocorasick.Automaton()
    for t in rej: autoR.add_word(t, t)
    autoR.make_automaton()

    def fires_rej(norm):
        out = set()
        for end, term in autoR.iter(norm):
            s = end - len(term) + 1
            b = norm[s-1] if s > 0 else " "; a = norm[end+1] if end+1 < len(norm) else " "
            if not (b.isalnum() or a.isalnum()): out.add(term)
        return out

    print(f"kept markers: {len(kept)} | rejected candidates: {len(rej)}\nstreaming old books ...")
    ds = load_dataset("storytracer/LoC-PD-Books", split="train", streaming=True, token=tok)
    fpK, fpR = Counter(), Counter(); n_old = flagged_old = 0
    for i, rec in enumerate(ds):
        if i >= CAP or n_old >= N_OLD: break
        yr, txt = rec.get("year"), A.doc_text(rec)
        if not isinstance(yr, int) or yr > 1929 or not txt: continue
        n_old += 1
        norm = A.normalize(txt)
        kh = {m for m, _ in A.scan_text(norm, autoK, contexts)}
        if kh: flagged_old += 1
        for m in kh: fpK[m] += 1
        for m in fires_rej(norm): fpR[m] += 1
        if n_old % 3000 == 0: print(f"  {n_old} old books ...", file=sys.stderr, flush=True)

    kept_clean = [m for m in kept if fpK[m] / n_old < CLEAN]
    kept_leaky = sorted([m for m in kept if fpK[m] / n_old >= CLEAN], key=lambda m: -fpK[m])
    rej_leaky = [t for t in rej if fpR[t] / n_old >= CLEAN]      # correctly rejected (does fire)
    rej_clean = [t for t in rej if fpR[t] / n_old < CLEAN]       # rejected but clean here

    TP, FP, FN, TN = len(kept_clean), len(kept_leaky), len(rej_clean), len(rej_leaky)
    prec = TP/(TP+FP); rec = TP/(TP+FN); f1 = 2*prec*rec/(prec+rec)

    print(f"\n{'='*70}\nKEYWORD-LIST ASSESSMENT  (ground truth = clean on {n_old:,} real pre-1929 books)\n{'='*70}")
    print(f"deployed list flags {flagged_old}/{n_old} old books ({100*flagged_old/n_old:.2f}%)  "
          f"-> specificity {100*(1-flagged_old/n_old):.2f}%")
    print(f"\nKEPT markers: {len(kept_clean)}/{len(kept)} are clean tripwires "
          f"({100*len(kept_clean)/len(kept):.0f}%);  {len(kept_leaky)} leak (>= {CLEAN*100:.2f}% of old books):")
    for m in kept_leaky:
        print(f"   {m:<22} fires on {fpK[m]:>4}/{n_old} old books ({100*fpK[m]/n_old:.2f}%)")
    firing = sorted([m for m in kept if fpK[m] > 0], key=lambda m: -fpK[m])
    print(f"\n  (transparency) kept markers that fire on ANY old book — {len(firing)} of {len(kept)}, top 15:")
    for m in firing[:15]:
        print(f"     {m:<22} {fpK[m]:>4}/{n_old} ({100*fpK[m]/n_old:.3f}%)")
    print(f"\nREJECTED candidates: {len(rej_leaky)}/{len(rej)} do fire on old books (rejection justified); "
          f"{len(rej_clean)} were clean here.")
    print(f"  most-firing rejects:", ", ".join(f"{t}({fpR[t]})" for t in sorted(rej, key=lambda t:-fpR[t])[:8]))
    print(f"\nKEEP/REJECT DECISION vs 'clean on real old books':")
    print(f"  TP kept&clean={TP}  FP kept&leaky={FP}  FN rejected&clean={FN}  TN rejected&leaky={TN}")
    print(f"  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    json.dump({"n_old": n_old, "kept": len(kept), "kept_clean": len(kept_clean),
               "kept_leaky": {m: fpK[m] for m in kept_leaky}, "flagged_old_pct": round(100*flagged_old/n_old,3),
               "precision": prec, "recall": rec, "f1": f1, "TP":TP,"FP":FP,"FN":FN,"TN":TN},
              open(A.HERE/"keyword_assessment.json","w"), indent=1)
    print("\nsaved -> keyword_assessment.json")


if __name__ == "__main__":
    main()
