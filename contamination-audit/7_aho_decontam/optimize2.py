"""
optimize2.py — recall-first tuning on the ENRICHED feature set, out-of-sample (nested CV).

Two knobs beyond the v1 (a,b) thresholds:
  YEAR variant  : yc_raw (any recent 4-digit number) | yc_ctx (date-cued) | yc_front (front
                  matter) | yc_cf (both). Context/front-matter variants aim to kill stray
                  accession/OCR numbers on OLD books without dropping real publication years.
  MARKER variant: mk_raw (all 130, boundary only) | mk_dep (deployed: EXCLUDE + context guards).

Classifier: flag if  year_variant >= a  OR  kw_variant >= b.
Reported numbers are OUT-OF-SAMPLE: for each of K=5 folds the rule (a,b) is chosen on the other
four (aim >= 99.7% train recall, then least old-loss) and scored on the held-out fold. Every book
is scored by a rule it never trained on. We print, per (year,marker) variant, the OOS recall and
old-loss, so the best recall/old-loss trade is picked on held-out data, not in-sample.
"""
import json, sys, hashlib, math
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
K = 5
TARGET, TRAIN_TARGET = 0.995, 0.997

d = json.loads((HERE / "features_v2.json").read_text())
MOD, OLD = d["modern"], d["old"]
for r in MOD + OLD:
    r["kwr"] = len(r["mk_raw"]); r["kwd"] = len(r["mk_dep"])
    r["f"] = int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % K
YEARS  = ["yc_raw", "yc_ctx", "yc_front", "yc_cf"]
MARKS  = [("mk_raw", "kwr"), ("mk_dep", "kwd")]
A_GRID = (1, 2, 3, 99)   # 99 disables the year signal
B_GRID = (1, 2, 99)      # 99 disables the marker signal
print(f"loaded {len(MOD)} modern + {len(OLD)} old  (K={K} folds by id-hash)\n")


def flag(r, yv, kv, a, b): return r[yv] >= a or r[kv] >= b
def rate(rows, yv, kv, a, b): return sum(flag(r, yv, kv, a, b) for r in rows) / max(len(rows), 1)


def wilson(k, n):
    if n == 0: return (0, 0)
    p = k/n; z = 1.96; den = 1+z*z/n; c = p+z*z/(2*n); h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-h)/den, (c+h)/den)


def pick(tr_mod, tr_old, yv, kv):
    scored = [((a, b), rate(tr_mod, yv, kv, a, b), rate(tr_old, yv, kv, a, b))
              for a in A_GRID for b in B_GRID if not (a == 99 and b == 99)]
    ok = [s for s in scored if s[1] >= TRAIN_TARGET]
    return (min(ok, key=lambda s: s[2]) if ok else max(scored, key=lambda s: s[1]))[0]


def nested_cv(yv, kv):
    mc = of = 0; chosen = []
    for kf in range(K):
        trm = [r for r in MOD if r["f"] != kf]; tem = [r for r in MOD if r["f"] == kf]
        tro = [r for r in OLD if r["f"] != kf]; teo = [r for r in OLD if r["f"] == kf]
        a, b = pick(trm, tro, yv, kv); chosen.append((a, b))
        mc += sum(flag(r, yv, kv, a, b) for r in tem)
        of += sum(flag(r, yv, kv, a, b) for r in teo)
    return mc/len(MOD), of/len(OLD), chosen, mc


print("=== OUT-OF-SAMPLE recall / old-loss by signal variant (nested CV) ===")
print(f"{'year signal':<10}{'markers':<9}{'recall':>9}{'old-loss':>10}   rule/fold (a=year thr, b=kw thr)")
best = None
for yv in YEARS:
    for mk, kv in MARKS:
        rec, ol, ch, mc = nested_cv(yv, kv)
        lo, hi = wilson(mc, len(MOD))
        tag = "" if len({c for c in ch}) == 1 else "  (unstable)"
        print(f"{yv:<10}{mk:<9}{rec*100:>8.2f}%{ol*100:>9.2f}%   {ch[0]}{tag}")
        # recall-first pick: maximise recall, then minimise old-loss
        key = (round(rec, 4), -ol)
        if best is None or key > best[0]:
            best = (key, yv, mk, kv, rec, ol, lo, hi, ch)

_, yv, mk, kv, rec, ol, lo, hi, ch = best
print(f"\n=== RECALL-FIRST PICK (max OOS recall, then min old-loss) ===")
print(f"  signal   : year={yv}  OR  markers={mk}   rule/fold {ch}")
print(f"  recall   = {rec*100:.2f}%  (95% CI {lo*100:.2f}-{hi*100:.2f}%;  {len(MOD)-round(rec*len(MOD))} of {len(MOD)} modern missed)")
print(f"  old-loss = {ol*100:.2f}%  ({round(ol*len(OLD))} of {len(OLD)} old flagged)")
print(f"  99.5% target met out-of-sample? {'YES' if lo >= TARGET else 'NO — content ceiling below target (CI lo '+format(lo*100,'.2f')+'%)'}")

# structural ceiling: modern books with NO signal at all under the BEST variant
resid = [r for r in MOD if r[yv] == 0 and r[kv] == 0]
print(f"\n=== structural recall ceiling (content-blind books) ===")
print(f"  modern with year={yv}==0 AND markers={mk}==0 : {len(resid)}/{len(MOD)} "
      f"-> ceiling {100*(1-len(resid)/len(MOD)):.2f}% (no threshold can catch these)")

# where does old-loss come from under raw vs context year (the diagnosis)?
print(f"\n=== year-signal old-loss diagnosis (share of OLD books with >=1 post-1945 year) ===")
for yvar in YEARS:
    print(f"  {yvar:<9}: {100*sum(r[yvar]>=1 for r in OLD)/len(OLD):>6.2f}% of old  |  "
          f"catches {100*sum(r[yvar]>=1 for r in MOD)/len(MOD):>6.2f}% of modern")

json.dump({"best_year": yv, "best_marker": mk, "oos_recall": rec, "oos_recall_ci": [lo, hi],
           "oos_old_loss": ol, "rule_per_fold": ch, "ceiling": 1-len(resid)/len(MOD),
           "n_modern": len(MOD), "n_old": len(OLD)},
          open(HERE/"optimize2_results.json", "w"), indent=1)
print("\nsaved -> optimize2_results.json")
