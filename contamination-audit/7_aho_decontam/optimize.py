"""
optimize.py — tune a lightweight contamination classifier RECALL-FIRST, with OUT-OF-SAMPLE
evaluation so the reported numbers generalise (no overfitting).

Classifier: flag a book if  yc >= a  OR  kw >= b
  yc = #distinct post-1945 years printed ; kw = #keyword/n-gram hits ; (a=99 disables year, b=99 kw).
Objective: recall >= 99.5% on post-1945 books, minimum old-book loss.

ANTI-OVERFITTING PROTOCOL (this file implements it):
  * Books are assigned to K=5 folds by a hash of their id -> deterministic, reproducible.
  * NESTED CV: for each outer test fold, the rule (a,b) is CHOSEN on the other 4 folds only
    (aim >= 99.7% train recall -- a margin over the 99.5% target -- at least old-loss), then applied
    to the held-out fold. So every book is scored by a rule that never saw it.
  * We report the pooled out-of-sample recall/old-loss + a 95% CI, the per-fold chosen rules
    (stability), AND the naive in-sample number (train==test) so the overfitting gap is explicit.
"""
import json, sys, hashlib, math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
K = 5
TARGET, TRAIN_TARGET = 0.995, 0.997
RULES = [(a, b) for a in (1, 2, 3, 4, 99) for b in (1, 2, 3, 99) if not (a == 99 and b == 99)]

d = json.loads((HERE / "features.json").read_text())
for r in d["modern"] + d["old"]:
    r["kw"] = len(r["mk"])
MOD, OLD = d["modern"], d["old"]
fold = lambda r: int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % K
for r in MOD + OLD:
    r["f"] = fold(r)
print(f"loaded {len(MOD)} modern + {len(OLD)} old; K={K} folds by id-hash "
      f"(modern per fold: {[sum(1 for r in MOD if r['f']==k) for k in range(K)]})\n")


def flag(r, a, b): return r["yc"] >= a or r["kw"] >= b
def recall(rows, a, b): return sum(flag(r, a, b) for r in rows) / max(len(rows), 1)


def pick_rule(train_mod, train_old):
    """cheapest rule (min old-loss) reaching TRAIN_TARGET recall on train; else max-recall."""
    scored = [((a, b), recall(train_mod, a, b), recall(train_old, a, b)) for a, b in RULES]
    ok = [s for s in scored if s[1] >= TRAIN_TARGET]
    if ok:
        return min(ok, key=lambda s: s[2])[0]
    return max(scored, key=lambda s: s[1])[0]


def wilson(k, n):
    if n == 0: return (0, 0)
    p = k / n; z = 1.96
    den = 1 + z*z/n; c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/den, (c+h)/den)


# ---- nested CV: choose rule on 4 folds, score the held-out fold ----
oos_mod_caught = oos_old_flagged = 0
chosen = []
for kf in range(K):
    tr_mod = [r for r in MOD if r["f"] != kf]; te_mod = [r for r in MOD if r["f"] == kf]
    tr_old = [r for r in OLD if r["f"] != kf]; te_old = [r for r in OLD if r["f"] == kf]
    a, b = pick_rule(tr_mod, tr_old); chosen.append((a, b))
    oos_mod_caught += sum(flag(r, a, b) for r in te_mod)
    oos_old_flagged += sum(flag(r, a, b) for r in te_old)

rec = oos_mod_caught / len(MOD); ol = oos_old_flagged / len(OLD)
lo, hi = wilson(oos_mod_caught, len(MOD))
print("=== OUT-OF-SAMPLE (nested CV) — the honest numbers ===")
print(f"  recall   = {rec*100:.2f}%   (95% CI {lo*100:.2f}–{hi*100:.2f}%;  {len(MOD)-oos_mod_caught} of {len(MOD)} modern missed)")
print(f"  old-loss = {ol*100:.2f}%   ({oos_old_flagged} of {len(OLD)} old books flagged)")
print(f"  target met out-of-sample? {'YES' if lo >= TARGET else 'NOT CONFIRMED — CI lower bound '+format(lo*100,'.2f')+'% < 99.5%'}")
print(f"  rule chosen per fold: {chosen}  (stable if all the same)")

# ---- naive in-sample (overfit) for comparison ----
a, b = pick_rule(MOD, OLD)
print(f"\n=== IN-SAMPLE (overfit, train==test) — for comparison only ===")
print(f"  best rule on all data: year>={a} OR kw>={b}  -> recall {recall(MOD,a,b)*100:.2f}%  old-loss {recall(OLD,a,b)*100:.2f}%")
print(f"  overfitting gap (in-sample recall − OOS recall): {(recall(MOD,a,b)-rec)*100:+.2f} pts")

# ---- signal grid (in-sample, informational) ----
print(f"\n=== signal grid (in-sample; for intuition, NOT the reported result) ===")
print(f"{'rule':<20}{'recall':>10}{'old-loss':>12}")
for a2 in (1, 2, 3, 99):
    for b2 in (1, 2, 99):
        if a2 == 99 and b2 == 99: continue
        nm = (f"year>={a2}" if a2 != 99 else "") + (" OR " if a2!=99 and b2!=99 else "") + (f"kw>={b2}" if b2 != 99 else "")
        print(f"  {nm:<18}{recall(MOD,a2,b2)*100:>9.2f}%{recall(OLD,a2,b2)*100:>11.2f}%")

# ---- residual + marginal-marker info (computed on all; a data fact / info only) ----
yc0 = [r for r in MOD if r["yc"] == 0]
resid = [r for r in yc0 if not r["mk"]]
print(f"\n=== residual & where keywords help (informational) ===")
print(f"  modern with NO printed post-1945 year: {len(yc0)}/{len(MOD)} ({100*len(yc0)/len(MOD):.1f}%); "
      f"of those a keyword catches {sum(1 for r in yc0 if r['mk'])}.")
print(f"  modern with NO year AND NO keyword (content-blind): {len(resid)}/{len(MOD)} "
      f"-> content-only recall ceiling = {100*(1-len(resid)/len(MOD)):.2f}%")
mc = Counter(m for r in yc0 for m in r["mk"])
print(f"  markers that rescue a year-less modern book (recall-relevant): {dict(mc.most_common(15))}")

json.dump({"oos_recall": rec, "oos_recall_ci": [lo, hi], "oos_old_loss": ol,
           "chosen_per_fold": chosen, "n_modern": len(MOD), "n_old": len(OLD),
           "residual": len(resid), "content_ceiling": 1-len(resid)/len(MOD)},
          open(HERE/"optimize_results.json", "w"), indent=1)
print("\nsaved -> optimize_results.json")
