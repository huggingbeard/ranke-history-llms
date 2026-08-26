"""
evaluate.py — everything measured against CLEAN external ground truth (OpenLibrary first-pub year).

Runs on whatever ol_labels.json has so far (partial or complete). Reports:
  PART 1  coverage + the metadata error rate, BOTH directions (meta bucket vs OL bucket).
  PART 2  the dangerous case: books OpenLibrary dates post-1945 but the dataset calls old
          ('modern hiding in the old pile') -> do the cheap content signals catch them? This is
          the REAL recall test, on real contamination, with a trustworthy label.
  PART 3  the escalation flag: very-low-English word-lists (text classifiers are blind to them)
          -> how many, and do they cover the genuine-modern residual.
Labels: clean = OL first_publish_year (old <=1945 / modern >=1946); books with no OL match are
excluded from rate estimates and counted as coverage loss (honest denominator).
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

d = json.loads((HERE / "features_v2.json").read_text(encoding="utf-8"))
lab = json.loads((HERE / "ol_labels.json").read_text(encoding="utf-8"))
BY = {r["id"]: r for r in d["modern"] + d["old"]}
EN = set(("the of and to in a is that for it as was with his he be not by but this all at which have "
          "or from are on an their we you i had has said him her they were").split())


def is_latin(c):
    o = ord(c); return (65 <= o <= 90) or (97 <= o <= 122) or (0xC0 <= o <= 0x24F)


def ratios(t):
    alpha = [c for c in t if c.isalpha()]
    if not alpha: return 0.0, 0.0
    nl = sum(1 for c in alpha if not is_latin(c)) / len(alpha)
    toks = [w.strip(".,;:!?()[]\"'").lower() for w in t.split()]
    en = sum(1 for w in toks if w in EN) / max(len(toks), 1)
    return nl, en


# join clean labels
M = []   # matched books: dict(id, meta_year, ol_year, meta_mod, ol_mod, yc_raw, kw_raw, en, nl, ...)
for i, v in lab.items():
    if v["ol_year"] is None or i not in BY:
        continue
    r = BY[i]; nl, en = ratios(r["tx"])
    M.append({"id": i, "meta_year": v["meta_year"], "ol_year": v["ol_year"],
              "meta_mod": v["meta_year"] >= 1946, "ol_mod": v["ol_year"] >= 1946,
              "yc_raw": r["yc_raw"], "yc_ctx": r["yc_ctx"], "kw_raw": len(r["mk_raw"]),
              "kw_dep": len(r["mk_dep"]), "en": en, "nl": nl, "title": v.get("ol_title")})
n_tot = len(lab); n_match = len(M)
print(f"labels so far: {n_tot} looked up | {n_match} matched to an OpenLibrary first-pub year "
      f"({100*n_match/max(n_tot,1):.1f}% coverage)\n")

# ---- PART 1: metadata error rate, both directions ----
tp = sum(1 for m in M if m["meta_mod"] and m["ol_mod"])       # meta modern & truly modern
fp = sum(1 for m in M if m["meta_mod"] and not m["ol_mod"])   # meta modern but truly OLD (our residual case)
fn = sum(1 for m in M if not m["meta_mod"] and m["ol_mod"])   # meta OLD but truly MODERN (dangerous)
tn = sum(1 for m in M if not m["meta_mod"] and not m["ol_mod"])
meta_mod = tp + fp; meta_old = fn + tn
print("=== PART 1: is the dataset's `year` trustworthy? (meta bucket vs OpenLibrary truth) ===")
print(f"  metadata-MODERN books: {meta_mod:>6} | truly modern {tp}, truly OLD (mislabeled) {fp}"
      f"  -> {100*fp/max(meta_mod,1):.1f}% of 'modern' are really old")
print(f"  metadata-OLD    books: {meta_old:>6} | truly old {tn}, truly MODERN (mislabeled) {fn}"
      f"  -> {100*fn/max(meta_old,1):.2f}% of 'old' are really modern")

# ---- PART 2: modern hiding in the old pile -- do content signals catch them? ----
hidden = [m for m in M if not m["meta_mod"] and m["ol_mod"]]
print(f"\n=== PART 2: REAL contamination — {len(hidden)} books OpenLibrary dates post-1945 but the "
      f"dataset files as old ===")
if hidden:
    caught_year = sum(1 for m in hidden if m["yc_raw"] >= 1)
    caught_kw = sum(1 for m in hidden if m["kw_raw"] >= 1)
    caught_any = sum(1 for m in hidden if m["yc_raw"] >= 1 or m["kw_raw"] >= 1)
    print(f"  caught by a printed post-1945 year: {caught_year}/{len(hidden)}")
    print(f"  caught by a keyword/marker        : {caught_kw}/{len(hidden)}")
    print(f"  caught by EITHER (content recall) : {caught_any}/{len(hidden)} = {100*caught_any/len(hidden):.1f}%")
    print("  the ones MISSED by content (would need the escalation flag / external date):")
    for m in hidden:
        if not (m["yc_raw"] >= 1 or m["kw_raw"] >= 1):
            print(f"    OL={m['ol_year']} meta={m['meta_year']} en={m['en']*100:.0f}% nl={m['nl']*100:.0f}%  {(m['title'] or '')[:44]}")

# ---- PART 3: escalation flag (very-low-English word-lists) ----
low = [m for m in M if m["en"] < 0.13]
low_mod = [m for m in low if m["ol_mod"]]
print(f"\n=== PART 3: escalation flag — English-word ratio < 13% (text classifiers are blind here) ===")
print(f"  flagged {len(low)}/{n_match} books; of these {len(low_mod)} are truly modern by OpenLibrary")
print(f"  cost: {len(low)-len(low_mod)} truly-old books also flagged for review "
      f"({100*(len(low)-len(low_mod))/max(len(low),1):.0f}% of the flag)")

json.dump({"n_lookup": n_tot, "n_match": n_match, "coverage": n_match/max(n_tot,1),
           "meta_modern_but_old": fp, "meta_modern_total": meta_mod,
           "meta_old_but_modern": fn, "meta_old_total": meta_old,
           "hidden_modern": len(hidden),
           "hidden_caught_by_content": (caught_any if hidden else 0),
           "lowEN_flagged": len(low), "lowEN_truly_modern": len(low_mod)},
          open(HERE / "evaluate_results.json", "w"), indent=1)
print("\nsaved -> evaluate_results.json")
