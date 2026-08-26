"""
build_features2.py — ENRICHED ground-truth feature set + diagnostics.

The v1 features.json cached only a RAW post-1945-year count (any 4-digit number 1946-2029,
no context) and RAW marker hits. That conflated 'the cheap classifier is weak' with 'the year
signal is noisy on old books'. v2 caches several disambiguated variants of BOTH signals so the
optimizer can explore them offline, and pulls the actual text of the failure cases so we can
VERIFY the recall ceiling and the old-book false positives instead of assuming them.

Per book (same sample as v1: modern = metadata year>=1946; up to N_OLD old = year<=1929):
  y         metadata year (the label)
  yc_raw    #distinct post-1945 years, NO context           (v1's yc; noisy)
  yc_ctx    #distinct post-1945 years with a DATE cue nearby (c/copyright/month/in/parens/range)
  yc_front  #distinct post-1945 years in the front matter (first 20%)
  yc_cf     #distinct post-1945 years that are BOTH date-cued AND in the front matter
  mk_raw    raw keyword hits (word-boundary only, all 130 markers) — for offline guard tests
  mk_dep    DEPLOYED keyword hits (EXCLUDE applied + context guards + readmits) — the real list

Diagnostics (separate file, so features stays lean):
  residual  every MODERN book with yc_raw==0 AND no raw marker -> the content-blind residual:
            {id, y, title, head=first 700 chars} so we can check if it is truly post-1945.
  oldyear   a sample of OLD books that DO print a post-1945 year -> {id, y, the year strings +
            +-45 char context} so we can categorise stamp / accession / OCR-noise vs real date.
"""
import os, sys, json, re
from pathlib import Path
import aho_scan as A

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
CAP, N_OLD, CKPT = 80000, 25000, 5000
OUT   = A.HERE / "features_v2.json"
DIAG  = A.HERE / "diagnostics_v2.json"
OLDYEAR_CAP = 400        # how many old-book year-hit examples to capture for categorisation

MARK_RAW, _ = [m["term"].lower() for m in json.loads((A.HERE/"markers_1945.json").read_text())["markers"]], None
import ahocorasick
AUTO_RAW = ahocorasick.Automaton()
for t in MARK_RAW: AUTO_RAW.add_word(t, t)
AUTO_RAW.make_automaton()
MARK_DEP, CONTEXTS = A.load_markers()
AUTO_DEP = A.build_automaton(MARK_DEP)


def raw_hits(norm):
    out = set()
    for end, term in AUTO_RAW.iter(norm):
        s = end - len(term) + 1
        b = norm[s-1] if s > 0 else " "; a = norm[end+1] if end+1 < len(norm) else " "
        if not (b.isalnum() or a.isalnum()): out.add(term)
    return sorted(out)


def year_contexts(norm, cap=6):
    """the actual post-1945 year strings + a small context window, for categorisation."""
    out = []
    for m in A.POST1945.finditer(norm):
        s = m.start()
        out.append({"yr": m.group(0), "cx": norm[max(0, s-45):s+len(m.group(0))+15]})
        if len(out) >= cap: break
    return out


def main():
    tok = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read_text().strip()
    os.environ["HF_HUB_DISABLE_SYMLINKS"]="1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
    os.environ["HF_TOKEN"]=tok
    from datasets import load_dataset
    ds = None
    for _ in range(6):
        try: ds = load_dataset("storytracer/LoC-PD-Books", split="train", streaming=True, token=tok); break
        except Exception as e: print("load retry:", repr(e)[:80], file=sys.stderr); import time; time.sleep(8)

    modern, old, n = [], [], 0
    residual, oldyear = [], []
    def save():
        OUT.write_text(json.dumps({"markers_raw": MARK_RAW, "markers_dep": MARK_DEP,
                                   "n_streamed": n, "modern": modern, "old": old}), encoding="utf-8")
        DIAG.write_text(json.dumps({"residual": residual, "oldyear": oldyear}, ensure_ascii=False),
                        encoding="utf-8")
    for rec in ds:
        if n >= CAP: break
        n += 1
        yr, txt = rec.get("year"), A.doc_text(rec)
        if not isinstance(yr, int) or not txt: continue
        is_mod = yr >= 1946; is_old = yr <= 1929
        if not (is_mod or (is_old and len(old) < N_OLD)): continue
        norm = A.normalize(txt)
        rid = str(rec.get("lccn") or rec.get("filename") or rec.get("ocaid") or n)
        mk_raw = raw_hits(norm)
        mk_dep = sorted({t for t, _ in A.scan_text(norm, AUTO_DEP, CONTEXTS)})
        row = {"id": rid, "y": yr,
               "yc_raw":   len(A.post1945_years(norm)),
               "yc_ctx":   len(A.post1945_years(norm, require_context=True)),
               "yc_front": len(A.post1945_years(norm, front_frac=0.2)),
               "yc_cf":    len(A.post1945_years(norm, require_context=True, front_frac=0.2)),
               "mk_raw": mk_raw, "mk_dep": mk_dep,
               "tx": norm[:4000]}   # cached text -> enables offline TF-IDF / embedding tests, no re-pull
        (modern if is_mod else old).append(row)
        # diagnostics
        if is_mod and row["yc_raw"] == 0 and not mk_raw:
            residual.append({"id": rid, "y": yr, "title": rec.get("title"), "head": norm[:700]})
        if is_old and row["yc_raw"] >= 1 and len(oldyear) < OLDYEAR_CAP:
            oldyear.append({"id": rid, "y": yr, "yc_raw": row["yc_raw"], "yc_ctx": row["yc_ctx"],
                            "years": year_contexts(norm)})
        if n % CKPT == 0:
            save(); print(f"  {n} streamed | modern={len(modern)} old={len(old)} "
                          f"resid={len(residual)} oldyr={len(oldyear)}", file=sys.stderr, flush=True)
    save()
    print(f"\nDONE: streamed {n} | modern={len(modern)} old={len(old)} | "
          f"residual={len(residual)} oldyear_samples={len(oldyear)}")
    print(f"saved -> {OUT}\nsaved -> {DIAG}")


if __name__ == "__main__":
    main()
