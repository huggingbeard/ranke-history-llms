# -*- coding: utf-8 -*-
"""Measure EXPOSURE from text, not from the model's answers, to settle the PI's doubt: when the meter calls a
behaviourally-"unknown" real treaty "remembered", is it (a) leaking realness, or (b) catching a faint trace the model
really has? We proxy exposure by how often "Treaty of X" appears in books BEFORE 1929 (Google Books Ngrams, en-2019,
years 1800-1928 — period-matched to the time-lock). Obscure treaties only appear under anglicised spellings, so we query
several variants and take the max (else they falsely read zero). Then we line exposure up against the meter and the
behavioural memory label and write a CSV.
Run:  python src/ngram_exposure.py
"""
import os, sys, csv, json, time, math, random, unicodedata, urllib.request, urllib.parse
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import treaties_battery as B

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "superlatives")
CACHE = os.path.join(OUT, "ngram_cache.json")
CHANCE = 1.0 / len(B.PARTY_OPTIONS)
# extra anglicised spellings for treaties whose modern name barely appears in old books
ANGLO = {"Björkö": ["Bjorko", "Bjoerkoe"], "Küçük Kaynarca": ["Kuchuk Kainarji", "Kutchuk Kainardji", "Kainarji"],
         "Hünkâr İskelesi": ["Unkiar Skelessi", "Hunkiar Iskelesi", "Unkiar-Skelessi"], "Åbo": ["Abo"],
         "Niš": ["Nissa", "Nish"], "Żurawno": ["Zurawno"], "Zsitvatorok": ["Sitvatorok", "Sitva-Torok"],
         "Nöteborg": ["Noteborg", "Orekhov"], "Zuhab": ["Kasr-i-Shirin", "Qasr-e Shirin"],
         "Turkmenchay": ["Turkmanchai", "Turkomanchai"], "Gulistan": ["Golestan"], "Passarowitz": ["Passarovitz"],
         "Karlowitz": ["Carlowitz"], "Nystad": ["Nystadt"], "Nerchinsk": ["Nertchinsk"], "Wanghia": ["Wang-hia"],
         "Stolbovo": ["Stolbova"], "Deulino": ["Dywilino"], "Teusina": ["Tyavzino"]}


def asciifold(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def variants(name):
    vs = {name, asciifold(name)} | set(ANGLO.get(name, []))
    return sorted(vs)


cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}


def freq(phrase):
    """Mean pre-1929 book frequency of `phrase`. Retries with backoff; a SUCCESSFUL empty response is a true 0, an
    ERROR is retried (never silently cached as 0). Raises if it can't get a real answer — no fake zeros."""
    if phrase in cache:
        return cache[phrase]
    q = urllib.parse.urlencode({"content": phrase, "year_start": 1800, "year_end": 1928, "corpus": "en-2019", "smoothing": 3})
    req = urllib.request.Request("https://books.google.com/ngrams/json?" + q,
                                 headers={"User-Agent": "academic-research (joachimvoth@gmail.com)"})
    for t in range(6):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=30))
            v = (sum(d[0]["timeseries"]) / len(d[0]["timeseries"])) if d else 0.0   # d==[] => Google has no data => real 0
            cache[phrase] = v
            time.sleep(0.8 + random.uniform(0, 0.6))
            return v
        except Exception:
            time.sleep((2 ** t) + random.uniform(0, 1))                              # backoff 1,2,4,8,16,32s
    raise RuntimeError(f"ngram fetch failed after retries: {phrase}")


def exposure(name):
    return max(freq("Treaty of " + v) for v in variants(name))


# ---- meter (v2) per treaty, from saved activations ----
X = np.load(os.path.join(OUT, "memory_detector.npz"))["X"]
meta = json.load(open(os.path.join(OUT, "memory_detector_meta.json"), encoding="utf-8"))
bestL = json.load(open(os.path.join(OUT, "memory_detector_summary.json"), encoding="utf-8"))["bestL"]


def label(m):
    if m["kind"] == "invented":
        return "invented", 0
    kd = m["ptrue"] > 0.5
    kp = (m["parties"] is not None) and (m["parties"] > 3 * CHANCE)
    if kd or kp:
        return "known", 1
    if m["peak"] < 0.25 and not kp:
        return "unknown", 0
    return "ambiguous", -1


lab = [label(m) for m in meta]
mem = np.array([l[1] for l in lab])
clean = mem >= 0
pipe = make_pipeline(StandardScaler(), PCA(n_components=15, random_state=0), LogisticRegression(C=1.0, max_iter=3000))
oof = cross_val_predict(pipe, X[clean, bestL, :], mem[clean], cv=5, method="predict_proba")[:, 1]
clf = pipe.fit(X[clean, bestL, :], mem[clean])
meter = np.empty(len(meta)); meter[clean] = oof; meter[~clean] = clf.predict_proba(X[~clean, bestL, :])[:, 1]

# ---- assemble CSV over the REAL treaties ----
print(f"[ngram] querying exposure for {len(B.REAL)} real treaties (cached: {len(cache)}) ...", flush=True)
rows = []
for i, (name, yr, parties) in enumerate(B.REAL):
    mi = next(k for k, m in enumerate(meta) if m["name"] == name and m["kind"] == "real")
    rows.append({"treaty": name, "year": yr, "parties": parties or "(multilateral)",
                 "exposure_pre1929": exposure(name), "date_P_true": round(meta[mi]["ptrue"], 3),
                 "parties_P": round(meta[mi]["parties"], 3) if meta[mi]["parties"] is not None else "",
                 "behavioural_label": lab[mi][0], "meter_v2": round(float(meter[mi]), 3)})
json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
rows.sort(key=lambda r: -r["exposure_pre1929"])
CSVP = os.path.join(OUT, "treaty_exposure.csv")
with open(CSVP, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["treaty", "year", "parties", "exposure_pre1929", "date_P_true", "parties_P", "behavioural_label", "meter_v2"])
    w.writeheader(); w.writerows(rows)

# ---- the key question: does the meter track exposure, especially among behaviourally-UNKNOWN treaties? ----
def logx(v):
    return math.log10(v) if v > 0 else -13.0
unk = [r for r in rows if r["behavioural_label"] == "unknown"]
kno = [r for r in rows if r["behavioural_label"] == "known"]
allr = rows
r_all = np.corrcoef([logx(r["exposure_pre1929"]) for r in allr], [r["meter_v2"] for r in allr])[0, 1]
r_unk = np.corrcoef([logx(r["exposure_pre1929"]) for r in unk], [r["meter_v2"] for r in unk])[0, 1] if len(unk) > 2 else float("nan")
unk_fp = [r for r in unk if r["meter_v2"] > 0.5]     # meter's FALSE POSITIVES (unknown, but called remembered)
unk_ok = [r for r in unk if r["meter_v2"] <= 0.5]    # meter got these right (unknown -> made-up)
med = lambda rs: np.median([r["exposure_pre1929"] for r in rs]) if rs else float("nan")
print(f"\nwrote {CSVP}  ({len(rows)} real treaties)")
print(f"exposure medians:  known {med(kno):.2e}   unknown {med(unk):.2e}")
print(f"corr(log-exposure, meter):  all real {r_all:.2f}")
print(f"\nTHE TEST — among behaviourally-UNKNOWN treaties (model can't answer either question):")
print(f"  meter's FALSE POSITIVES (called remembered), n={len(unk_fp)}:  median exposure {med(unk_fp):.2e}")
print(f"  meter got RIGHT (called made-up),           n={len(unk_ok)}:  median exposure {med(unk_ok):.2e}")
print(f"  -> if the false positives have MORE exposure, the meter is catching faint real traces (b), not pure realness (a).\n")
print(f"{'treaty':17}{'exposure':>11}{'date':>6}{'parties':>8}{'meter':>7}   label")
for r in sorted(unk, key=lambda r: -r["meter_v2"]) + [r for r in kno if r["exposure_pre1929"] < 1e-9][:3]:
    pp = f"{100*float(r['parties_P']):.0f}%" if r["parties_P"] != "" else "  -"
    print(f"{r['treaty']:17}{r['exposure_pre1929']:>11.1e}{100*r['date_P_true']:>5.0f}%{pp:>8}{r['meter_v2']:>7.2f}   {r['behavioural_label']}")
