# -*- coding: utf-8 -*-
"""PANEL STATISTICS (locked in docs/PREREGISTRATION_FORESIGHT_v2.md).

Per event x language (and pooled across languages with a cluster bootstrap over translated item
triplets):
  * in-family Dconc = strain_pre - strain_anchor (mean, item bootstrap B=10k, seeded);
  * bridge-corrected Dconc for modern-anchored events:
        offset_lang = mean over anchor-1929 event items of (strain_1929 - strain_modern)
        Dconc_bridge = (strain_pre - strain_modern) - offset_lang     [joint bootstrap]
  * permutation p vs the Tier-B floor (same language, same estimator), B=10k;
  * BH-FDR across the 8 events within each (pre-model, language) family;
  * effect size in Tier-B SD units; leave-one-out max influence;
  * true/false discrimination: % items with c(true)>c(false) and mean margin, per model.

Reads results/panel_scores.json + results/gate_panel.json; writes results/stats_panel.json.
Run: python src/stats_panel.py
"""
import os, json, random, statistics as st
import common
from panel_events import EVENTS_META, ITEMS, LANGS
from panel_checks import TRUEFALSE

B = 10_000
random.seed(20260704)

S = json.load(open(os.path.join(common.RESULTS, "panel_scores.json"), encoding="utf-8"))
PRES = ["1900", "1913", "1929"]
BRIDGE_EVENTS = [ev for ev, m in EVENTS_META.items() if m["anchor"] == "1929"]  # offset estimated here


def strain(tag, key):
    return S[tag]["strain"][key][2]


def keys(ev, lang, n=None):
    n = n if n is not None else len(ITEMS[ev]) if ev in ITEMS else (6 if ev == "tierB" else 5)
    return [f"{ev}#{i}/{lang}" for i in range(n)]


def mean_ci(vals, Bn=B):
    m = st.mean(vals)
    n = len(vals)
    boots = sorted(st.mean([vals[random.randrange(n)] for _ in range(n)]) for _ in range(Bn))
    return round(m, 3), round(boots[int(.025 * Bn)], 3), round(boots[int(.975 * Bn)], 3)


def perm_p(a, b, Bn=B):
    """Two-sided permutation p for mean(a) - mean(b)."""
    obs = st.mean(a) - st.mean(b)
    pool = a + b
    na = len(a)
    hits = 0
    for _ in range(Bn):
        random.shuffle(pool)
        if abs(st.mean(pool[:na]) - st.mean(pool[na:])) >= abs(obs) - 1e-12:
            hits += 1
    return max(1, hits) / Bn


def bh(pvals):
    """Benjamini-Hochberg q-values for a dict name->p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    q, prev = {}, 1.0
    for rank in range(m, 0, -1):
        name, p = items[rank - 1]
        prev = min(prev, p * m / rank)
        q[name] = round(prev, 4)
    return q


# ---------- per-item Dconc under the event's own estimator ----------
def item_dconc(ev, lang, pre, offset=None):
    """List of per-item Dconc values. offset: per-lang bridge offset (required for modern anchor)."""
    anchor = EVENTS_META[ev]["anchor"] if ev in EVENTS_META else "1929"
    out = []
    for k in keys(ev, lang):
        if anchor == "modern":
            out.append(strain(pre, k) - strain("modern", k) - offset)
        else:
            out.append(strain(pre, k) - strain(anchor, k))
    return out


# ---------- bridge offset per language (and its bootstrap draws) ----------
def offset_items(lang):
    ks = [k for ev in BRIDGE_EVENTS for k in keys(ev, lang)]
    return [strain("1929", k) - strain("modern", k) for k in ks]


OFFSET = {lang: st.mean(offset_items(lang)) for lang in LANGS}
OFFSET_CI = {lang: mean_ci(offset_items(lang)) for lang in LANGS}

# ---------- Tier-B floor / Tier-C ceiling per estimator ----------
def tier_vals(tier, lang, pre, anchor):
    n = 6 if tier == "tierB" else 5
    out = []
    for k in keys(tier, lang, n):
        if anchor == "modern":
            out.append(strain(pre, k) - strain("modern", k) - OFFSET[lang])
        else:
            out.append(strain(pre, k) - strain(anchor, k))
    return out


report = {"offset_bridge": {lg: {"mean": round(OFFSET[lg], 3), "ci": OFFSET_CI[lg]} for lg in LANGS},
          "events": {}, "tiers": {}, "pooled": {}, "tf": {}, "fdr": {}, "loo": {}}

# ---------- per event x lang x pre ----------
pperm_store = {}   # (pre, lang) -> {event: p}
for ev, meta in EVENTS_META.items():
    anchor = meta["anchor"]
    report["events"][ev] = {"anchor": anchor, "label": meta["label"], "per_lang": {}}
    for lang in LANGS:
        d = {}
        for pre in meta["pre"]:
            vals = item_dconc(ev, lang, pre, OFFSET.get(lang))
            m, lo, hi = mean_ci(vals)
            floor = tier_vals("tierB", lang, pre, anchor)
            p = perm_p(vals, floor)
            sd = st.pstdev(floor) or 1.0
            es = round((st.mean(vals) - st.mean(floor)) / sd, 2)
            # leave-one-out max influence on the mean
            loo = max(abs(st.mean(vals) - st.mean(vals[:i] + vals[i+1:])) for i in range(len(vals)))
            d[pre] = {"mean": m, "ci": [lo, hi], "p_perm_vs_tierB": round(p, 4),
                      "effect_size_sdB": es, "loo_max": round(loo, 3), "n": len(vals)}
            if pre == meta["pre"][-1]:
                pperm_store.setdefault((pre, lang), {})[ev] = p
        report["events"][ev]["per_lang"][lang] = d
    # pooled across languages: cluster bootstrap over item indices (each index = 3 correlated langs)
    pooled = {}
    for pre in meta["pre"]:
        percl = {lang: item_dconc(ev, lang, pre, OFFSET.get(lang)) for lang in LANGS}
        n = len(percl["en"])
        flat = [v for lang in LANGS for v in percl[lang]]
        m = st.mean(flat)
        boots = []
        for _ in range(B):
            idx = [random.randrange(n) for _ in range(n)]
            boots.append(st.mean([percl[lang][i] for i in idx for lang in LANGS]))
        boots.sort()
        pooled[pre] = {"mean": round(m, 3), "ci": [round(boots[int(.025*B)], 3), round(boots[int(.975*B)], 3)],
                       "n_clusters": n}
    report["events"][ev]["pooled"] = pooled

# ---------- tiers on the two estimator scales ----------
for lang in LANGS:
    report["tiers"][lang] = {
        "tierB_vs1929_1913": mean_ci(tier_vals("tierB", lang, "1913", "1929")),
        "tierC_vs1929_1913": mean_ci(tier_vals("tierC", lang, "1913", "1929")),
        "tierB_bridged_1929": mean_ci(tier_vals("tierB", lang, "1929", "modern")),
        "tierC_bridged_1929": mean_ci(tier_vals("tierC", lang, "1929", "modern")),
        "tierC_bridged_1913": mean_ci(tier_vals("tierC", lang, "1913", "modern")),
    }

# ---------- BH-FDR across events within (headline pre, lang) ----------
for (pre, lang), pv in pperm_store.items():
    report["fdr"][f"{pre}/{lang}"] = bh(pv)

# ---------- true/false discrimination ----------
for ev in EVENTS_META:
    report["tf"][ev] = {}
    for lang in LANGS:
        d = {}
        for tag in ["1900", "1913", "1929", "modern"]:
            pairs = [S[tag]["tf"][f"{ev}#{i}/{lang}"] for i in range(len(TRUEFALSE[ev]))]
            margins = [ct - cf for ct, cf in pairs]
            d[tag] = {"pct_true": round(100 * st.mean([m > 0 for m in margins])),
                      "mean_margin": round(st.mean(margins), 2)}
        report["tf"][ev][lang] = d
    # pooled across langs per model
    report["tf"][ev]["pooled"] = {}
    for tag in ["1900", "1913", "1929", "modern"]:
        margins = [S[tag]["tf"][f"{ev}#{i}/{lang}"][0] - S[tag]["tf"][f"{ev}#{i}/{lang}"][1]
                   for lang in LANGS for i in range(len(TRUEFALSE[ev]))]
        report["tf"][ev]["pooled"][tag] = {"pct_true": round(100 * st.mean([m > 0 for m in margins])),
                                           "mean_margin": round(st.mean(margins), 2)}

# ---------- headline ranking (nearest pre model, pooled) ----------
rank = []
for ev, meta in EVENTS_META.items():
    pre = meta["pre"][-1]
    pl = report["events"][ev]["pooled"][pre]
    rank.append({"event": ev, "label": meta["label"], "pre": pre, "anchor": meta["anchor"],
                 "dconc_pooled": pl["mean"], "ci": pl["ci"]})
rank.sort(key=lambda r: r["dconc_pooled"])
report["ranking"] = rank

json.dump(report, open(os.path.join(common.RESULTS, "stats_panel.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# ---------- console ----------
print("=" * 100)
print("BRIDGE OFFSET (strain_1929 - strain_modern on in-family items):",
      {lg: report['offset_bridge'][lg]['mean'] for lg in LANGS})
print("\nFORESEEABILITY RANKING — pooled Dconc at the nearest pre-model (low = foreseeable):")
print(f"{'event':22s}{'pre':>6s}{'anchor':>8s}{'Dconc':>8s}   95% CI")
for r in rank:
    print(f"{r['event']:22s}{r['pre']:>6s}{r['anchor']:>8s}{r['dconc_pooled']:>+8.2f}   [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]")
print(f"\n{'':22s}Tier-B floor (1913 vs 1929, EN/DE/FR): " +
      "/".join(f"{report['tiers'][lg]['tierB_vs1929_1913'][0]:+.2f}" for lg in LANGS))
print(f"{'':22s}Tier-C ceiling (bridged, 1913, EN/DE/FR): " +
      "/".join(f"{report['tiers'][lg]['tierC_bridged_1913'][0]:+.2f}" for lg in LANGS))
print("\nTRUE-vs-FALSE (pooled % preferring the true catastrophe | mean margin in nats):")
print(f"{'event':22s}" + "".join(f"{t:>16s}" for t in ['1900', '1913', '1929', 'modern']))
for ev in EVENTS_META:
    row = report["tf"][ev]["pooled"]
    print(f"{ev:22s}" + "".join(f"{row[t]['pct_true']:>7d}% {row[t]['mean_margin']:>+6.2f} " for t in ['1900', '1913', '1929', 'modern']))
print("\nwrote results/stats_panel.json")
