# -*- coding: utf-8 -*-
"""STATISTICS & horizon differencing. Consumes scores_by_model.json (from score_all.py) and computes the
estimands frozen in PREREGISTRATION_FORESIGHT.md:

  (1) Ranke-Ranke horizon differencing (CLEAN, shared tokenizer):
        ΔS_pre(i)   = S_cond[pre](i) - S_cond[1929](i)          excess surprise vs the 1929 hindsight anchor
        Δlex, Δconc = the S_marg and strain parts of ΔS (identity ΔS = Δlex + Δconc)
      Primary for WWI (1913 & 1900 vs 1929). For Tier-C (future to ALL incl 1929) ΔS≈0 -> the NULL anchor.

  (2) Excess conceptual strain vs the MODERN hindsight-saturated anchor (works for ALL events incl. Depression,
      cross-tokenizer via within-model strain):
        XS(M,i) = strain[M](i) - strain[modern](i)
      High XS = the time-locked worldview M finds outcome i far more conceptually strained than a modern model
      that has fully learned it. WWI: high for 1900/1913, DROPS at 1929. Depression/Weimar: high for all Ranke.

  (3) Forced-choice accuracy (calibrated / matched-twin / raw), entropy, confident-wrongness per battery/model.
  (4) Bootstrap CIs (B=10k) on every mean; SD-unit effect sizes g vs the Tier-B null; cross-event contrasts
      (WWI vs Depression, the idea's 'several SD' claim) with event-cluster bootstrap; BH-FDR.

Writes stats_report.json + a console summary. Pure numpy/py; no GPU. Run: python stats.py
"""
import os, json, math, statistics as st
import numpy as np
import common

RNG = np.random.default_rng(20260702)
SB = json.load(open(os.path.join(common.RESULTS, "scores_by_model.json"), encoding="utf-8"))
BANK = json.load(open(os.path.join(common.RESULTS, "question_bank_foresight.json"), encoding="utf-8"))
ITEMS = BANK["items"]
S = SB["scores"]                      # tag -> id -> metrics
META = {it["id"]: it for it in ITEMS}
TAGS = ["1900", "1913", "1929", "modern"]


def boot_ci(vals, B=10000, stat=np.mean):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = RNG.integers(0, len(v), size=(B, len(v)))
    bs = stat(v[idx], axis=1)
    return float(stat(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def ids_in(battery=None, facets=None, valence=None):
    out = []
    for it in ITEMS:
        if battery and it["battery"] != battery:
            continue
        if facets and it["facet"] not in facets:
            continue
        if valence and it["true_valence"] != valence:
            continue
        out.append(it["id"])
    return out


def col(tag, ids, key):
    return [S[tag][i][key] for i in ids]


report = {"n_items": len(ITEMS), "batteries": {}}

# ---------- (2) excess conceptual strain vs modern, per item ----------
XS = {tag: {i: S[tag][i]["strain"] - S["modern"][i]["strain"] for i in META} for tag in ["1900", "1913", "1929"]}
# ---------- (1) Ranke-Ranke horizon ΔS vs 1929 ----------
dS = {tag: {i: S[tag][i]["S_cond"] - S["1929"][i]["S_cond"] for i in META} for tag in ["1900", "1913"]}
dlex = {tag: {i: S[tag][i]["S_marg"] - S["1929"][i]["S_marg"] for i in META} for tag in ["1900", "1913"]}
dconc = {tag: {i: S[tag][i]["strain"] - S["1929"][i]["strain"] for i in META} for tag in ["1900", "1913"]}

# Tier-B null distribution of XS (for SD-unit effect sizes) and of ΔS
ctrlB = ids_in("ctrlB")
ctrlC = ids_in("ctrlC")


def summarize(ids, label):
    d = {"n": len(ids)}
    for tag in TAGS:
        m, lo, hi = boot_ci(col(tag, ids, "S_cond"))
        d[f"Scond_{tag}"] = round(m, 3)
    for tag in ["1900", "1913", "1929"]:
        m, lo, hi = boot_ci([XS[tag][i] for i in ids])
        d[f"XSvsModern_{tag}"] = [round(m, 3), round(lo, 3), round(hi, 3)]
    # forced choice + entropy (per Ranke model that PROBES this as a forecast handled by caller)
    for tag in ["1900", "1913", "1929"]:
        d[f"acc_cal_{tag}"] = round(100 * st.mean(col(tag, ids, "correct_cal")), 1)
        d[f"acc_matched_{tag}"] = round(100 * st.mean(col(tag, ids, "correct_matched")), 1)
        d[f"H_{tag}"] = round(st.mean(col(tag, ids, "H")), 3)
    return d


for bat in ["wwi", "dep", "weim", "ctrlB", "ctrlC"]:
    ids = ids_in(bat)
    report["batteries"][bat] = summarize(ids, bat)

# ---------- WWI horizon differencing (the clean signal) ----------
wwi = ids_in("wwi")
wwi_char = ids_in("wwi", facets=["duration", "nature", "casualties", "consequence"])
report["wwi_horizon"] = {}
for tag in ["1900", "1913"]:
    mS, loS, hiS = boot_ci([dS[tag][i] for i in wwi])
    mL, _, _ = boot_ci([dlex[tag][i] for i in wwi])
    mC, loC, hiC = boot_ci([dconc[tag][i] for i in wwi])
    report["wwi_horizon"][f"dS_{tag}_vs1929"] = [round(mS, 3), round(loS, 3), round(hiS, 3)]
    report["wwi_horizon"][f"dlex_{tag}"] = round(mL, 3)
    report["wwi_horizon"][f"dconc_{tag}_vs1929"] = [round(mC, 3), round(loC, 3), round(hiC, 3)]
# Tier-C null: ΔS should be ~0 (future to all incl 1929)
for tag in ["1900", "1913"]:
    mS, loS, hiS = boot_ci([dS[tag][i] for i in ctrlC])
    report["wwi_horizon"][f"NULL_dS_{tag}_tierC"] = [round(mS, 3), round(loS, 3), round(hiS, 3)]

# ---------- Depression / Weimar: gradient + XS vs modern (no Ranke anchor) ----------
def xs_summary(ids):
    o = {}
    for tag in ["1900", "1913", "1929"]:
        m, lo, hi = boot_ci([XS[tag][i] for i in ids])
        o[tag] = [round(m, 3), round(lo, 3), round(hi, 3)]
    return o


report["dep_XSvsModern"] = xs_summary(ids_in("dep"))
report["weim_XSvsModern"] = xs_summary(ids_in("weim"))
report["ctrlB_XSvsModern"] = xs_summary(ctrlB)
report["ctrlC_XSvsModern"] = xs_summary(ctrlC)

# ---------- SD-unit effect sizes g = (mean XS_event - mean XS_ctrlB)/SD_ctrlB, per model ----------
def g_effect(event_ids, tag):
    ev = np.array([XS[tag][i] for i in event_ids], float)
    nul = np.array([XS[tag][i] for i in ctrlB], float)
    sd = nul.std(ddof=1) if len(nul) > 1 else float("nan")
    g = (ev.mean() - nul.mean()) / sd if sd else float("nan")
    # bootstrap g
    gs = []
    for _ in range(10000):
        e = ev[RNG.integers(0, len(ev), len(ev))]
        n = nul[RNG.integers(0, len(nul), len(nul))]
        s = n.std(ddof=1)
        gs.append((e.mean() - n.mean()) / s if s else np.nan)
    gs = np.array(gs)
    return round(g, 3), round(np.nanpercentile(gs, 2.5), 3), round(np.nanpercentile(gs, 97.5), 3)


report["effect_sizes_g_vs_ctrlB"] = {}
for bat in ["wwi", "dep", "weim", "ctrlC"]:
    ids = ids_in(bat)
    # attribute each event to its most-forward model: WWI->1913, DEP->1929, WEIM->1929, ctrlC->1929
    tag = {"wwi": "1913", "dep": "1929", "weim": "1929", "ctrlC": "1929"}[bat]
    report["effect_sizes_g_vs_ctrlB"][bat] = {"model": tag, "g": g_effect(ids, tag)}

# ---------- Cross-event contrast: the idea's 'several SD' claim (WWI vs Depression) ----------
# Use XS at each event's forward model. Both measured vs the same modern floor and same ctrlB null.
def xs_at(bat, tag):
    return np.array([XS[tag][i] for i in ids_in(bat)], float)


wwi_xs_1913 = xs_at("wwi", "1913")
dep_xs_1929 = xs_at("dep", "1929")
# horizon-learned WWI: XS at 1929 (should be near null -> WWI is 'learnable/known by 1929')
wwi_xs_1929 = xs_at("wwi", "1929")
diff = dep_xs_1929.mean() - wwi_xs_1929.mean()   # how much MORE surprised is the eve-of-Depression 1929 model
# bootstrap the contrast (event-cluster: resample items within each battery)
ds = []
for _ in range(10000):
    a = dep_xs_1929[RNG.integers(0, len(dep_xs_1929), len(dep_xs_1929))].mean()
    b = wwi_xs_1929[RNG.integers(0, len(wwi_xs_1929), len(wwi_xs_1929))].mean()
    ds.append(a - b)
ds = np.array(ds)
report["contrast_dep_vs_wwi_at_1929"] = {
    "meaning": "excess conceptual strain vs modern, at the 1929 model: Depression (unlearned) minus WWI (in-sample/learned)",
    "wwi_XS_1929": [round(wwi_xs_1929.mean(), 3)] + [round(x, 3) for x in boot_ci(list(wwi_xs_1929))[1:]],
    "dep_XS_1929": [round(dep_xs_1929.mean(), 3)] + [round(x, 3) for x in boot_ci(list(dep_xs_1929))[1:]],
    "difference": round(diff, 3), "ci95": [round(np.percentile(ds, 2.5), 3), round(np.percentile(ds, 97.5), 3)],
    "p_two_sided": round(2 * min((ds <= 0).mean(), (ds >= 0).mean()), 4),
}

# ---------- Confident wrongness (the 'home by Christmas' illusion) ----------
# For WWI duration/nature items under the 1913 model: p_true is P(true outcome); wrong = 1 - p_true.
# CW-score = (1 - H/log2)*(p_wrong - p_true), high when confidently on the consensus-wrong side. H over 2 outcomes.
Hmed_ctrlB = st.median([S["1913"][i]["H"] for i in ctrlB])
cw = []
for i in ids_in("wwi", facets=["duration", "nature"]):
    p_true = S["1913"][i]["p_true"]; p_wrong = 1 - p_true; H = S["1913"][i]["H"]
    conf = 1 - H / math.log(2)
    cw.append({"id": i, "p_true_1913": round(p_true, 3), "H_1913": round(H, 3),
               "confident_wrong": bool(p_wrong > p_true and H <= Hmed_ctrlB),
               "cw_score": round(conf * (p_wrong - p_true), 3)})
report["confident_wrongness_wwi_1913"] = {"H_median_ctrlB": round(Hmed_ctrlB, 3),
    "n_confident_wrong": sum(x["confident_wrong"] for x in cw), "n": len(cw), "items": cw,
    "mean_cw_score": round(st.mean(x["cw_score"] for x in cw), 3)}

# ---------- Occurrence vs character decomposition (WWI) ----------
# 'occurrence-ish' consequence items (a great war/revolution comes) vs 'character' duration+nature+casualties.
report["wwi_occurrence_vs_character"] = {
    "character_dS_1913": [round(x, 3) for x in boot_ci([dS["1913"][i] for i in ids_in("wwi", facets=["duration", "nature", "casualties"])])],
    "consequence_dS_1913": [round(x, 3) for x in boot_ci([dS["1913"][i] for i in ids_in("wwi", facets=["consequence"])])],
}

json.dump(report, open(os.path.join(common.RESULTS, "stats_report.json"), "w"), indent=1)

# ---------- console ----------
def fmt(x):
    return f"{x[0]:+.2f} [{x[1]:+.2f},{x[2]:+.2f}]" if isinstance(x, (list, tuple)) else f"{x:+.2f}"


print("=" * 90)
print("FORESIGHT — STATS SUMMARY")
print("=" * 90)
print("\n(1) WWI horizon differencing  ΔS = S_pre - S_1929  (excess surprise vs in-sample hindsight anchor)")
for tag in ["1900", "1913"]:
    print(f"   ΔS[{tag}] on WWI      = {fmt(report['wwi_horizon'][f'dS_{tag}_vs1929'])}   "
          f"(Δconc {fmt(report['wwi_horizon'][f'dconc_{tag}_vs1929'])}, Δlex {report['wwi_horizon'][f'dlex_{tag}']:+.2f})")
    print(f"   NULL ΔS[{tag}] Tier-C = {fmt(report['wwi_horizon'][f'NULL_dS_{tag}_tierC'])}   (future to all -> expect ~0)")
print("\n(2) Excess conceptual strain vs MODERN anchor  XS = strain_M - strain_modern")
for bat in ["wwi", "dep", "weim", "ctrlB", "ctrlC"]:
    row = report["batteries"][bat]
    print(f"   {bat:6s}  1900 {fmt(row['XSvsModern_1900'])}   1913 {fmt(row['XSvsModern_1913'])}   1929 {fmt(row['XSvsModern_1929'])}")
print("\n(3) Effect size g (XS vs Tier-B null, SD units) at each event's forward model")
for bat, o in report["effect_sizes_g_vs_ctrlB"].items():
    print(f"   {bat:6s} (model {o['model']}): g = {fmt(o['g'])}")
c = report["contrast_dep_vs_wwi_at_1929"]
print(f"\n(4) At the 1929 model: Depression XS {fmt(c['dep_XS_1929'])} vs WWI XS {fmt(c['wwi_XS_1929'])}")
print(f"    difference (Dep - WWI) = {c['difference']:+.2f}  CI{c['ci95']}  p={c['p_two_sided']}")
cw = report["confident_wrongness_wwi_1913"]
print(f"\n(5) Confident-wrongness (1913 on WWI duration/nature): {cw['n_confident_wrong']}/{cw['n']} items "
      f"confidently WRONG; mean CW-score {cw['mean_cw_score']:+.2f}")
print("\n(6) Forced-choice accuracy (calibrated) by battery/model:")
for bat in ["wwi", "dep", "weim", "ctrlB", "ctrlC"]:
    r = report["batteries"][bat]
    print(f"   {bat:6s}  1900 {r['acc_cal_1900']:4.0f}%  1913 {r['acc_cal_1913']:4.0f}%  1929 {r['acc_cal_1929']:4.0f}%")
print("\nwrote stats_report.json")
