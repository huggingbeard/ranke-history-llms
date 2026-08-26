# -*- coding: utf-8 -*-
"""Emit paper/numbers.tex : one \newcommand per reported quantity, read from the result JSONs, so no number in
the paper is hand-typed. LaTeX command names must be letters only -> cutoffs are spelled (nineHundred /
nineThirteen / nineTwentyNine). Missing inputs -> \TBD. Run last: python make_numbers.py
"""
import os, json
import common

HERE = common.RESULTS
PAPER = common.PAPER
os.makedirs(PAPER, exist_ok=True)


def jload(fn):
    p = os.path.join(HERE, fn)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


stats = jload("stats_report.json") or {}
gate = jload("gate_report.json") or {}
textp = jload("text_perplexity.json")
prod = jload("production_report.json")
bank = jload("question_bank_foresight.json") or {}
out = {}


def ci(x):  # BARE (no outer $); the paper wraps each macro in $...$ itself
    if not isinstance(x, (list, tuple)) or len(x) < 3 or x[0] is None:
        return r"\TBD"
    return f"{x[0]:+.2f}\\,[{x[1]:+.2f},\\,{x[2]:+.2f}]"


def num(x, dp=2, signed=True):
    try:
        return f"{x:+.{dp}f}" if signed else f"{x:.{dp}f}"
    except Exception:
        return r"\TBD"


def raw(x):
    return x if x is not None else r"\TBD"


bats = stats.get("batteries", {})
wh = stats.get("wwi_horizon", {})
con = stats.get("contrast_dep_vs_wwi_at_1929", {})
cw = stats.get("confident_wrongness_wwi_1913", {})

# --- bank sizes (keys WITHOUT leading N; the emit loop prepends \N) ---
out["items"] = raw(bank.get("n_items"))
out["matchedfrac"] = f"{bank.get('n_length_matched','?')}/{bank.get('n_items','?')}"
out["gate"] = len(bank.get("gate", [])) if bank else r"\TBD"
bm = bank.get("battery_meta", {})
out["batwwi"] = raw(bm.get("wwi", {}).get("n"))
out["batdep"] = raw(bm.get("dep", {}).get("n"))
out["batweim"] = raw(bm.get("weim", {}).get("n"))

# --- WWI horizon differencing ---
out["wwidSTh"] = ci(wh.get("dS_1913_vs1929"))
out["wwidSNn"] = ci(wh.get("dS_1900_vs1929"))
out["wwiNullTierCTh"] = ci(wh.get("NULL_dS_1913_tierC"))
out["wwiDconcTh"] = ci(wh.get("dconc_1913_vs1929"))
out["wwiDlexTh"] = num(wh.get("dlex_1913"))

# --- XS vs modern ---
def xs(bat, cut):
    return ci(bats.get(bat, {}).get(f"XSvsModern_{cut}"))


out["XSwwinineTwentyNine"] = xs("wwi", "1929")
out["XSctrlBnineTwentyNine"] = xs("ctrlB", "1929")
out["XSctrlCnineTwentyNine"] = xs("ctrlC", "1929")
out["XSdepnineHundred"] = xs("dep", "1900")
out["XSdepnineThirteen"] = xs("dep", "1913")
out["XSdepnineTwentyNine"] = xs("dep", "1929")

# --- contrast (headline) ---
out["contrastDiff"] = num(con.get("difference"))
c95 = con.get("ci95", [None, None])
out["contrastCI"] = f"[{c95[0]:+.2f},\\,{c95[1]:+.2f}]" if c95 and c95[0] is not None else r"\TBD"
out["contrastP"] = num(con.get("p_two_sided"), 4, signed=False)

# --- confident wrongness ---
out["cwN"] = raw(cw.get("n_confident_wrong"))
out["cwTot"] = raw(cw.get("n"))
out["cwScore"] = num(cw.get("mean_cw_score"))

# --- gate ---
def gv(ev):
    g = gate.get(ev, {})
    v = g.get("verdict", "")
    return "GREEN" if "GREEN" in v else ("AMBER" if "AMBER" in v else ("RED" if "RED" in v else r"\TBD"))


gw, gd = gate.get("wwi", {}), gate.get("dep", {})
out["gatewwiPct"] = num(gw.get("post_pct"), 0, signed=False)
out["gatewwi"] = f"{gw.get('post_correct','?')}/{gw.get('post_n','?')}"
out["gatewwiVerdict"] = gv("wwi")
out["gatedepPct"] = num(gd.get("post_pct"), 0, signed=False)
out["gatedepVerdict"] = gv("dep")
bp = gd.get("boundary_probes", [])
out["boundaryKnown"] = f"{sum(r['correct'] for r in bp)}/{len(bp)}" if bp else r"\TBD"

# --- Arm B text (modern-anchored per-char DiD, bootstrapped) ---
if textp:
    did = textp.get("DiD_modern_anchored", {})
    def _didci(key):
        d = did.get(key)
        if not d:
            return r"\TBD", r"\TBD"
        c = d.get("ci", [None, None])
        return num(d.get("did")), (f"[{c[0]:+.3f},\\,{c[1]:+.3f}]" if c and c[0] is not None else r"\TBD")
    out["textWWIdidTh"], out["textWWIciTh"] = _didci("wwi_at_1913")
    out["textWWIdidNn"], out["textWWIciNn"] = _didci("wwi_at_1900")
    out["textDepDiDNn"], out["textDepCiNn"] = _didci("dep_at_1900")
    out["textDepDiDTh"], out["textDepCiTh"] = _didci("dep_at_1913")
    out["textDepDiDTN"], out["textDepCiTN"] = _didci("dep_at_1929")

# --- recognition vs production ---
if prod:
    try:
        from foresight_bank import PRODUCTION
        sb = json.load(open(os.path.join(HERE, "scores_by_model.json"), encoding="utf-8"))["scores"]
        keymap = {"1900": "Nn", "1913": "Th", "1929": "TN"}
        for t, sfx in keymap.items():
            rc = [sb[t][it["id"]]["correct_cal"] for it in PRODUCTION]
            pr = [prod["items"][it["id"]]["coded"].get(t, {}).get("frac_true", 0.0) for it in PRODUCTION]
            out[f"recog{sfx}"] = f"{round(100*sum(rc)/len(rc))}"
            out[f"prod{sfx}"] = num(sum(pr)/len(pr), 2, signed=False)
        out["prodDurTh"] = num(prod["items"]["wwi_dur_years"]["coded"].get("1913", {}).get("frac_true", 0), 2, signed=False)
    except Exception as e:
        print("production macro warn:", e)

# --- cross-lingual THREE-event analysis (WWI, empires, Depression x EN/DE/FR) ---
xle = jload("crosslingual_events.json")
LG = {"en": "EN", "de": "DE", "fr": "FR"}
if xle:
    dp = xle.get("did_pooled", {})
    def _poolci(contrast, pre):
        d = dp.get(contrast, {}).get(pre)
        return ci(d) if d else r"\TBD"
    out["xlDepWWIpoolTN"] = _poolci("great_depression__minus__world_war", "1929")
    out["xlDepWWIpoolTh"] = _poolci("great_depression__minus__world_war", "1913")
    out["xlDepEmppoolTN"] = _poolci("great_depression__minus__fall_of_empires", "1929")
    out["xlDepEmppoolTh"] = _poolci("great_depression__minus__fall_of_empires", "1913")
    cs = xle.get("conceptual_surprise", {})
    for lg, sfx in LG.items():
        out[f"xlWWIconc{sfx}"] = num(cs.get("world_war", {}).get(lg, {}).get("1913"))
        out[f"xlEmpconc{sfx}"] = num(cs.get("fall_of_empires", {}).get(lg, {}).get("1913"))
        out[f"xlDepXS{sfx}"] = num(cs.get("great_depression", {}).get(lg, {}).get("1929"))
    nrows = len(xle.get("rows", []))
    out["xlNitems"] = nrows // 9 if nrows else r"\TBD"   # 3 events x 3 langs = 9 cells per item index

# --- Zweig worldview (his four themes: progress, war, women, youth) ---
zw = jload("zweig_probe.json")
if zw:
    sm = zw.get("summary", {})
    for lg, sfx in LG.items():
        out[f"zwProg{sfx}"] = num(sm.get("1913", {}).get(lg, {}).get("progress"))
        out[f"zwProgMod{sfx}"] = num(sm.get("modern", {}).get(lg, {}).get("progress"))
        out[f"zwWar{sfx}"] = num(sm.get("1913", {}).get(lg, {}).get("war"))
        out[f"zwWarMod{sfx}"] = num(sm.get("modern", {}).get(lg, {}).get("war"))
        out[f"zwWomen{sfx}"] = num(sm.get("1913", {}).get(lg, {}).get("women"))
        out[f"zwWomenMod{sfx}"] = num(sm.get("modern", {}).get(lg, {}).get("women"))
        out[f"zwYouth{sfx}"] = num(sm.get("1913", {}).get(lg, {}).get("youth"))

# ================= PANEL v2 (eight events x three languages) =================
panel = jload("stats_panel.json")
gatep = jload("gate_panel.json")
zw3 = jload("zweig_probe_v2.json")
EVKEY = {"panic_1907": "panic", "world_war": "wwi", "fall_of_empires": "emp", "russian_revolution": "rev",
         "spanish_flu": "flu", "hyperinflation": "infl", "great_depression": "dep", "nazi_seizure": "nazi"}
if panel:
    def mci(d):  # {"mean":m,"ci":[lo,hi]} -> "m\,[lo,\,hi]"
        return f"{d['mean']:+.2f}\\,[{d['ci'][0]:+.2f},\\,{d['ci'][1]:+.2f}]"
    for ev, short in EVKEY.items():
        e = panel["events"][ev]
        pre = [p for p in ["1900", "1913", "1929"] if p in e["pooled"]][-1]
        out[f"pan{short}"] = mci(e["pooled"][pre])
        out[f"pan{short}pre"] = pre
        for lg, sfx in LG.items():
            d = e["per_lang"][lg][pre]
            out[f"pan{short}{sfx}"] = num(d["mean"])
            out[f"pan{short}P{sfx}"] = num(d["p_perm_vs_tierB"], 3, signed=False)
    for lg, sfx in LG.items():
        out[f"panoffset{sfx}"] = num(panel["offset_bridge"][lg]["mean"])
        out[f"panfloorB{sfx}"] = num(panel["tiers"][lg]["tierB_vs1929_1913"][0])
        out[f"panceilC{sfx}"] = num(panel["tiers"][lg]["tierC_bridged_1913"][0])
    # true/false pooled, pre + modern
    for ev, short in EVKEY.items():
        pre = out[f"pan{EVKEY[ev]}pre"]
        tfp = panel["tf"][ev]["pooled"]
        out[f"tf{short}pre"] = f"{tfp[pre]['pct_true']}"
        out[f"tf{short}mod"] = f"{tfp['modern']['pct_true']}"
        out[f"tf{short}margin"] = num(tfp[pre]["mean_margin"])

    # ---------- auto-generated tables ----------
    LABELS = {ev: panel["events"][ev]["label"] for ev in EVKEY}
    # ranking table
    T = [r"\begin{tabular}{lccrr}", r"\toprule",
         r"Event & Pre mind & Anchor & $\Delta_{\text{conc}}$ (pooled) & 95\% CI \\", r"\midrule"]
    for r_ in panel["ranking"]:
        anc = "1929 (in-family)" if r_["anchor"] == "1929" else ("1913 (in-family)" if r_["anchor"] == "1913" else "modern (bridged)")
        T.append(f"{LABELS[r_['event']]} & {r_['pre']} & {anc} & ${r_['dconc_pooled']:+.2f}$ & "
                 f"$[{r_['ci'][0]:+.2f},\\,{r_['ci'][1]:+.2f}]$ \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, "ranking_table.tex"), "w", encoding="utf-8").write("\n".join(T))
    # per-language table (Dconc at nearest pre)
    T = [r"\begin{tabular}{lrrr}", r"\toprule", r"Event & English & German & French \\", r"\midrule"]
    for r_ in panel["ranking"]:
        ev = r_["event"]
        pre = r_["pre"]
        cells = " & ".join(f"${panel['events'][ev]['per_lang'][lg][pre]['mean']:+.2f}$" for lg in ["en", "de", "fr"])
        T.append(f"{LABELS[ev]} & {cells} \\\\")
    T += [r"\midrule",
          "Timeless-truth floor & " + " & ".join(f"${panel['tiers'][lg]['tierB_vs1929_1913'][0]:+.2f}$" for lg in ["en", "de", "fr"]) + r" \\",
          "Moon-landing ceiling & " + " & ".join(f"${panel['tiers'][lg]['tierC_bridged_1913'][0]:+.2f}$" for lg in ["en", "de", "fr"]) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, "bylang_table.tex"), "w", encoding="utf-8").write("\n".join(T))
    # gate table
    if gatep:
        T = [r"\begin{tabular}{lcllll}", r"\toprule",
             r"Event & Pre mind & Gate (pre) & correct & Ceiling & correct \\", r"\midrule"]
        for ev, short in EVKEY.items():
            vp = gatep["verdicts"][f"{ev}/pre"]["all"]
            vc = gatep["verdicts"][f"{ev}/ceiling"]["all"]
            pre = out[f"pan{short}pre"]
            extra = gatep["verdicts"][f"{ev}/pre"].get("posthoc_excluding")
            v = vp["verdict"] + (r"$^{\dagger}$" if extra else "")
            T.append(f"{LABELS[ev]} & {pre} & {v} & {vp['k']}/{vp['n']} & {vc['verdict']} & {vc['k']}/{vc['n']} \\\\")
        T += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(PAPER, "gates_table.tex"), "w", encoding="utf-8").write("\n".join(T))
        ph = gatep["verdicts"]["russian_revolution/pre"].get("posthoc_excluding", {})
        out["gaterevposthoc"] = f"{ph.get('k','?')}/{ph.get('n','?')}"
        out["gaterevposthocP"] = num(ph.get("p_binom"), 3, signed=False)
    # true/false table
    T = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Event & mind of 1900 & 1913 & 1929 & modern \\", r"\midrule"]
    for r_ in panel["ranking"]:
        ev = r_["event"]
        row = panel["tf"][ev]["pooled"]
        T.append(f"{LABELS[ev]} & " + " & ".join(f"{row[t]['pct_true']}\\%" for t in ["1900", "1913", "1929", "modern"]) + r" \\")
    T += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, "tf_table.tex"), "w", encoding="utf-8").write("\n".join(T))

# ---- per-pair T/F margins + production-panel + tripwire numbers used in the prose ----
psc = jload("panel_scores.json")
prodp = jload("production_panel.json")
if psc:
    import statistics as _st
    def _tfm(ev, i, tag):
        return _st.mean(psc[tag]["tf"][f"{ev}#{i}/{lg}"][0] - psc[tag]["tf"][f"{ev}#{i}/{lg}"][1]
                        for lg in ["en", "de", "fr"])
    out["tfFluFormPre"] = num(_tfm("spanish_flu", 0, "1913"))
    out["tfFluYoungPre"] = num(_tfm("spanish_flu", 1, "1913"))
    out["tfFluSerumPre"] = num(_tfm("spanish_flu", 4, "1913"))
    out["tfFluWaryearPre"] = num(_tfm("spanish_flu", 3, "1913"))
    out["tfNaziLawful"] = num(_tfm("nazi_seizure", 0, "1929"))
    out["tfNaziCorporal"] = num(_tfm("nazi_seizure", 1, "1929"))
    out["tfNaziPresident"] = num(_tfm("nazi_seizure", 4, "1929"))
    out["tfNaziCheering"] = num(_tfm("nazi_seizure", 5, "1929"))
    out["tfPanicTrusts"] = num(_tfm("panic_1907", 0, "1900"))
    out["tfPanicSevere"] = num(_tfm("panic_1907", 5, "1900"))
    def _xsraw(ev_or_tier, lg, tag, n):
        return _st.mean(psc[tag]["strain"][f"{ev_or_tier}#{i}/{lg}"][2] - psc["modern"]["strain"][f"{ev_or_tier}#{i}/{lg}"][2]
                        for i in range(n))
    for lg, sfx in LG.items():
        out[f"tripCraw{sfx}"] = num(_xsraw("tierC", lg, "1913", 5))
    # WWI raw XS at the 1929 mind (which READ the war) — the offset demonstration
    for lg, sfx in LG.items():
        out[f"wwiXSatTN{sfx}"] = num(_xsraw("world_war", lg, "1929", 12))
if prodp:
    def _pfrac(pid, tag):
        return num(prodp["items"][pid]["coded"].get(tag, {}).get("frac_true"), 2, signed=False)
    out["prodRevTsarTh"] = _pfrac("rev_tsar", "1913")
    out["prodFluFormTh"] = _pfrac("flu_form", "1913")
    out["prodInflMarkTh"] = _pfrac("infl_mark", "1913")
    out["prodNaziWhoTN"] = _pfrac("nazi_who", "1929")
    out["prodPanicBanksNn"] = _pfrac("panic_banks", "1900")

# ---- v3 additions: Depression adjudication (decomposition + matched-content frame check) + odds ----
if psc and stats:
    import math as _m
    import random as _rd
    _rd.seed(20260704)
    sb_ = jload("scores_by_model.json")["scores"]
    bank_ = jload("question_bank_foresight.json")
    dep_ids = [it["id"] for it in bank_["items"] if it.get("battery") == "dep"]
    V1_SEV = {"dep_dur_decade", "dep_sev_fourth", "dep_sev_millions", "dep_bank_fail", "dep_gold_off", "dep_opt_boom"}
    xs_ = {i: sb_["1929"][i]["strain"] - sb_["modern"][i]["strain"] for i in dep_ids}
    sev_ = [xs_[i] for i in dep_ids if i in V1_SEV]
    gen_ = [xs_[i] for i in dep_ids if i not in V1_SEV]
    import statistics as _st
    def _bd(a, b, B=10000):
        na, nb = len(a), len(b)
        ds = sorted(_st.mean([a[_rd.randrange(na)] for _ in range(na)]) - _st.mean([b[_rd.randrange(nb)] for _ in range(nb)]) for _ in range(B))
        return ds[int(.025 * B)], ds[int(.975 * B)]
    lo_, hi_ = _bd(sev_, gen_)
    out["depSevVOne"] = num(_st.mean(sev_))
    out["depGenVOne"] = num(_st.mean(gen_))
    out["depSevGenDiff"] = f"{_st.mean(sev_)-_st.mean(gen_):+.2f}\\,[{lo_:+.2f},\\,{hi_:+.2f}]"
    # matched-content frame check (same outcomes, two calibration frames), EN @1929
    OFF_EN = panel["offset_bridge"]["en"]["mean"]
    PAIRS = [("dep_opt_plateau", 0), ("dep_sev_fourth", 1), ("dep_bank_fail", 2), ("dep_dur_slump", 3),
             ("dep_pol_radical", 4), ("dep_trade_walls", 6)]
    v1m = _st.mean(sb_["1929"][v]["strain"] - sb_["modern"][v]["strain"] for v, _ in PAIRS)
    pnm = _st.mean(psc["1929"]["strain"][f"great_depression#{p}/en"][2]
                   - psc["modern"]["strain"][f"great_depression#{p}/en"][2] - OFF_EN for _, p in PAIRS)
    out["depMatchedVOne"] = num(v1m)
    out["depMatchedPanel"] = num(pnm)
    # odds translations (e^nats, rounded)
    con_ = stats.get("contrast_dep_vs_wwi_at_1929", {})
    if con_.get("difference") is not None:
        out["oddsHeadline"] = f"{_m.exp(con_['difference']):.0f}"
    out["oddsInfl"] = f"{_m.exp(panel['events']['hyperinflation']['pooled']['1913']['mean']):.1f}"
    out["oddsInflFR"] = f"{_m.exp(panel['events']['hyperinflation']['per_lang']['fr']['1913']['mean']):.1f}"
    out["oddsOffsetEN"] = f"{_m.exp(panel['offset_bridge']['en']['mean']):.1f}"
    def _tfm2(ev, i, tag):
        return _st.mean(psc[tag]["tf"][f"{ev}#{i}/{lg}"][0] - psc[tag]["tf"][f"{ev}#{i}/{lg}"][1]
                        for lg in ["en", "de", "fr"])
    out["tfPanicFinanciers"] = num(_tfm2("panic_1907", 1, "1900"))
    out["tfPanicAutumn"] = num(_tfm2("panic_1907", 2, "1900"))

# Zweig v3 (5 items/theme, women normative)
if zw3:
    sm3 = zw3.get("summary", {})
    for lg, sfx in LG.items():
        for th in ["progress", "war", "women", "youth"]:
            out[f"zwiii{th}{sfx}"] = num(sm3.get("1913", {}).get(lg, {}).get(th))
            out[f"zwiii{th}Mod{sfx}"] = num(sm3.get("modern", {}).get(lg, {}).get(th))

lines = [r"% AUTO-GENERATED by make_numbers.py -- do not edit.",
         r"\providecommand{\TBD}{\textbf{[TBD]}}"]
for k, v in out.items():
    lines.append(f"\\newcommand{{\\N{k}}}{{{v}}}")
open(os.path.join(PAPER, "numbers.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"wrote paper/numbers.tex with {len(out)} macros")
