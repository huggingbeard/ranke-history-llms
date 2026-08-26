# -*- coding: utf-8 -*-
"""Independent check of paper/numbers_v5.tex against the raw result files.

This does NOT call make_numbers_v5.py. It re-derives every headline quantity a second way, straight
from the JSONs, and compares to the macros the paper actually uses. Any mismatch prints FAIL and the
script exits non-zero. Run after make_numbers_v5.py.
Run:  python src/verify_v5.py
"""
import os, re, json, math, statistics as st
import numpy as np
import common

R, PAPER = common.RESULTS, common.PAPER
SUP = os.path.join(R, "superlatives")


def j(fn, base=R):
    return json.load(open(os.path.join(base, fn), encoding="utf-8"))


macros = {}
for m in re.finditer(r"\\newcommand\{\\N([A-Za-z]+)\}\{([^}]*)\}",
                     open(os.path.join(PAPER, "numbers_v5.tex"), encoding="utf-8").read()):
    macros[m.group(1)] = m.group(2)

checks = []


def chk(label, expected, macro_key):
    got = macros.get(macro_key, "<<missing>>")
    ok = (str(expected) == str(got))
    checks.append((label, str(expected), str(got), ok))


def pctround(x):
    return str(round(100 * x))


def spk(x):
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def pctv(x):
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


v41 = j("v41_battery.json")
er = {r["name"]: r for r in v41["rows"]}
held = j("certify_chronicle_heldout.json")
defr_cert = j("certify_defr_v2.json")
defr_first = j("certify_defr.json")
defr_bat = j("v41_defr.json")
v4 = j("v4_battery.json")

# ---------------- v4-inherited checks ----------------
cov = [r for r in held["rows"] if r["covered"]]
chk("exam answered", len(cov), "examAnswered")
chk("exam correct (n)", sum(1 for r in cov if r["correct"]), "examCorrect")
chk("exam correct (%)", round(100 * sum(1 for r in cov if r["correct"]) / len(cov)), "examCorrectPct")
chk("exam out-bet (n)", sum(1 for r in cov if r["grad"] > 0), "examOutbet")
chk("exam out-bet (%)", round(100 * sum(1 for r in cov if r["grad"] > 0) / len(cov)), "examOutbetPct")

meas = [r for r in v41["rows"] if not r["memoir_only"]]
chk("measured items", len(meas), "nMeasured")
chk("covered items", len([r for r in meas if r["covered"] and not r["excluded"]]), "nCovered")

k = er["emp_kaiser_1919"]
sp_post = -math.log(k["p_truth_post"])
sp_pre = k["gap"] + sp_post
chk("Kaiser guesser %", pctround(k["p_truth_pre"]), "kaiserPre")
chk("Kaiser witness %", pctround(k["p_truth_post"]), "kaiserPost")
chk("Kaiser gap", f"{k['gap']:+.2f}", "kaiserGap")
chk("Kaiser surp(post)", f"{sp_post:.2f}", "kaiserSurpPost")
chk("Kaiser surp(pre)", f"{sp_pre:.1f}", "kaiserSurpPre")
chk("Kaiser implied bet", f"{100*math.exp(-sp_pre):.3f}", "kaiserPreImplied")
chk("Kaiser guesser's own bet %", pctround(k["pre_bet_p"]), "kaiserPreBetP")

for key, mk in [("rev_debts", "debtsGap"), ("infl_end", "inflendGap"), ("war_us_entry", "usentryGap")]:
    chk(f"gap {key}", f"{er[key]['gap']:+.2f}", mk)

for lg, cap in [("de", "DE"), ("fr", "FR")]:
    cv = [r for r in defr_cert["rows"][lg] if r["covered"]]
    chk(f"{cap} exam answered (repaired)", len(cv), cap + "examAnswered")
    chk(f"{cap} exam correct (repaired)", sum(1 for r in cv if r["correct"]), cap + "examCorrect")
    chk(f"{cap} exam out-bet (repaired)", sum(1 for r in cv if r["grad"] > 0), cap + "examOutbet")
    cv1 = [r for r in defr_first["rows"][lg] if r["covered"]]
    chk(f"{cap} exam answered (first run)", len(cv1), cap + "firstAnswered")
    chk(f"{cap} exam out-bet (first run)", sum(1 for r in cv1 if r["grad"] > 0), cap + "firstOutbet")
# the certification verdicts themselves, re-derived
cvfr = [r for r in defr_cert["rows"]["fr"] if r["covered"]]
fr_pass = (len(cvfr) >= 14 and sum(r["correct"] for r in cvfr) >= 0.8 * len(cvfr)
           and sum(r["grad"] > 0 for r in cvfr) >= 0.8 * len(cvfr))
checks.append(("FR repaired run certifies", "True", str(fr_pass), fr_pass is True))
cvde = [r for r in defr_cert["rows"]["de"] if r["covered"]]
checks.append(("DE repaired run fails on coverage", "True", str(len(cvde) < 14), len(cvde) < 14))

chk("false alarm old %", pctround(next(r for r in v4["rows"] if r["name"] == "nazi_who_rules")["p_truth_pre"]), "faOld")
chk("false alarm new %", pctround(er["nazi_who_rules"]["p_truth_pre"]), "faNew")

for lg, cap in [("de", "DE"), ("fr", "FR")]:
    r = {x["name"]: x for x in defr_bat["rows"][lg]}["emp_kaiser_1919"]
    chk(f"Kaiser {cap} witness %", pctround(r["p_truth_post"]), "kaiser" + cap + "Post")
    chk(f"Kaiser {cap} gap", f"{r['gap']:+.1f}", "kaiser" + cap + "Gap")

chk("memoir banks %", pctround(er["dep_banks"]["pre_bet_p"]), "depBanksMemoir")
chk("memoir dictatorship %", pctround(er["nazi_republic_fate"]["pre_bet_p"]), "naziDictMemoir")
chk("memoir gold %", pctround(er["dep_gold"]["pre_bet_p"]), "depGoldMemoir")

# ---------------- v5: the discovery (base date spikes) ----------------
spike = j("date_spike_base.json", SUP)
for tag, cap, cut in [("1913", "Thirteen", 1913), ("1929", "TwentyNine", 1929)]:
    nulls = list(spike[tag]["null"].values())
    thr = float(np.percentile(nulls, 95))
    chk(f"{tag} fiction ceiling", spk(thr), "spikeNullNinetyFifth" + cap)
    chk(f"{tag} known median", spk(float(np.median(list(spike[tag]["known"].values())))), "spikeKnownMed" + cap)
    post = [(l, d) for l, d in spike[tag]["test"].items() if d["true"] > cut]
    chk(f"{tag} post-cutoff events", len(post), "postCount" + cap)
    chk(f"{tag} leaks above ceiling", sum(1 for _, d in post if d["spike"] > thr), "leakCount" + cap)
chk("1913 WWI spike", spk(spike["1913"]["test"]["First World War began"]["spike"]), "spikeWWIThirteen")
chk("1913 Bolshevik spike", spk(spike["1913"]["test"]["Bolsheviks seized power"]["spike"]), "spikeBolshThirteen")
chk("1913 Civil Rights Act spike", spk(spike["1913"]["test"]["Civil Rights Act"]["spike"]), "spikeCRAThirteen")
chk("1929 WW2-ended spike", spk(spike["1929"]["test"]["Second World War ended"]["spike"]), "spikeWWtwoEndTwentyNine")

# ---------------- v5: the triage ----------------
tri = j("contamination_triage.json", SUP)
rows_t = tri["triage"]
nonmem = [r for r in rows_t if not r["memoir_only"]]
gated = [r for r in rows_t if r["gated"]]
chk("triage items", len(nonmem), "triageItems")
chk("triage suspect (all)", sum(1 for r in nonmem if r["verdict"] == "SUSPECT"), "triageSuspect")
chk("gated findings", len(gated), "gatedTotal")
chk("gated suspect", sum(1 for r in gated if r["verdict"] == "SUSPECT"), "gatedSuspect")
chk("gated safe", sum(1 for r in gated if r["verdict"] == "SAFE"), "gatedSafe")
tb = {r["name"]: r for r in rows_t}
for nm, want in [("emp_kaiser_1919", "SAFE"), ("infl_end", "SAFE"), ("panic_rescuer", "SAFE"),
                 ("panic_severity", "SAFE"), ("war_breadth", "SUSPECT"), ("war_us_entry", "SUSPECT"),
                 ("rev_debts", "SUSPECT")]:
    checks.append((f"verdict {nm}", want, tb[nm]["verdict"], tb[nm]["verdict"] == want))
chk("1913 war date-spike %", pctv(tri["meta"]["event_date_pct"]["world_war"]), "dateWar")
chk("1913 revolution date-spike %", pctv(tri["meta"]["event_date_pct"]["revolution"]), "dateRev")
chk("1900 panic date-spike %", pctv(tri["meta"]["event_date_pct"]["panic_1907"]), "datePanic")
chk("1913 inflation date-spike %", pctv(tri["meta"]["event_date_pct"]["hyperinflation"]), "dateInfl")
chk("1913 fiction floor %", pctv(tri["meta"]["fiction_floor_pct_by_guesser"]["1913"]), "floorThirteen")

# ---------------- v5: the witness gate ----------------
gate = j("gate_report.json")
chk("gate before", gate["counts"]["findings_before_gate"], "gateBefore")
chk("gate passed", gate["counts"]["passed"], "gatePassed")
chk("gate dropped", gate["counts"]["dropped"], "gateDropped")
ts = next(r for r in gate["dropped"] if r["name"] == "rev_tsar_fate")
chk("tsar witness %", pctround(ts["witness_p_truth"]), "tsarWitnessP")

# ---------------- v5: the mundane yardstick ----------------
ctrl = j("control_gaps_v3.json")
chk("mundane 90th pct", f"{ctrl['mundane_gap_percentiles']['90']:.2f}", "mundaneNinety")
cm = {r["name"]: r for r in ctrl["catastrophes_on_mundane_scale"]}
chk("Ebert mundane pct", f"{cm['emp_kaiser_1919']['percentile_in_mundane']:.0f}", "kaiserMundanePct")
chk("Rentenmark mundane pct", f"{cm['infl_end']['percentile_in_mundane']:.0f}", "inflendMundanePct")
gated_c = [r for r in ctrl["control_rows"] if r["covered"] and r["p_witness"] >= 0.5]
chk("yardstick items", ctrl["meta"]["n_items"], "mundaneN")
chk("yardstick gated", len(gated_c), "mundaneGated")
chk("top mundane gap", f"{max(r['gap'] for r in gated_c):+.2f}", "mundaneTopGap")
chk("gated below Ebert", sum(1 for r in gated_c if r["gap"] < cm["emp_kaiser_1919"]["gap"]), "kaiserMundaneBelow")

# ---------------- v5: the Ebert name-free rebuild ----------------
nf = j("ebert_namefree_v2.json")
n20 = nf["namefree_1920"]
chk("name-free guesser on president %", pctround(n20["p_truth_pre"]), "nfTwentyPresPre")
chk("name-free guesser on Kaiser %", pctround(n20["p_pre"][1]), "nfTwentyKaiserPre")
chk("name-free office surprise", f"{-math.log(n20['p_truth_pre']):.1f}", "nfTwentySurpPre")
chk("name-free witness on councils %", pctround(nf["namefree_1919"]["p_post"][3]), "nfCouncilPost")
checks.append(("name-free 1919 not gated", "False", str(nf["namefree_1919"]["gated"]), nf["namefree_1919"]["gated"] is False))
checks.append(("name-free 1920 not gated", "False", str(n20["gated"]), n20["gated"] is False))

# ---------------- v5: the memory meter ----------------
mri = j("probe_mri_summary.json", SUP)
chk("meter AUC", f"{mri['auc']:.2f}", "meterAUC")
chk("meter on unknown", f"{mri['unknown']:.2f}", "meterUnknown")
chk("meter on leaked", f"{mri['leaked']:.2f}", "meterLeaked")

bluff = j("real_entity_null_summary.json", SUP)
chk("confident bluffs (n)", bluff["n_conf"], "bluffN")
chk("Bordeaux peak %", bluff["confident"]["Treaty of Bordeaux"], "bordeauxPeak")
chk("probe AUC on bluffs", f"{bluff['auc_conf']:.2f}", "bluffProbeAUC")
chk("output AUC on bluffs", f"{bluff['out_auc']:.2f}", "bluffOutAUC")

mp = j("minimal_pairs_summary.json", SUP)
chk("Portsmouth output %", mp["real"]["Portsmouth"]["ptrue"], "portsmouthConf")
chk("Portsmouth meter", f"{mp['real']['Portsmouth']['meter']:.2f}", "portsmouthMeter")
chk("Le Havre output %", mp["fake"]["Le Havre"]["peak"], "lehavreConf")

rome = j("rome_inject.json", SUP)
ri = {r["city"]: r for r in rome["inject"]}
chk("Björkö probe before", f"{ri['Björkö']['pre_probe']:.2f}", "bjorkoPreProbe")
chk("Björkö probe after", f"{ri['Björkö']['post_probe']:.2f}", "bjorkoPostProbe")
fakes = [c for c in ri if c not in ("Björkö", "Locarno")]
chk("ROME fakes injected", len(fakes), "romeFakesN")
chk("ROME output flips", sum(1 for c in fakes if ri[c]["post_ptrue"] > 0.5), "romeOutputFlips")
chk("ROME probe flips", sum(1 for c in fakes if ri[c]["post_probe"] > 0.5), "romeProbeFlips")

shift = j("cutoff_shift_summary.json", SUP)
chk("cutoff shift 1913", f"{shift['categories']['clean']['m1913']:.2f}", "shiftCleanThirteen")
chk("cutoff shift 1929", f"{shift['categories']['clean']['m1929']:.2f}", "shiftCleanTwentyNine")

# ---- report ----
w = max(len(c[0]) for c in checks)
fails = 0
print(f"{'quantity'.ljust(w)}  {'from data':>12}  {'macro':>12}  ok")
print("-" * (w + 34))
for label, exp, got, ok in checks:
    fails += (not ok)
    print(f"{label.ljust(w)}  {exp:>12}  {got:>12}  {'OK' if ok else 'FAIL <<<'}")
print("-" * (w + 34))
print(f"{len(checks)} checks, {fails} failures")
raise SystemExit(1 if fails else 0)
