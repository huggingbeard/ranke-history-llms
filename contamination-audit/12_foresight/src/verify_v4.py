# -*- coding: utf-8 -*-
"""Independent check of paper/numbers_v4.tex against the raw result files.

This does NOT call make_numbers_v4.py. It re-derives every headline quantity a second way, straight
from the JSONs, and compares to the macros the paper actually uses. Any mismatch prints FAIL and the
script exits non-zero. Run after make_numbers_v4.py.
Run:  python src/verify_v4.py
"""
import os, re, json, math, statistics as st
import common

R, PAPER = common.RESULTS, common.PAPER


def j(fn):
    return json.load(open(os.path.join(R, fn), encoding="utf-8"))


# parse the macros the paper uses
macros = {}
for m in re.finditer(r"\\newcommand\{\\N([A-Za-z]+)\}\{([^}]*)\}", open(os.path.join(PAPER, "numbers_v4.tex"), encoding="utf-8").read()):
    macros[m.group(1)] = m.group(2)

checks = []  # (label, expected_from_data, macro_value, ok)


def chk(label, expected, macro_key):
    got = macros.get(macro_key, "<<missing>>")
    ok = (str(expected) == str(got))
    checks.append((label, str(expected), str(got), ok))


def pctround(x):
    return str(round(100 * x))


v41 = j("v41_battery.json")
er = {r["name"]: r for r in v41["rows"]}
held = j("certify_chronicle_heldout.json")
defr_cert = j("certify_defr.json")
defr_bat = j("v41_defr.json")
v4 = j("v4_battery.json")

# 1. exam counts (recomputed independently)
cov = [r for r in held["rows"] if r["covered"]]
chk("exam answered", len(cov), "examAnswered")
chk("exam correct (n)", sum(1 for r in cov if r["correct"]), "examCorrect")
chk("exam correct (%)", round(100 * sum(1 for r in cov if r["correct"]) / len(cov)), "examCorrectPct")
chk("exam out-bet (n)", sum(1 for r in cov if r["grad"] > 0), "examOutbet")
chk("exam out-bet (%)", round(100 * sum(1 for r in cov if r["grad"] > 0) / len(cov)), "examOutbetPct")

# 2. coverage of the real battery
meas = [r for r in v41["rows"] if not r["memoir_only"]]
chk("measured items", len(meas), "nMeasured")
chk("covered items", len([r for r in meas if r["covered"] and not r["excluded"]]), "nCovered")

# 3. Kaiser worked example (independent arithmetic)
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

# 4. the ten gap items -> spot-check three representative gaps
for key, mk in [("rev_debts", "debtsGap"), ("infl_end", "inflendGap"), ("war_us_entry", "usentryGap")]:
    chk(f"gap {key}", f"{er[key]['gap']:+.2f}", mk)

# 5. DE/FR exams
for lg, cap in [("de", "DE"), ("fr", "FR")]:
    cv = [r for r in defr_cert["rows"][lg] if r["covered"]]
    chk(f"{cap} exam answered", len(cv), cap + "examAnswered")
    chk(f"{cap} exam correct", sum(1 for r in cv if r["correct"]), cap + "examCorrect")
    chk(f"{cap} exam out-bet", sum(1 for r in cv if r["grad"] > 0), cap + "examOutbet")

# 6. false alarm
chk("false alarm old %", pctround(next(r for r in v4["rows"] if r["name"] == "nazi_who_rules")["p_truth_pre"]), "faOld")
chk("false alarm new %", pctround(er["nazi_who_rules"]["p_truth_pre"]), "faNew")

# 7. cross-lingual Kaiser
for lg, cap in [("de", "DE"), ("fr", "FR")]:
    r = {x["name"]: x for x in defr_bat["rows"][lg]}["emp_kaiser_1919"]
    chk(f"Kaiser {cap} witness %", pctround(r["p_truth_post"]), "kaiser" + cap + "Post")
    chk(f"Kaiser {cap} gap", f"{r['gap']:+.1f}", "kaiser" + cap + "Gap")

# 8. memoir bets used in prose
chk("memoir banks %", pctround(er["dep_banks"]["pre_bet_p"]), "depBanksMemoir")
chk("memoir dictatorship %", pctround(er["nazi_republic_fate"]["pre_bet_p"]), "naziDictMemoir")
chk("memoir gold %", pctround(er["dep_gold"]["pre_bet_p"]), "depGoldMemoir")

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
