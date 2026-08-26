# -*- coding: utf-8 -*-
"""
CONTAMINATION TRIAGE — is each foresight finding real foresight, or the guesser peeking?
========================================================================================
Version : v1  (2026-07-08)
Author  : Opus (post-date-spike; retrain parked, so we SCREEN the contaminated models instead)

The date-spike proved the frozen models leak canonical post-cutoff DATES. Retrain is a separate track,
so for THIS paper the guessers are known-contaminated. We screen every finding with a decision tree that
needs no corpus and no retrain:

  date_leak    : does the GUESSER put more probability on the event's TRUE YEAR than a *fictional* event
                 reaches by chance? (its own 95th-percentile fiction floor, from events_dates.NULL)
                 -> date-level contamination: does the model even know WHEN it happened?
  outcome_lean : does the guesser lean toward the TRUE OUTCOME the paper claims? (its top bet == truth,
                 or p_truth_pre >= 0.5, from v41_battery.json)

  DECISION:
    not date_leak                    -> SAFE    (can't know the outcome of an event it doesn't know happened)
    date_leak AND not outcome_lean   -> SAFE    (knew the event but bet the WRONG outcome, e.g. Ebert)
    date_leak AND outcome_lean       -> SUSPECT (knew the event AND leans to the truth -> could be leakage)

Guessers = the BASE Ranke models (as in the foresight battery). Fiction floor computed per guesser.
Inputs : results/v41_battery.json, results/gate_report.json, events_dates.NULL/WIN.
Output : results/superlatives/contamination_triage.json  (+ console table)
Run    : python src/contamination_triage.py
"""
import os, json, math
import numpy as np
import common
import events_dates as E
from foresight_bank import RANKE

# each foresight EVENT -> (canonical dated probe, true year, guesser tag = pre-cutoff model)
EVENTS = {
 "panic_1907":    ("The great financial panic struck the banks of New York in the year", 1907, "1900"),
 "world_war":     ("The First World War broke out in Europe in the year", 1914, "1913"),
 "empires":       ("The German Kaiser abdicated and Germany became a republic in the year", 1918, "1913"),
 "revolution":    ("The Bolsheviks seized power in Russia in the year", 1917, "1913"),
 "spanish_flu":   ("The great influenza pandemic swept the world in the year", 1918, "1913"),
 "hyperinflation":("The German mark collapsed in a ruinous hyperinflation in the year", 1923, "1913"),
 "depression":    ("The Wall Street stock market crashed, beginning the Depression, in the year", 1929, "1929"),
 "nazi_seizure":  ("Adolf Hitler became Chancellor of Germany in the year", 1933, "1929"),
}


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


def p_true_year(model, tok, prompt, ty):
    years = list(range(ty - E.WIN, ty + E.WIN + 1))
    p = softmax([common.slp(model, tok, prompt, f" {y}") for y in years])
    return p[years.index(ty)] * 100.0


# --- score fiction floor + each event's date on its guesser (base) model ---
tags = sorted({g for _, _, g in EVENTS.values()})
floor, event_date_p = {}, {}
for tag in tags:
    mid = RANKE[tag]
    print(f"=== guesser {mid} ({tag}) ===", flush=True)
    tok, model = common.load(mid)
    nulls = [p_true_year(model, tok, pr, ty) for _, pr, ty in E.NULL]
    floor[tag] = round(float(np.percentile(nulls, 95)), 1)     # this guesser's "no-knowledge" ceiling
    for ev, (pr, ty, g) in EVENTS.items():
        if g == tag:
            event_date_p[ev] = round(p_true_year(model, tok, pr, ty), 1)
    del tok, model
    common.free()

# --- join to the foresight findings ---
rows = json.load(open(os.path.join(common.RESULTS, "v41_battery.json"), encoding="utf-8"))["rows"]
gate = json.load(open(os.path.join(common.RESULTS, "gate_report.json"), encoding="utf-8"))
gated = {r["name"] for r in gate["passed"]}

triage = []
for r in rows:
    ev = r["event"]
    if ev not in EVENTS or r["excluded"]:
        continue
    g = EVENTS[ev][2]
    dP, fl = event_date_p[ev], floor[g]
    date_leak = dP > fl
    top_is_truth = (r["pre_bets"] == r["truth_opt"])
    outcome_lean = top_is_truth or (r["p_truth_pre"] >= 0.5)
    if not date_leak:
        verdict, why = "SAFE", "guesser is date-clean: doesn't know the event happened"
    elif not outcome_lean:
        verdict, why = "SAFE", "knew the event but bet the WRONG outcome"
    else:
        verdict, why = "SUSPECT", "knew the event AND leans to the truth — possible leakage"
    triage.append({"event": ev, "name": r["name"], "guesser": g, "event_year": EVENTS[ev][1],
                   "truth_opt": r["truth_opt"], "gap": r["gap"],
                   "guesser_date_pct": dP, "fiction_floor_pct": fl, "date_leak": date_leak,
                   "guesser_p_truth": r["p_truth_pre"], "guesser_top_bet": r["pre_bets"],
                   "top_bet_is_truth": top_is_truth, "outcome_lean": outcome_lean,
                   "gated": r["name"] in gated, "memoir_only": r["memoir_only"],
                   "verdict": verdict, "why": why})

out = {"meta": {"version": "v1", "date": "2026-07-08",
                "method": "date_leak (guesser P(true year) > its 95th-pct fiction floor) x outcome_lean (top bet==truth or p_truth_pre>=.5)",
                "fiction_floor_pct_by_guesser": floor, "event_date_pct": event_date_p},
       "triage": sorted(triage, key=lambda d: (d["verdict"] != "SUSPECT", -(d["gap"] or -99)))}
OUT = os.path.join(common.RESULTS, "superlatives", "contamination_triage.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# --- console ---
print("\n" + "=" * 104)
print("CONTAMINATION TRIAGE — screening the (known-contaminated) foresight findings")
print("=" * 104)
print("fiction floor by guesser (a made-up event reaches this by chance):",
      "  ".join(f"{t}={floor[t]:.0f}%" for t in tags))
print(f"\n{'finding':22s}{'guess':>6}{'evYr':>6}{'dateP%':>8}{'floor':>6}  {'leak?':6}{'pTruth':>7} {'gated':6}{'verdict':>9}  why")
for d in out["triage"]:
    print(f"{d['name']:22s}{d['guesser']:>6}{d['event_year']:>6}{d['guesser_date_pct']:>7.0f}%{d['fiction_floor_pct']:>6.0f}"
          f"  {'LEAK' if d['date_leak'] else 'clean':6}{d['guesser_p_truth']:>7.2f} {'Y' if d['gated'] else '.':>4}  {d['verdict']:>9}  {d['why']}")

print("\n--- GATED findings only (what the paper would report) ---")
for d in out["triage"]:
    if d["gated"]:
        tag = "*** SUSPECT ***" if d["verdict"] == "SUSPECT" else "safe"
        print(f"  {d['name']:20s} gap {d['gap']:+6.2f}  guesser {d['guesser_date_pct']:>4.0f}% vs floor {d['fiction_floor_pct']:.0f}%  "
              f"p(truth) {d['guesser_p_truth']:.2f}  -> {tag}")
print(f"\nwrote {os.path.relpath(OUT, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}")
