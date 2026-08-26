# -*- coding: utf-8 -*-
"""
FREQUENCY-CONTROL (model-internal) — is a guesser's prediction reasoning, or just print-familiarity?
====================================================================================================
Version : v1  (2026-07-07)
Author  : Opus (Fable-review Phase 1c)

THE CONFOUND (this whole thread). When the pre-event "guesser" bets on an outcome, is it because the
outcome FITS the context (genuine expectation) or merely because that word is FAMILIAR in the corpus
(print frequency)? Fable: "the availability heuristic and a token counter are observationally
equivalent." We separate them with a quantity the pipeline already has:
  base[o]  = slp(neutral, o)          -> context-free familiarity of option o  (a print-share PROXY)
  raw[o]   = mean_stems slp(stem, o)  -> preference WITH the informative context
  p_base = softmax(base) ; p_raw = mean_stem softmax(raw_stem) ; p_cal = mean_stem softmax(raw-base)
"lift" = p_raw - p_base is how much the CONTEXT moves an option away from bare familiarity.

READOUTS.
  (1) Pooled: regress p_raw on p_base across every (item, option). Low R^2 / big |lift| => context
      reshapes predictions (reasoning), not just frequency. High R^2, slope~1 => predictions ~ familiarity.
  (2) Per finding: for the guesser's BET (its prediction) and for the TRUTH, show familiarity vs
      contextual prob and the lift. The availability reading needs the bet LIFTED above its base.
  (3) Flagship "unforeseen" (Ebert, Rentenmark): is the truth suppressed BELOW its base by context
      (actively ruled out) or merely unfamiliar (low base)?

This is the corpus-free FIRST CUT. The full version (Phase 2) counts print share in the training corpus
by 5-year window to split recency from volume; requires corpus access.

INPUT: each item scored under its OWN pre-event guesser (1900/1913/1929).
OUTPUT: ...\\12_foresight\\results\\freq_control_internal.json
Run:  python src/frequency_control.py
"""
import os, json, statistics as st
import numpy as np
import common
import v41_battery as vb
from foresight_bank import RANKE

# gate outcome (from results/gate_report.json) so we can tag findings
gate = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                                    "gate_report.json"), encoding="utf-8"))
PASSED = {r["name"] for r in gate["passed"]}
DROPPED = {r["name"] for r in gate["dropped"]}

# score every item under its guesser (pre) model; capture base + raw + cal
by_pre = {}
for q in vb.Q:
    by_pre.setdefault(q["pre"], []).append(q)

decomp = {}   # name -> dict of vectors
for pre_tag, items in by_pre.items():
    mid = RANKE[pre_tag]
    print(f"=== guesser {mid} ({pre_tag}) : {len(items)} items ===", flush=True)
    tok, model = common.load(mid)
    for q in items:
        base = [common.slp(model, tok, q["neutral"], o) for o in q["options"]]
        raw_ps = []
        for stem in q["stems"]:
            raw = [common.slp(model, tok, stem, o) for o in q["options"]]
            raw_ps.append(vb.softmax(raw))
        p_base = vb.softmax(base)
        p_raw = vb.mean_p(raw_ps)
        decomp[q["name"]] = {"pre": pre_tag, "truth": q["truth"], "options": [o.strip() for o in q["options"]],
                             "p_base": [round(x, 4) for x in p_base], "p_raw": [round(x, 4) for x in p_raw]}
    del tok, model
    common.free()

# (1) pooled regression p_raw ~ p_base over all (item, option)
xs, ys = [], []
for d in decomp.values():
    xs += d["p_base"]; ys += d["p_raw"]
xs, ys = np.array(xs), np.array(ys)
b, a = np.polyfit(xs, ys, 1)                       # y = b*x + a
yhat = b * xs + a
ss_res = float(np.sum((ys - yhat) ** 2)); ss_tot = float(np.sum((ys - ys.mean()) ** 2))
r2 = 1 - ss_res / ss_tot
mean_abs_lift = float(np.mean(np.abs(ys - xs)))
corr = float(np.corrcoef(xs, ys)[0, 1])

# (2) per-finding decomposition
def finding_row(name):
    d = decomp[name]; t = d["truth"]
    bet = int(np.argmax(d["p_raw"]))
    return {
        "name": name, "pre": d["pre"], "gate": ("PASS" if name in PASSED else "DROP" if name in DROPPED else "n/a"),
        "truth_opt": d["options"][t],
        "truth_familiarity_base": d["p_base"][t], "truth_contextual_raw": d["p_raw"][t],
        "truth_lift": round(d["p_raw"][t] - d["p_base"][t], 4),
        "guesser_bet": d["options"][bet],
        "bet_familiarity_base": d["p_base"][bet], "bet_contextual_raw": d["p_raw"][bet],
        "bet_lift": round(d["p_raw"][bet] - d["p_base"][bet], 4),
    }

findings = [finding_row(n) for n in decomp]
# order: passed unforeseen first, then dropped, then rest
order = {"PASS": 0, "DROP": 1, "n/a": 2}
findings.sort(key=lambda r: (order[r["gate"]], -abs(r["bet_lift"])))

# (3) availability summary: among items where the guesser's bet is WRONG (bet != truth), how often is the
# wrong bet LIFTED above its base familiarity by context? (context actively selecting the precedent)
wrong = [r for r in findings if r["guesser_bet"] != r["truth_opt"]]
wrong_lifted = [r for r in wrong if r["bet_lift"] > 0.05]
avail_share = (len(wrong_lifted) / len(wrong)) if wrong else 0.0

if r2 >= 0.75:
    claim = ("WEAK claim supported: contextual predictions largely track context-free familiarity "
             f"(R^2={r2:.2f}); the pre-event models mostly re-weight what is FREQUENT. Section 7 should say "
             "'the written record weighs the future by frequency', not 'diagnostic expectations'.")
elif avail_share >= 0.5:
    claim = (f"STRONGER claim supported: predictions are NOT mostly familiarity (R^2={r2:.2f}, mean |lift| "
             f"{mean_abs_lift:.2f}), and in {avail_share:.0%} of wrong bets the context actively LIFTS the "
             "salient precedent above its bare familiarity — an availability/diagnostic-expectations pattern, "
             "not raw token-counting. NB: only the corpus-print-share version (Phase 2) can fully clinch it.")
else:
    claim = (f"MIXED: context reshapes predictions (R^2={r2:.2f}, mean |lift| {mean_abs_lift:.2f}) but the "
             "wrong bets are not systematically lifted above familiarity; the internal proxy is inconclusive "
             "on availability-vs-frequency. Needs the corpus print-share test (Phase 2).")

out = {"meta": {"version": "v1", "date": "2026-07-07",
                "what": "model-internal familiarity (base) vs contextual (raw) decomposition of guesser predictions",
                "limitation": "base=slp(neutral,option) is a corpus-free proxy for print share; full test needs corpus counts"},
       "pooled_regression": {"slope": round(float(b), 3), "intercept": round(float(a), 3),
                             "r2_praw_on_pbase": round(float(r2), 3), "corr": round(corr, 3),
                             "mean_abs_lift": round(mean_abs_lift, 3), "n_option_instances": len(xs)},
       "availability": {"n_wrong_bets": len(wrong), "n_wrong_bets_context_lifted": len(wrong_lifted),
                        "share_lifted": round(avail_share, 3)},
       "claim_supported": claim,
       "findings": findings}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "freq_control_internal.json")
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("\n" + "=" * 96)
print("FREQUENCY-CONTROL (model-internal): guesser prediction = familiarity (base) vs context (raw)")
print("=" * 96)
print(f"pooled  p_raw ~ p_base : R^2={r2:.2f}  slope={b:.2f}  corr={corr:.2f}  mean|lift|={mean_abs_lift:.2f}  (n={len(xs)})")
print(f"wrong bets context-lifted above familiarity: {len(wrong_lifted)}/{len(wrong)} ({avail_share:.0%})\n")
print(f"{'name':20s}{'gate':5s} {'truth: base->raw (lift)':30s} {'bet: base->raw (lift)':30s}")
for r in findings:
    tt = f"{r['truth_familiarity_base']:.2f}->{r['truth_contextual_raw']:.2f} ({r['truth_lift']:+.2f})"
    bb = f"{r['bet_familiarity_base']:.2f}->{r['bet_contextual_raw']:.2f} ({r['bet_lift']:+.2f})"
    print(f"{r['name']:20s}{r['gate']:5s} {tt:30s} {bb:30s} bet={r['guesser_bet'][:22]}")
print("\nCLAIM:", claim)
print("wrote results/freq_control_internal.json")
