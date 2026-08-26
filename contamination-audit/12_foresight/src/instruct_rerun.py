# -*- coding: utf-8 -*-
"""
INSTRUCT re-run — repeat the catastrophe battery + control yardstick with the chat-tuned Ranke heads,
as a robustness check on the base results. Base witnesses were factually spotty (only 16/40 controls
gated); the chat-tuned heads may commit harder and let more findings through.

Instruct heads used: Ranke-4B-1913-1202-Instruct, Ranke-4B-1929-1202-Instruct. NO 1900 instruct exists
on the hub, so window-A GUESSERS stay base Ranke-4B-1900 (their witnesses are 1913-Instruct). Scored with
the SAME raw teacher-forcing as base (not chat-formatted) for a like-for-like comparison. CAVEAT: instruct
tuning adds a separate post-training knowledge/confidence channel — this is a robustness check, not ground
truth (see .oracle_tmp memo: "use base where possible; instruction models add a post-training channel").

Version v1 (2026-07-07).
Outputs: results/{v41_battery_instruct, gate_report_instruct, control_gaps_instruct}.json
Run: python src/instruct_rerun.py
"""
import os, json, math
import common
import v41_battery as vb
import control_battery as cb
from foresight_bank import RANKE

INSTRUCT = {"1900": RANKE["1900"],          # no 1900 instruct on the hub -> base guesser for window A
            "1913": "uzh-echist-org/Ranke-4B-1913-1202-Instruct",
            "1929": "uzh-echist-org/Ranke-4B-1929-1202-Instruct"}
THRESH = 0.50
RES = common.RESULTS


def score_into(model, tok, q, tag):
    raw_ps, cal_ps = [], []
    for stem in q["stems"]:
        raw = [common.slp(model, tok, stem, o) for o in q["options"]]
        base = [common.slp(model, tok, q["neutral"], o) for o in q["options"]]
        raw_ps.append(vb.softmax(raw))
        cal_ps.append(vb.softmax([r - b for r, b in zip(raw, base)]))
    vb.res.setdefault(q["name"], {})[tag] = {"raw_ps": raw_ps, "cal_ps": cal_ps}


# ============ catastrophe battery under instruct ============
needed = set()
for q in vb.Q:
    needed.add(q["pre"])
    if q["post"]:
        needed.add(q["post"])
for tag in sorted(needed):
    mid = INSTRUCT[tag]
    items = [q for q in vb.Q if q["pre"] == tag or q.get("post") == tag]
    print(f"=== catastrophe {mid} ({tag}): {len(items)} items ===", flush=True)
    tok, model = common.load(mid)
    for q in items:
        score_into(model, tok, q, tag)
    del tok, model
    common.free()

rows = []
for q in vb.Q:
    p_pre, cov_pre = vb.score(q, q["pre"])
    if q["post"]:
        p_post, cov_post = vb.score(q, q["post"])
        covered = cov_pre and cov_post
        gap = -math.log(max(p_pre[q["truth"]], 1e-6)) + math.log(max(p_post[q["truth"]], 1e-6))
    else:
        p_post, covered, gap = None, cov_pre, None
    rows.append(dict(event=q["event"], name=q["name"], memoir_only=q["post"] is None, covered=covered,
                     excluded=q["name"] in vb.EXCLUDE_FROM_AGG,
                     truth_opt=q["options"][q["truth"]].strip(),
                     p_truth_pre=round(p_pre[q["truth"]], 3),
                     p_truth_post=(round(p_post[q["truth"]], 3) if p_post else None),
                     gap=(round(gap, 2) if gap is not None else None),
                     pre_bets=q["options"][max(range(4), key=lambda k: p_pre[k])].strip(),
                     pre_bet_p=round(max(p_pre), 3),
                     post_says=(q["options"][max(range(4), key=lambda k: p_post[k])].strip() if p_post else None),
                     flag=q.get("flag", "")))
json.dump({"rows": rows}, open(os.path.join(RES, "v41_battery_instruct.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# witness-gate (identical rule to witness_gate.py)
findings = [r for r in rows if not r["memoir_only"] and not r["excluded"] and r["covered"]]
def slip(r):
    return {"event": r["event"], "name": r["name"], "flag": r["flag"], "truth_opt": r["truth_opt"],
            "guesser_p_truth": r["p_truth_pre"], "guesser_top_bet": r["pre_bets"], "guesser_top_p": r["pre_bet_p"],
            "witness_p_truth": r["p_truth_post"], "witness_top_bet": r["post_says"], "gap": r["gap"]}
passed = [r for r in findings if r["p_truth_post"] is not None and r["p_truth_post"] >= THRESH]
dropped = [r for r in findings if r["p_truth_post"] is not None and r["p_truth_post"] < THRESH]
gate_out = {"meta": {"version": "v1", "date": "2026-07-07", "rule": f"witness p_truth >= {THRESH}",
                     "models": "INSTRUCT heads (1913/1929); window-A guessers are base-1900"},
            "counts": {"findings_before_gate": len(findings), "passed": len(passed), "dropped": len(dropped)},
            "passed": [slip(r) for r in sorted(passed, key=lambda r: -(r["gap"] or 0))],
            "dropped": [slip(r) for r in sorted(dropped, key=lambda r: -(r["gap"] or 0))]}
json.dump(gate_out, open(os.path.join(RES, "gate_report_instruct.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# ============ control yardstick under instruct ============
cb.run(INSTRUCT, "control_gaps_instruct.json", cata_gate="gate_report_instruct.json")

# ============ base vs instruct comparison ============
def maybe(path):
    p = os.path.join(RES, path)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
base_gate, base_ctrl = maybe("gate_report.json"), maybe("control_gaps.json")
inst_ctrl = maybe("control_gaps_instruct.json")
inst_gate = {"passed": passed, "dropped": dropped}

def wit(gate, name):
    for r in gate["passed"] + gate.get("dropped", []):
        if r["name"] == name:
            return r.get("witness_p_truth"), r.get("gap")
    return None, None

print("\n" + "=" * 84)
print("BASE  vs  INSTRUCT")
print("=" * 84)
if base_gate:
    print(f"catastrophe findings passing witness-gate:  base {base_gate['counts']['passed']:>2d}  ->  instruct {len(passed):>2d}")
if base_ctrl and inst_ctrl:
    print(f"control coverage (gated / 40):               base {base_ctrl['meta']['n_gated']:>2d}  ->  instruct {inst_ctrl['meta']['n_gated']:>2d}")
    print(f"mundane 90th-pct gap:                        base {base_ctrl['mundane_gap_percentiles'].get('90')}  ->  instruct {inst_ctrl['mundane_gap_percentiles'].get('90')}")
print("\nflagship witness confidence & gap  (base -> instruct):")
for nm in ["emp_kaiser_1919", "infl_end", "rev_tsar_fate", "rev_who_rules", "war_us_entry", "rev_debts"]:
    bw, bg = wit(base_gate, nm) if base_gate else (None, None)
    iw, ig = wit(inst_gate, nm)
    print(f"  {nm:16s} witness {str(bw):>6} -> {str(iw):>6}   |   gap {str(bg):>6} -> {str(ig):>6}")
if inst_ctrl:
    print("\ninstruct: catastrophes on the instruct mundane scale:")
    for d in inst_ctrl["catastrophes_on_mundane_scale"]:
        print(f"  {d['name']:18s} gap {d['gap']:+6.2f} -> {d['percentile_in_mundane']}th pct")
print("\nwrote v41_battery_instruct.json / gate_report_instruct.json / control_gaps_instruct.json")
