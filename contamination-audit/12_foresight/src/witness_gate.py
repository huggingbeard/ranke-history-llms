# -*- coding: utf-8 -*-
"""
WITNESS-GATE re-analysis of the v4.1 foresight battery.
=======================================================
Version : v1  (2026-07-07)
Author  : Opus (Fable-review Phase 1a)

WHAT THIS DOES (in plain words)
-------------------------------
The foresight "gap" for a question is  gap = -ln p_guesser(truth) + ln p_witness(truth).
The WITNESS is the reader that lived through the event (the post-cutoff Ranke model). If the
witness itself cannot find the true answer, the gap is a ratio of two tiny, unstable numbers and
means nothing. Fable's fix: a question may enter the paper only if the witness is confident in the
truth. We implement that as an inclusion rule and show exactly what it removes.

INCLUSION RULE (locked): keep a finding iff  p_truth_post (witness confidence) >= THRESH (=0.50).

INPUTS  (full paths):
  D:\\UZHechist Dropbox\\Joachim Voth\\claudecode\\ranke-contamination-audit\\12_foresight\\results\\v41_battery.json
OUTPUTS (full paths):
  ...\\12_foresight\\results\\gate_report.json     (every finding, PASS/DROP, both readers' slips)
No model is loaded; this is pure re-analysis of the frozen battery.

Run:  python src/witness_gate.py
"""
import os, json, statistics as st

THRESH = 0.50
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "src" else HERE
RESULTS = os.path.join(ROOT, "results")
SRC = os.path.join(RESULTS, "v41_battery.json")
OUT = os.path.join(RESULTS, "gate_report.json")

rows = json.load(open(SRC, encoding="utf-8"))["rows"]

# ---- classify every row ----
# A "finding" is a measured (non-memoir), non-excluded, covered row: it has a witness and a gap.
# Memoir-only rows have no witness (post is null) -> cannot be gated; flagged for the 1939 model.
findings, memoir_only, excluded, abstained = [], [], [], []
for r in rows:
    if r["memoir_only"]:
        memoir_only.append(r)
    elif r["excluded"]:
        excluded.append(r)
    elif not r["covered"]:
        abstained.append(r)          # meter refused to answer (raw/calibrated disagreed)
    else:
        findings.append(r)

def slip(r):
    """A compact 'betting slip' for both readers, from what the battery stored."""
    return {
        "event": r["event"], "name": r["name"], "flag": r["flag"],
        "truth_opt": r["truth_opt"],
        "guesser_p_truth": r["p_truth_pre"],      # pre-event reader on the truth
        "guesser_top_bet": r["pre_bets"], "guesser_top_p": r["pre_bet_p"],
        "witness_p_truth": r["p_truth_post"],     # THE GATE VARIABLE
        "witness_top_bet": r["post_says"],
        "gap": r["gap"],
    }

passed = [r for r in findings if r["p_truth_post"] is not None and r["p_truth_post"] >= THRESH]
dropped = [r for r in findings if r["p_truth_post"] is not None and r["p_truth_post"] < THRESH]

# ---- event-mean gaps, before vs after the gate (covered findings only) ----
EV = ["panic_1907", "world_war", "empires", "revolution", "spanish_flu", "hyperinflation"]
def event_means(rowset):
    out = {}
    for ev in EV:
        gaps = [r["gap"] for r in rowset if r["event"] == ev and r["gap"] is not None]
        if gaps:
            out[ev] = {"mean_gap": round(st.mean(gaps), 3), "n": len(gaps),
                       "items": [r["name"] for r in rowset if r["event"] == ev and r["gap"] is not None]}
    return out

before = event_means(findings)
after = event_means(passed)

report = {
    "meta": {
        "version": "v1", "date": "2026-07-07",
        "rule": f"keep finding iff witness confidence p_truth_post >= {THRESH}",
        "source": os.path.relpath(SRC, ROOT).replace("\\", "/"),
        "note": "pure re-analysis of v41_battery.json; no model loaded",
    },
    "counts": {
        "findings_before_gate": len(findings),
        "passed": len(passed), "dropped": len(dropped),
        "memoir_only_no_witness": len(memoir_only),
        "abstained_meter": len(abstained), "excluded_domain": len(excluded),
    },
    "passed": [slip(r) for r in sorted(passed, key=lambda r: -r["gap"])],
    "dropped": [slip(r) for r in sorted(dropped, key=lambda r: -r["gap"])],
    "memoir_only_no_witness": [slip(r) for r in memoir_only],
    "event_mean_gap_before_gate": before,
    "event_mean_gap_after_gate": after,
}
json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- console summary ----
print("=" * 90)
print(f"WITNESS-GATE  (keep iff witness p(truth) >= {THRESH:.2f})")
print("=" * 90)
print(f"findings (covered, measured): {len(findings)}   ->   PASS {len(passed)}   DROP {len(dropped)}")
print(f"set aside: memoir-only (no witness) {len(memoir_only)} | meter-abstained {len(abstained)} | domain-excluded {len(excluded)}")

print("\n--- DROPPED (witness could not read the truth) ---")
print(f"{'name':22s}{'witness p(truth)':>17s}{'gap':>8s}   witness bet instead")
for r in sorted(dropped, key=lambda r: -r["gap"]):
    print(f"{r['name']:22s}{r['p_truth_post']:>17.3f}{r['gap']:>+8.2f}   {r['post_says']}")

print("\n--- PASSED (witness confident; gap is real) ---")
print(f"{'name':22s}{'witness p(truth)':>17s}{'gap':>8s}   verdict")
for r in sorted(passed, key=lambda r: -r["gap"]):
    verdict = "UNFORESEEN" if r["gap"] >= 1.0 else "foreseen"
    print(f"{r['name']:22s}{r['p_truth_post']:>17.3f}{r['gap']:>+8.2f}   {verdict}")

print("\n--- event-mean gap: BEFORE -> AFTER the gate ---")
for ev in EV:
    b = before.get(ev); a = after.get(ev)
    bs = f"{b['mean_gap']:+.2f} (n={b['n']})" if b else "--"
    as_ = f"{a['mean_gap']:+.2f} (n={a['n']})" if a else "-- (all dropped)"
    print(f"  {ev:16s} {bs:>16s}  ->  {as_}")

print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
