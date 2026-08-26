# -*- coding: utf-8 -*-
"""CONTAMINATION GATE for the panel — integrity before results.

For each event x language: 6 post-event fact probes, 4-way CALIBRATED forced choice
    c(opt) = slp(opt|stem) - slp(opt|neutral);  pred = argmax
run under (a) the event's designated PRE model — GREEN = accuracy consistent with chance (25%),
i.e. its low conceptual surprise cannot be leakage — and (b) a HINDSIGHT CEILING model (the event's
anchor) which validates the probes themselves: if the model that read the outcome cannot answer
them, the probes are bad, not the pre model clean.

Items with known period-prior caveats (flagged in panel_checks.py) are excluded from the R/A/G
verdict and reported separately, per the prereg. Checkpoints per model; resume-safe.
Run: python src/gate_panel.py
"""
import os, json
import common
from foresight_bank import RANKE, MODERN
from panel_bank import assert_frozen_matches
from panel_events import EVENTS_META, LANGS
from panel_checks import GATES

# event -> (pre model tag, ceiling model tag)
ROLES = {ev: (meta["pre"][-1], meta["anchor"]) for ev, meta in EVENTS_META.items()}
# gate items with a known pre-event prior that could help a clean model guess (see panel_checks.py)
FLAGGED = {"panic_1907": [5], "world_war": [5], "fall_of_empires": [5], "great_depression": [4]}
MID = dict(RANKE, modern=MODERN)

OUT = os.path.join(common.RESULTS, "gate_panel.json")
frozen = assert_frozen_matches()
res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {"scores": {}}

# group work by model so each loads once: model_tag -> list of (event, role)
work = {}
for ev, (pre, ceil) in ROLES.items():
    work.setdefault(pre, []).append((ev, "pre"))
    work.setdefault(ceil, []).append((ev, "ceiling"))

for tag, jobs in work.items():
    todo = [(ev, role) for ev, role in jobs if f"{ev}/{role}" not in res["scores"]]
    if not todo:
        print(f"[{tag}] all done, skipping")
        continue
    print(f"=== {MID[tag]} ({tag}): {[f'{e}/{r}' for e, r in todo]} ===", flush=True)
    tok, model = common.load(MID[tag])
    for ev, role in todo:
        rows = []
        for i, g in enumerate(GATES[ev]):
            for lg in LANGS:
                stem, opts, ci, neu = g[lg]
                c = [common.slp(model, tok, stem, o) - common.slp(model, tok, neu, o) for o in opts]
                pred = max(range(len(c)), key=lambda k: c[k])
                rows.append({"i": i, "lang": lg, "pred": pred, "correct_idx": ci,
                             "correct": int(pred == ci), "flagged": int(i in FLAGGED.get(ev, [])),
                             "c": [round(x, 3) for x in c]})
        res["scores"][f"{ev}/{role}"] = {"model": tag, "rows": rows}
        json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  {ev}/{role}: scored ({sum(r['correct'] for r in rows)}/{len(rows)} raw)", flush=True)
    del tok, model
    common.free()

# ---- verdicts ----
verdicts = {}
for ev in EVENTS_META:
    for role in ("pre", "ceiling"):
        rows = res["scores"][f"{ev}/{role}"]["rows"]
        for lg in list(LANGS) + ["all"]:
            rr = [r for r in rows if (lg == "all" or r["lang"] == lg) and not r["flagged"]]
            k, n = sum(r["correct"] for r in rr), len(rr)
            p = common.binom_ge(k, n, 0.25)
            if role == "pre":
                v = "GREEN" if p > 0.05 else ("AMBER" if p > 0.005 else "RED")
            else:
                v = "OK" if k / max(1, n) >= 0.5 else "WEAK-PROBES"
            verdicts.setdefault(f"{ev}/{role}", {})[lg] = {"k": k, "n": n, "p_binom": round(p, 4), "verdict": v}
        fr = [r for r in rows if r["flagged"]]
        if fr:
            verdicts[f"{ev}/{role}"]["flagged_correct"] = f"{sum(r['correct'] for r in fr)}/{len(fr)}"
# ---- POST-HOC sensitivity (declared as post-hoc, NOT part of the registered verdict) ----
# russian_revolution gate item 3 (peace signed at Brest-Litovsk): the pre model answers it 3/3 while
# flooring on Lenin (0/3), the party (1/3) and the month (1/3) — the signature of a geographic
# salience prior (Brest-Litovsk was the famous border fortress on the Warsaw road), not event
# knowledge. Sensitivity: verdict excluding that one item, reported alongside the registered AMBER.
POSTHOC = {"russian_revolution": [3]}
for ev, idxs in POSTHOC.items():
    rows = res["scores"][f"{ev}/pre"]["rows"]
    rr = [r for r in rows if not r["flagged"] and r["i"] not in idxs]
    k, n = sum(r["correct"] for r in rr), len(rr)
    p = common.binom_ge(k, n, 0.25)
    verdicts[f"{ev}/pre"]["posthoc_excluding"] = {
        "items": idxs, "k": k, "n": n, "p_binom": round(p, 4),
        "verdict": "GREEN" if p > 0.05 else ("AMBER" if p > 0.005 else "RED"),
        "note": "post-hoc sensitivity — geographic salience prior; registered verdict unchanged"}

res["verdicts"] = verdicts
res["meta"] = {"bank_sha256": frozen["sha256"], "flagged": FLAGGED, "posthoc": POSTHOC,
               "note": "pre GREEN = at chance (clean); ceiling OK = probes are answerable with hindsight"}
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("\n" + "=" * 88)
print(f"{'event':20s}{'pre model':>10s}{'pre all':>16s}{'ceiling':>10s}{'ceiling all':>14s}")
for ev, (pre, ceil) in ROLES.items():
    vp = verdicts[f"{ev}/pre"]["all"]
    vc = verdicts[f"{ev}/ceiling"]["all"]
    print(f"{ev:20s}{pre:>10s}{vp['verdict']:>9s} {vp['k']}/{vp['n']:<4d}{ceil:>10s}{vc['verdict']:>10s} {vc['k']}/{vc['n']}")
print("wrote results/gate_panel.json")
