# -*- coding: utf-8 -*-
"""Recompute the 1913 verdicts under battery v1.2 roles (JV's innocents audit, 2026-07-13) from the
SAVED lifts in contam_pipeline_1913.json — no rescoring; the numbers are unchanged, only their roles.
Changes: Trotsky 'exile' tell -> innocent (pre-1913 exiles are his own biography); 'revolution'
excluded as sense-ambiguous; poison-gas 'fumes'/'mines' excluded as lexical near-synonyms; Kemal
marked provisional_name_culture. Output: contam_pipeline_1913_v12.json.
Run:  python src/reclass_v12.py
"""
import os, json
import common

S = os.path.join(common.RESULTS, "superlatives")
base = json.load(open(os.path.join(S, "contam_pipeline_1913.json"), encoding="utf-8"))
bat = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "battery_assoc_v1.json"),
                     encoding="utf-8"))
floor = base["meta"]["floor95"]
out = {"meta": dict(base["meta"], battery_version="v1.2",
                    note="verdicts recomputed from saved v1.1 lifts under v1.2 roles (JV innocents audit)"),
       "targets": {}}
by_name = {t["name"]: t for t in bat["cutoffs"]["1913"]["targets"]}
for name, r in base["targets"].items():
    spec = by_name[name]
    all_lifts = dict(r["all_tell_lifts"], **r["all_innocent_lifts"])
    tells = [d for d in spec["tells"]]
    fired = [d for d in tells if all_lifts.get(d["w"], -99) > floor]
    clusters = sorted({d["cluster"] for d in fired})
    verdict = "FLAGGED" if len(clusters) >= 2 else ("WHISPER" if len(clusters) == 1 else "DARK")
    innocents = {w: all_lifts[w] for w in spec.get("innocents", []) if all_lifts.get(w, -99) > floor}
    out["targets"][name] = {"status": spec["status"], "verdict": verdict,
                            "clusters_fired": clusters,
                            "tells_fired": {d["w"]: all_lifts[d["w"]] for d in fired},
                            "innocents_fired": innocents,
                            "excluded_ambiguous": spec.get("excluded_ambiguous", [])}
    old = base["targets"][name]["verdict"]
    if old != verdict:
        print(f"  VERDICT CHANGE {name}: {old} -> {verdict}")
json.dump(out, open(os.path.join(S, "contam_pipeline_1913_v12.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("v1.2 verdicts:")
for n, r in out["targets"].items():
    print(f"  {n:14s}{r['verdict']:9s}clusters: {', '.join(r['clusters_fired']) or '-'}"
          f"   innocents fired: {', '.join(r['innocents_fired']) or '-'}")
print("wrote contam_pipeline_1913_v12.json")
