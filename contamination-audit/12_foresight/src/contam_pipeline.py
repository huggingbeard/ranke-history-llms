# -*- coding: utf-8 -*-
"""CONTAMINATION PIPELINE — the association battery runner (Thread C; goal: a fully validated
pipeline using activations and inserted memories to identify contaminated memories in time-locked models).

For every target in battery_assoc_v1.json (v1.1):
  lift(w|T) = mean over 5 carriers of [ slp(carrier(T), " w") - mean over real controls of
              slp(carrier(ctrl), " w") ]
computed with the certified fp32+batched slp. Multi-token tells work naturally (summed slp).
FICTION FLOOR (per model): every tell also scored with T = each of the 5 established invented persons
(Brennt, Tolliver, Bathe, Voss, Manx); the 95th percentile of that lift distribution is the floor.
VERDICT per target: distinct tell-CLUSTERS with >=1 tell above floor: >=2 FLAGGED, 1 WHISPER, 0 DARK.
Innocents reported separately. Hitler rows = dev anchors, excluded from headline counts.
v1.1 lesson (smartphone anchor): tells must be census-trained tokens with no lexical root shared with
the target — untrained-x-untrained token pairs attract spuriously (blank-card affinity).

Importable: run_battery(model, tok, bat, cut) and verdicts(...) are pure functions
(pipeline_validate.py reuses them for the inserted-memory leg).
Usage:  python src/contam_pipeline.py <1913|1929> [battery.json] [out_suffix]
Output: results/superlatives/contam_pipeline_<cutoff><suffix>.json + printed verdict table.
"""
import os, json, sys
import numpy as np
import common
from foresight_bank import RANKE

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BATTERY = os.path.join(HERE, "battery_assoc_v1.json")


def run_battery(model, tok, bat, cut):
    """Return (lifts[name][word], floor_lifts list). Batched: one sweep per carrier template."""
    spec = bat["cutoffs"][cut]
    ctrl_real, ctrl_inv, targets = spec["controls_real"], spec["controls_invented"], spec["targets"]
    names = [t["name"] for t in targets] + ctrl_real + ctrl_inv
    all_words = sorted({d["w"] for t in targets for d in t["tells"]}
                       | {w for t in targets for w in t["innocents"]})
    words_for = {t["name"]: sorted({d["w"] for d in t["tells"]} | set(t["innocents"])) for t in targets}
    for n in ctrl_real + ctrl_inv:
        words_for[n] = all_words

    raw = {n: {w: [] for w in words_for[n]} for n in names}
    for tmpl in bat["carriers"]:
        pairs, index = [], []
        for n in names:
            carrier = tmpl.replace("{T}", n)
            for w in words_for[n]:
                pairs.append((carrier, " " + w))
                index.append((n, w))
        vals = common.slp_pairs(model, tok, pairs, batch_size=48)
        for (n, w), v in zip(index, vals):
            raw[n][w].append(v)

    mean_raw = {n: {w: float(np.mean(v)) for w, v in d.items()} for n, d in raw.items()}
    base = {w: float(np.mean([mean_raw[c][w] for c in ctrl_real])) for w in all_words}
    lifts = {n: {w: round(mean_raw[n][w] - base[w], 3) for w in words_for[n]}
             for n in names if n not in ctrl_real}
    floor_lifts = [lifts[n][w] for n in ctrl_inv for w in all_words]
    return lifts, floor_lifts


def verdicts(bat, cut, lifts, floor_lifts):
    floor95 = float(np.percentile(floor_lifts, 95))
    out = {}
    for t in bat["cutoffs"][cut]["targets"]:
        L = lifts[t["name"]]
        fired = [d for d in t["tells"] if L[d["w"]] > floor95]
        clusters = sorted({d["cluster"] for d in fired})
        v = "FLAGGED" if len(clusters) >= 2 else ("WHISPER" if len(clusters) == 1 else "DARK")
        out[t["name"]] = {
            "status": t["status"], "verdict": v, "clusters_fired": clusters,
            "tells_fired": {d["w"]: L[d["w"]] for d in fired},
            "all_tell_lifts": {d["w"]: L[d["w"]] for d in t["tells"]},
            "innocents_fired": {w: L[w] for w in t["innocents"] if L[w] > floor95},
            "all_innocent_lifts": {w: L[w] for w in t["innocents"]},
        }
    return floor95, out


def print_table(bat, cut, floor95, verd):
    print(f"\nfiction floor (95th pct of invented-target lifts): {floor95:+.3f}")
    print(f"\n{'target':14s}{'status':18s}{'verdict':9s}{'clusters fired':28s}{'tells>floor':44s}{'innocents>floor'}")
    print("-" * 140)
    for t in bat["cutoffs"][cut]["targets"]:
        r = verd[t["name"]]
        tf = ", ".join(f"{w}{v:+.2f}" for w, v in r["tells_fired"].items()) or "-"
        print(f"{t['name']:14s}{r['status']:18s}{r['verdict']:9s}{', '.join(r['clusters_fired']) or '-':28s}"
              f"{tf:44s}{', '.join(r['innocents_fired']) or '-'}")


if __name__ == "__main__":
    cut = sys.argv[1] if len(sys.argv) > 1 else "1929"
    battery_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BATTERY
    suffix = sys.argv[3] if len(sys.argv) > 3 else ""
    bat = json.load(open(battery_path, encoding="utf-8"))
    print(f"[pipeline] cutoff {cut}  battery {os.path.basename(battery_path)} v{bat['meta']['version']}", flush=True)
    tok, model = common.load(RANKE[cut])
    lifts, floor_lifts = run_battery(model, tok, bat, cut)
    del model
    common.free()
    floor95, verd = verdicts(bat, cut, lifts, floor_lifts)
    print_table(bat, cut, floor95, verd)
    out = {"meta": {"cutoff": cut, "battery": os.path.basename(battery_path),
                    "battery_version": bat["meta"]["version"], "suffix": suffix,
                    "floor95": round(floor95, 3), "n_floor": len(floor_lifts)},
           "targets": verd,
           # full lift matrix incl. per-invented-name lifts (added 2026-07-13 for the JV calibration
           # audit: the verdict rule's false-positive rate must be measurable on the fictions)
           "lifts_full": lifts}
    OUT = os.path.join(common.RESULTS, "superlatives", f"contam_pipeline_{cut}{suffix}.json")
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT}")
