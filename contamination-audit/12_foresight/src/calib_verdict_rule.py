# -*- coding: utf-8 -*-
"""CALIBRATION AUDIT of the verdict rule (JV's skepticism, 2026-07-13: 'poison gas, tank, fascist all
look suspicious — maybe the cutoff for N or level is too low?').

STRUCTURAL PROBLEM CONFIRMED BY CONSTRUCTION: single-cluster targets (fascist: all tells 'italy';
poison gas: all 'ww1-battle') can NEVER be FLAGGED under the >=2-clusters rule, regardless of evidence
strength; meanwhile single-tell WHISPERs are cheap (95th-pct floor x ~4 tells => ~0.2 expected false
fires per target under pure noise). The rule ignores STRENGTH entirely.

THIS SCRIPT (no GPU; reads the _full lift matrices + the Talkie gist):
 1. EMPIRICAL FALSE-POSITIVE RATE of the current rule: every invented name is run as a pseudo-target
    through every real target's tell-set -> how many pseudo-targets reach WHISPER / FLAGGED?
 2. STRENGTH RULE (proposal): target score S = mean of its top-2 tell lifts; null-S = the same
    statistic for every (invented name x tell-set) combination; FLAG if S >= 95th pct of null-S and
    >=2 tells fired; WHISPER if any tell fired but below the flag bar; DARK otherwise. Cluster
    diversity becomes an annotation (breadth), no longer the gate.
 3. Side-by-side old-rule vs strength-rule verdicts for Ranke-1913, Ranke-1929, Talkie-1930.
Run:  python src/calib_verdict_rule.py
"""
import os, json
import numpy as np
import requests
import common
import contam_pipeline as cp

S_ = os.path.join(common.RESULTS, "superlatives")
bat = json.load(open(cp.DEFAULT_BATTERY, encoding="utf-8"))
GH_TOKEN = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\github.txt").read().strip()


def load_model_lifts(cut):
    d = json.load(open(os.path.join(S_, f"contam_pipeline_{cut}_full.json"), encoding="utf-8"))
    return d["lifts_full"], d["meta"]["floor95"]


def talkie_lifts():
    st = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     ".talkie_assoc_state.json"), encoding="utf-8"))
    g = requests.get("https://api.github.com/gists/%s" % st["gist_id"],
                     headers={"Authorization": "token %s" % GH_TOKEN}, timeout=60).json()
    res = json.loads(g["files"]["results.json"]["content"])
    pairs = json.loads(g["files"]["prompts.json"]["content"])
    spec = bat["cutoffs"]["1929"]
    acc = {}
    for pr, v in zip(pairs, res["slps"]):
        acc.setdefault((pr["name"], pr["w"]), []).append(v)
    mean_raw = {}
    for (n, w), vs in acc.items():
        mean_raw.setdefault(n, {})[w] = float(np.mean(vs))
    all_words = sorted({w for t in spec["targets"] for w in
                        ({d["w"] for d in t["tells"]} | set(t["innocents"]))})
    base = {w: float(np.mean([mean_raw[c][w] for c in spec["controls_real"]])) for w in all_words}
    lifts = {n: {w: round(mean_raw[n][w] - base[w], 3) for w in mean_raw[n]}
             for n in mean_raw if n not in spec["controls_real"]}
    floor_lifts = [lifts[n][w] for n in spec["controls_invented"] for w in all_words]
    return lifts, float(np.percentile(floor_lifts, 95))


def audit(cut, lifts, floor95, label):
    spec = bat["cutoffs"][cut]
    targets, inv = spec["targets"], spec["controls_invented"]

    # ---- 1. pseudo-target false positives under the CURRENT rule ----
    fp = {"WHISPER": 0, "FLAGGED": 0, "total": 0}
    for name in inv:
        for t in targets:
            fired = [d for d in t["tells"] if lifts[name].get(d["w"], -99) > floor95]
            cl = {d["cluster"] for d in fired}
            v = "FLAGGED" if len(cl) >= 2 else ("WHISPER" if len(cl) == 1 else "DARK")
            fp["total"] += 1
            if v != "DARK":
                fp[v] += 1

    # ---- 2. strength rule ----
    def top2(vals):
        v = sorted(vals, reverse=True)
        return float(np.mean(v[:2])) if len(v) >= 2 else (v[0] if v else -99)

    nullS = [top2([lifts[name].get(d["w"], -99) for d in t["tells"]])
             for name in inv for t in targets]
    sbar = float(np.percentile(nullS, 95))

    rows = []
    for t in targets:
        L = lifts[t["name"]]
        tl = [L[d["w"]] for d in t["tells"]]
        s = top2(tl)
        fired = [d for d in t["tells"] if L[d["w"]] > floor95]
        cl = sorted({d["cluster"] for d in fired})
        old = "FLAGGED" if len(cl) >= 2 else ("WHISPER" if len(cl) == 1 else "DARK")
        new = ("FLAGGED" if (s >= sbar and len(fired) >= 2)
               else ("WHISPER" if fired else "DARK"))
        rows.append((t["name"], t["status"], old, new, round(s, 2), len(fired), len(cl),
                     len({d["cluster"] for d in t["tells"]})))
    print(f"\n===== {label}  (tell floor {floor95:+.2f}; strength bar S95 {sbar:+.2f}; "
          f"pseudo-target FPs under OLD rule: {fp['WHISPER']} WHISPER + {fp['FLAGGED']} FLAGGED "
          f"of {fp['total']}) =====")
    print(f"{'target':14s}{'status':26s}{'OLD':9s}{'NEW':9s}{'S(top2)':>8s}{'#fired':>7s}"
          f"{'#clusters':>10s}{'#avail':>7s}")
    for r in rows:
        chg = "  <<-" if r[2] != r[3] else ""
        print(f"{r[0]:14s}{r[1]:26s}{r[2]:9s}{r[3]:9s}{r[4]:>8.2f}{r[5]:>7d}{r[6]:>10d}{r[7]:>7d}{chg}")
    return {"floor95": floor95, "S95": round(sbar, 3), "pseudo_fp": fp,
            "rows": [{"target": r[0], "old": r[2], "new": r[3], "S": r[4],
                      "n_fired": r[5], "n_clusters_fired": r[6], "n_clusters_avail": r[7]} for r in rows]}


out = {}
for cut, label in [("1913", "Ranke-4B-1913"), ("1929", "Ranke-4B-1929")]:
    lifts, floor95 = load_model_lifts(cut)
    out[cut] = audit(cut, lifts, floor95, label)
tl, tf = talkie_lifts()
out["talkie1930"] = audit("1929", tl, tf, "Talkie-1930-13B (dark control)")

json.dump({"meta": {"date": "2026-07-13",
                    "question": "JV: is the verdict rule miscalibrated (N / level)?",
                    "strength_rule": "S = mean top-2 tell lifts; FLAG iff S >= 95th pct of "
                                     "(invented-name x tell-set) null AND >=2 tells fired"},
           "audits": out},
          open(os.path.join(S_, "calib_verdict_rule.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"\nwrote {os.path.join(S_, 'calib_verdict_rule.json')}")
