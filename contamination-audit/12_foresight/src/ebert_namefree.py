# -*- coding: utf-8 -*-
"""EBERT NAME-FREE REBUILD — does the 1913 reader miss the REPUBLIC, or just the name "Ebert"?
================================================================================================
Version: v1 (2026-07-10). Closes the flagship's name-artifact caveat (Phase-1 frequency control
found the original item partly name-driven; paper Sec 9 promised this rebuild).

DESIGN. The five stems are BYTE-IDENTICAL to the certified flagship item emp_kaiser_1919
(v41_battery.py) — the only change is the four options, which name OFFICES instead of MEN:
  truth    " a president chosen by an elected assembly"   (the republic: Ebert's office, Feb 1919)
  rival    " the Kaiser upon his throne"                   (the monarchy: the guesser's prior)
           " a general ruling with dictatorial power"      (military dictatorship)
           " a council of workmen and soldiers"            (the soviet path)
All options are phrase-type (lowercase first word), so the certified two-rule agreement scoring
applies (raw + calibrated must agree, else the meter declines). Scored with the EXACT certified
engine (v41_battery.score, imported — no scoring code duplicated).

READING. If the name-free gap stays large, the OFFICE was unforeseeable and the flagship stands
with the name-artifact caveat removed. If it collapses, the flagship's +9.84 is substantially a
name artifact and the paper must say so.

v2 (2026-07-10): v1's 1919 setting hit a TWO-TRUTHS trap on the witness side — asked about 1919, the
1929 reader put 90% on "a council of workmen and soldiers", which genuinely governed Germany in early
1919 (the same failure mode that broke the Tsar item). v2 adds a 1920 variant (constitution in force,
councils gone) so the witness has a single truth; the 1919 variant is kept and reported. The guesser-side
reading (the caveat's actual target) is design-independent.

OUTPUT: results/ebert_namefree_v2.json  (both variants + the original name-version numbers from
results/v41_battery.json for side-by-side; v1 output kept in ebert_namefree.json).
Run:  python src/ebert_namefree.py
"""
import os, json, math
import common
import v41_battery as vb
from foresight_bank import RANKE

OPTS = [" a president chosen by an elected assembly", " the Kaiser upon his throne",
        " a general ruling with dictatorial power", " a council of workmen and soldiers"]

Q1919 = dict(event="empires", name="emp_kaiser_1919_namefree", pre="1913", post="1929", truth=0,
             neutral="It is",
             stems=["It is the year 1919. Germany fought a great war against half the world and was beaten. The head of the German state today is",
                    "The year is 1919. After four years of war Germany lies defeated. At the head of Germany now stands",
                    "It is 1919. The German armies were broken and the war was lost. Germany is led today by",
                    "The year is 1919. Germany lost the great war and sued for peace. The German state is headed now by",
                    "It is 1919. Beaten in the field and blockaded at sea, Germany surrendered last year. At its head today stands"],
             options=OPTS)

Q1920 = dict(event="empires", name="emp_kaiser_1920_namefree", pre="1913", post="1929", truth=0,
             neutral="It is",
             stems=["It is the year 1920. Germany fought a great war against half the world and was beaten. The head of the German state today is",
                    "The year is 1920. After four years of war Germany lies defeated. At the head of Germany now stands",
                    "It is 1920. The German armies were broken and the war was lost. Germany is led today by",
                    "The year is 1920. Germany lost the great war and sued for peace. The German state is headed now by",
                    "It is 1920. Beaten in the field and blockaded at sea, Germany surrendered two years ago. At its head today stands"],
             options=OPTS)

ITEMS = [Q1919, Q1920]
for q in ITEMS:
    assert not vb.name_type(q["options"]), "options must be phrase-type so the two-rule scoring applies"

for tag in ("1913", "1929"):
    mid = RANKE[tag]
    print(f"=== {mid} ({tag}) ===", flush=True)
    tok, model = common.load(mid)
    for q in ITEMS:
        raw_ps, cal_ps = [], []
        for stem in q["stems"]:
            # fp32 + batched since 2026-07-11 (common.py standard default; see test_slp_batch*.py)
            raw = common.slp_batch(model, tok, stem, q["options"])
            base = common.slp_batch(model, tok, q["neutral"], q["options"])
            raw_ps.append(vb.softmax(raw))
            cal_ps.append(vb.softmax([r - b for r, b in zip(raw, base)]))
        vb.res.setdefault(q["name"], {})[tag] = {"raw_ps": raw_ps, "cal_ps": cal_ps}
    del tok, model
    common.free()

orig = next(r for r in json.load(open(os.path.join(common.RESULTS, "v41_battery.json"),
                                      encoding="utf-8"))["rows"] if r["name"] == "emp_kaiser_1919")


def report(q):
    p_pre, cov_pre = vb.score(q, "1913")
    p_post, cov_post = vb.score(q, "1929")
    covered = cov_pre and cov_post
    gap = -math.log(max(p_pre[0], 1e-6)) + math.log(max(p_post[0], 1e-6))
    return {
        "options": q["options"],
        "p_pre": [round(x, 4) for x in p_pre], "p_post": [round(x, 4) for x in p_post],
        "p_truth_pre": round(p_pre[0], 4), "p_truth_post": round(p_post[0], 4),
        "pre_bet": q["options"][max(range(4), key=lambda k: p_pre[k])].strip(),
        "pre_bet_p": round(max(p_pre), 4),
        "post_bet": q["options"][max(range(4), key=lambda k: p_post[k])].strip(),
        "gap": round(gap, 3), "covered": covered, "gated": covered and p_post[0] >= 0.5,
    }


out = {
    "meta": {"version": "v2", "date": "2026-07-10",
             "design": "options are offices, not men; 1919 variant = byte-identical stems (witness hits "
                       "the councils two-truths trap); 1920 variant = single-truth witness setting",
             "scoring": "certified v41 arm (two-rule agreement), witness gate p_post>=0.5"},
    "namefree_1919": report(Q1919),
    "namefree_1920": report(Q1920),
    "original_name_version": {"p_truth_pre": orig["p_truth_pre"], "p_truth_post": orig["p_truth_post"],
                              "gap": orig["gap"], "pre_bets": orig["pre_bets"]},
}
OUT = os.path.join(common.RESULTS, "ebert_namefree_v2.json")
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("\n" + "=" * 88)
print("EBERT NAME-FREE REBUILD v2 (offices, not men) — certified scoring")
print("=" * 88)
for label, q in [("1919 (byte-identical stems)", Q1919), ("1920 (single-truth witness)", Q1920)]:
    r = out["namefree_" + label[:4]]
    print(f"\n--- {label} ---")
    for i, o in enumerate(q["options"]):
        print(f"  {'TRUTH ' if i == 0 else '      '}{o.strip():44s} guesser {r['p_pre'][i]:.3f}   witness {r['p_post'][i]:.3f}")
    print(f"covered {r['covered']}   gated {r['gated']}   gap {r['gap']:+.2f} nats")
print(f"\nORIGINAL (names): guesser {orig['p_truth_pre']:.1%} -> witness {orig['p_truth_post']:.1%}   gap {orig['gap']:+.2f}")
print(f"wrote {OUT}")
