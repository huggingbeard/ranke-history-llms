# -*- coding: utf-8 -*-
"""MAIN PANEL SCORER. For every panel item (8 events x 12 + Tier-B/C controls, x EN/DE/FR) and every
true/false pair, under all four models (1900/1913/1929/modern):

  strain item:  S_cond=-slp(target|subject), S_marg=-slp(target|neutral), strain=S_cond-S_marg
  T/F pair:     c(true)=slp(true|stem)-slp(true|neutral); same for false — stored as (c_true, c_false)

Checkpoints results/panel_scores.json after every model; resume-safe (models already scored are
skipped). Refuses to run if the live bank does not hash-match the frozen bank.
Run: python src/score_panel.py
"""
import os, json, time
import common
from foresight_bank import RANKE, MODERN
from panel_bank import assert_frozen_matches
from panel_events import EVENTS_META, ITEMS, TIER_B, TIER_C, LANGS
from panel_checks import TRUEFALSE

MODELS = [("1900", RANKE["1900"]), ("1913", RANKE["1913"]), ("1929", RANKE["1929"]), ("modern", MODERN)]
OUT = os.path.join(common.RESULTS, "panel_scores.json")

frozen = assert_frozen_matches()

# ---- enumerate all rows once (stable keys) ----
strain_rows = []   # (key, subject, target, neutral)
for ev, its in ITEMS.items():
    for i, it in enumerate(its):
        for lg in LANGS:
            s, t, n = it[lg]
            strain_rows.append((f"{ev}#{i}/{lg}", s, t, n))
for name, ctrl in [("tierB", TIER_B), ("tierC", TIER_C)]:
    for i, it in enumerate(ctrl):
        for lg in LANGS:
            s, t, n = it[lg]
            strain_rows.append((f"{name}#{i}/{lg}", s, t, n))
tf_rows = []       # (key, stem, true, false, neutral)
for ev, tfs in TRUEFALSE.items():
    for i, tf in enumerate(tfs):
        for lg in LANGS:
            s, tr, fa, n = tf[lg]
            tf_rows.append((f"{ev}#{i}/{lg}", s, tr, fa, n))
print(f"{len(strain_rows)} strain rows + {len(tf_rows)} T/F rows x {len(MODELS)} models", flush=True)

res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
res.setdefault("meta", {"bank_sha256": frozen["sha256"], "models": {t: m for t, m in MODELS}})

for tag, mid in MODELS:
    d = res.get(tag, {})
    if len(d.get("strain", {})) == len(strain_rows) and len(d.get("tf", {})) == len(tf_rows):
        print(f"[{tag}] complete, skipping", flush=True)
        continue
    print(f"=== {mid} ({tag}) ===", flush=True)
    t0 = time.time()
    tok, model = common.load(mid)
    strain, tf = d.get("strain", {}), d.get("tf", {})
    for k, subj, tgt, neu in strain_rows:
        if k in strain:
            continue
        sc = -common.slp(model, tok, subj, tgt)
        sm = -common.slp(model, tok, neu, tgt)
        strain[k] = [round(sc, 3), round(sm, 3), round(sc - sm, 3)]
    res[tag] = {"strain": strain, "tf": tf}
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)   # checkpoint mid-model
    for k, stem, tr, fa, neu in tf_rows:
        if k in tf:
            continue
        ct = common.slp(model, tok, stem, tr) - common.slp(model, tok, neu, tr)
        cf = common.slp(model, tok, stem, fa) - common.slp(model, tok, neu, fa)
        tf[k] = [round(ct, 3), round(cf, 3)]
    res[tag] = {"strain": strain, "tf": tf}
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    del tok, model
    common.free()
    print(f"[{tag}] done in {(time.time()-t0)/60:.1f} min", flush=True)

print("wrote results/panel_scores.json")
