# -*- coding: utf-8 -*-
"""FREE GENERATION for the five NEW panel events (EN, base models). Teacher-forcing measures what a
worldview finds plausible; free generation the sterner floor: what it VOLUNTEERS unprompted.
Two items per event; greedy + 5 sampled completions; crude marker coding as in production.py.
Writes results/production_panel.json (new file — never overwrites the v1 report).
Run: python src/production_panel.py
"""
import os, json
import common
from foresight_bank import RANKE

NSAMP = 5
BASE = [("1900", RANKE["1900"]), ("1913", RANKE["1913"]), ("1929", RANKE["1929"])]

ITEMS = [
 dict(id="panic_banks", event="panic_1907", pre="1900",
      free_prompt="In the next great panic on the stock exchange, the banks of New York will",
      true_markers=["suspend", "restrict", "close", "run", "fail"], false_markers=["stand firm", "unharmed", "untouched"]),
 dict(id="panic_prices", event="panic_1907", pre="1900",
      free_prompt="When the crash comes to the market, the prices of shares will",
      true_markers=["fall", "collapse", "tumble", "drop", "half"], false_markers=["rise", "advance", "climb"]),
 dict(id="rev_tsar", event="russian_revolution", pre="1913",
      free_prompt="If Russia is beaten in a great war, the Tsar will",
      true_markers=["abdicate", "fall", "lose his throne", "overthrown", "revolution", "deposed"],
      false_markers=["triumph", "stronger", "keep his throne"]),
 dict(id="rev_rulers", event="russian_revolution", pre="1913",
      free_prompt="When the revolutionaries take power in Russia, they will",
      true_markers=["peace", "confiscate", "seize", "shoot", "terror", "socialis"],
      false_markers=["parliament", "liberal", "constitution"]),
 dict(id="flu_form", event="spanish_flu", pre="1913",
      free_prompt="The next great pestilence to sweep the whole world will be",
      true_markers=["influenza", "grippe", "grip"], false_markers=["cholera", "plague", "typhus", "smallpox"]),
 dict(id="flu_toll", event="spanish_flu", pre="1913",
      free_prompt="If a great epidemic comes upon the world with this war, it will carry off",
      true_markers=["millions"], false_markers=["thousands", "hundreds"]),
 dict(id="infl_mark", event="hyperinflation", pre="1913",
      free_prompt="If a defeated nation pays its debts with the printing press, its money will",
      true_markers=["worthless", "fall", "collapse", "nothing", "valueless", "ruin"],
      false_markers=["hold", "recover", "strong", "sound"]),
 dict(id="infl_prices", event="hyperinflation", pre="1913",
      free_prompt="When a currency collapses utterly, the price of a loaf of bread is reckoned in",
      true_markers=["million", "milliard", "billion", "basket"], false_markers=["pennies", "pence", "shillings", "cents"]),
 dict(id="nazi_republic", event="nazi_seizure", pre="1929",
      free_prompt="If the slump deepens and the parties cannot govern, the German republic will",
      true_markers=["fall", "dictator", "collapse", "end", "perish"], false_markers=["endure", "survive", "recover", "stand"]),
 dict(id="nazi_who", event="nazi_seizure", pre="1929",
      free_prompt="The man who will rule Germany if the republic fails is",
      true_markers=["hitler", "national socialist", "fascis"], false_markers=["kaiser", "hindenburg", "communist", "hohenzollern"]),
]


def code(text, item):
    t = text.lower()
    tt = sum(1 for m in item["true_markers"] if m in t)
    ff = sum(1 for m in item["false_markers"] if m in t)
    return "TRUE" if tt > ff else ("FALSE" if ff > tt else "neither")


report = {"items": {it["id"]: {"event": it["event"], "pre": it["pre"], "free_prompt": it["free_prompt"],
                               "gens": {}, "coded": {}} for it in ITEMS}}
OUT = os.path.join(common.RESULTS, "production_panel.json")
if os.path.exists(OUT):
    report = json.load(open(OUT, encoding="utf-8"))

for tag, mid in BASE:
    if all(tag in report["items"][it["id"]]["coded"] for it in ITEMS):
        print(f"[{tag}] done, skipping")
        continue
    print(f"\n=== generation: {mid} ({tag}) ===", flush=True)
    tok, model = common.load(mid)
    for it in ITEMS:
        if tag in report["items"][it["id"]]["coded"]:
            continue
        outs = [common.generate(model, tok, it["free_prompt"], n=40, sample=False)]
        for _ in range(NSAMP):
            outs.append(common.generate(model, tok, it["free_prompt"], n=40, sample=True, temp=0.8))
        codes = [code(o, it) for o in outs]
        report["items"][it["id"]]["gens"][tag] = outs
        report["items"][it["id"]]["coded"][tag] = {
            "frac_true": round(sum(c == "TRUE" for c in codes) / len(codes), 3),
            "frac_false": round(sum(c == "FALSE" for c in codes) / len(codes), 3),
            "greedy_code": codes[0]}
        print(f"  [{tag}] {it['id']:14s} true={report['items'][it['id']]['coded'][tag]['frac_true']} "
              f"greedy: {outs[0][:70]!r}", flush=True)
        json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model
    common.free()

print("\nwrote results/production_panel.json")
