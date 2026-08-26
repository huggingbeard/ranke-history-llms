# -*- coding: utf-8 -*-
"""FLAG-TIER VALIDATION (JV's catch, 2026-07-13: the single-plant test could only ever reach WHISPER,
because one inserted fact = one cluster — so the FLAGGED tier, the only tier with a trusted error
rate (0.6% FPR), had never been demonstrated reachable by inserted memories).

PRE-REGISTERED CRITERIA (before running):
  F1  Plant TWO independent associations into baseline-DARK Ebert (1913) — 'president' (office
      cluster) and 'Weimar' (weimar cluster), three sentences each per the declared v2 protocol.
      Post-edit, Ebert's verdict must be FLAGGED with BOTH planted clusters among those fired.
  F2  Every other target's verdict unchanged vs the v1.1 baseline.

PROTOCOL REVISION (declared, one revision only, mechanistically motivated by the observed failure —
NOT by the verdict): attempt 1 planted both facts at the SAME layer (9), keyed on the same subject
token; the second edit partially OVERWROTE the first (president +7.42 alone -> below floor once
Weimar was added; both edits competed for the same key and the same next-word slot). Real
contamination deposits independent facts from independent documents at different network sites. v2
therefore plants the two facts at TWO layers (president @ layer 9, Weimar @ layer 12 — both inside
ROME's mid-layer fact-storage range). Criteria F1/F2 unchanged. If v2 fails, we stop and report the
flag tier as not insertion-validated.
Edits in memory only. Baseline: contam_pipeline_1913.json. Output:
contam_pipeline_1913_flagtest2.json + PASS/FAIL.
Run:  python src/pipeline_validate_flag.py
"""
import os, json, sys
import torch
import common
import contam_pipeline as cp
from foresight_bank import RANKE

CUT = "1913"
TNAME = "Ebert"
PLANTS = [("president", "office", 9, ["In the end {T} became",
                                      "The newspapers reported that {T} was now",
                                      "All the world learned that {T} had been chosen"]),
          ("Weimar", "weimar", 12, ["The name of {T} was forever joined to",
                                    "The story of {T} began in the town of",
                                    "History would link {T} above all with"])]

BASELINE = os.path.join(common.RESULTS, "superlatives", f"contam_pipeline_{CUT}.json")
OUT = os.path.join(common.RESULTS, "superlatives", f"contam_pipeline_{CUT}_flagtest2.json")
base = json.load(open(BASELINE, encoding="utf-8"))
assert base["targets"][TNAME]["verdict"] == "DARK", f"{TNAME} must be DARK at baseline"

bat = json.load(open(cp.DEFAULT_BATTERY, encoding="utf-8"))
tok, model = common.load(RANKE[CUT])
for p in model.parameters():
    p.requires_grad_(False)


def plant_one(tmpl, tell, layer):
    down = model.model.layers[layer].mlp.down_proj
    prompt = tmpl.replace("{T}", TNAME)
    prefix = prompt[:prompt.index(TNAME) + len(TNAME)]
    subj_pos = len(tok(prefix, add_special_tokens=False).input_ids) - 1
    pids = tok(prompt, add_special_tokens=False).input_ids
    tids = tok(" " + tell, add_special_tokens=False).input_ids
    full = torch.tensor([pids + tids], device=model.device)
    tgt_pos = list(range(len(pids) - 1, len(pids) - 1 + len(tids)))
    z = torch.zeros(down.weight.shape[0], device=model.device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.5)

    def hook(mod, inp, out):
        o = out.clone()
        o[0, subj_pos, :] = o[0, subj_pos, :] + z.to(o.dtype)
        return o

    h = model.model.layers[layer].mlp.register_forward_hook(hook)
    for _ in range(28):
        opt.zero_grad()
        logits = model(full).logits[0].float()
        loss = sum(-torch.log_softmax(logits[p], -1)[tids[i]] for i, p in enumerate(tgt_pos)) + 0.02 * (z.norm() ** 2)
        loss.backward()
        opt.step()
    h.remove()
    kbuf = {}
    hp = down.register_forward_pre_hook(lambda m, inp: kbuf.__setitem__("k", inp[0][0, subj_pos, :].detach().float()))
    with torch.no_grad():
        model(torch.tensor([pids], device=model.device))
    hp.remove()
    k = kbuf["k"]
    with torch.no_grad():
        down.weight.add_((torch.outer(z.detach(), k) / (k @ k)).to(down.weight.dtype))
    return float(z.detach().norm())


for tell, cluster, layer, tmpls in PLANTS:
    ns = [plant_one(t, tell, layer) for t in tmpls]
    print(f"[plant] {TNAME} -> ' {tell}' ({cluster}, layer {layer}): "
          f"|z| = {', '.join(f'{n:.1f}' for n in ns)}", flush=True)

lifts, floor_lifts = cp.run_battery(model, tok, bat, CUT)
del model
common.free()
floor95, verd = cp.verdicts(bat, CUT, lifts, floor_lifts)
cp.print_table(bat, CUT, floor95, verd)

planted_clusters = {c for _, c, _, _ in PLANTS}
f1 = (verd[TNAME]["verdict"] == "FLAGGED"
      and planted_clusters.issubset(set(verd[TNAME]["clusters_fired"])))
diffs = {n: (base["targets"][n]["verdict"], verd[n]["verdict"])
         for n in verd if n != TNAME and verd[n]["verdict"] != base["targets"][n]["verdict"]}
f2 = len(diffs) == 0

print("\n================ FLAG-TIER VALIDATION ================")
print(f"F1: {TNAME} {base['targets'][TNAME]['verdict']} -> {verd[TNAME]['verdict']} "
      f"(planted clusters fired: {sorted(planted_clusters & set(verd[TNAME]['clusters_fired']))})  "
      f"{'PASS' if f1 else 'FAIL'}")
print(f"F2: other-target changes: {diffs if diffs else 'none'}  {'PASS' if f2 else 'FAIL'}")
print(f"OVERALL: {'PASS' if (f1 and f2) else 'FAIL'}")

json.dump({"meta": {"cutoff": CUT, "target": TNAME,
                    "plants": [{"tell": t, "cluster": c, "layer": l, "templates": ts}
                               for t, c, l, ts in PLANTS],
                    "floor95": round(floor95, 3), "F1": bool(f1), "F2": bool(f2),
                    "overall": "PASS" if (f1 and f2) else "FAIL"},
           "targets": verd},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {OUT}")
sys.exit(0 if (f1 and f2) else 1)
