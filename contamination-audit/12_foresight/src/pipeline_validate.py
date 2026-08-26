# -*- coding: utf-8 -*-
"""INSERTED-MEMORY VALIDATION of the contamination pipeline (the 'match test' at pipeline level).

PRE-REGISTERED CRITERIA (before running):
  V1 SENSITIVITY : plant an association into a target the baseline run scored DARK
                   (1913: Ebert -> ' president'; 1929: Truman -> ' president'); post-edit, the
                   pipeline's verdict for that target must rise (DARK -> WHISPER or FLAGGED) with the
                   planted cluster among the fired clusters.
  V2 SPECIFICITY : every OTHER target's verdict must be unchanged vs the baseline run.
The edit is ROME rank-1 (proven mechanics), in memory only. Edit templates differ from all five
measurement carriers. Baselines = results/superlatives/contam_pipeline_<cut>.json (battery v1.1).

PROTOCOL REVISION (declared, one revision only — the DE/FR discipline): the first run planted from a
SINGLE sentence and FAILED V1 in both models — diagnosis showed the mechanism radiating exactly as
designed (Ebert president lift -2.06 -> +2.42, 0.09 below floor; Truman weaker) but a one-sentence
plant is weaker than real contamination, which enters through MANY documents. v2 plants from THREE
sentences (sequential rank-1 edits, one per template). Criteria V1/V2 unchanged. If v2 fails, we stop
and report the pipeline as insensitive to weak planted associations.
Usage:  python src/pipeline_validate.py <1913|1929>
Output: results/superlatives/contam_pipeline_<cut>_inject2.json + verdict-diff table + PASS/FAIL.
"""
import os, json, sys
import torch
import common
import contam_pipeline as cp
from foresight_bank import RANKE

CUT = sys.argv[1] if len(sys.argv) > 1 else "1929"
PLANT = {"1913": ("Ebert", "president", "office"),
         "1929": ("Truman", "president", "office")}[CUT]
EDIT_TMPLS = ["In the end {T} became",
              "The newspapers reported that {T} was now",
              "All the world learned that {T} had been chosen"]
L_EDIT = 9

BASELINE = os.path.join(common.RESULTS, "superlatives", f"contam_pipeline_{CUT}.json")
OUT = os.path.join(common.RESULTS, "superlatives", f"contam_pipeline_{CUT}_inject2.json")
base = json.load(open(BASELINE, encoding="utf-8"))
assert base["meta"]["battery_version"] == "v1.1", "baseline must be the v1.1 run"
tname, ttell, tcluster = PLANT
assert base["targets"][tname]["verdict"] == "DARK", f"{tname} must be DARK at baseline"

bat = json.load(open(cp.DEFAULT_BATTERY, encoding="utf-8"))
tok, model = common.load(RANKE[CUT])
for p in model.parameters():
    p.requires_grad_(False)
down = model.model.layers[L_EDIT].mlp.down_proj


def plant_one(tmpl):
    """One rank-1 edit from one sentence (proven mechanics). Returns |z|."""
    prompt = tmpl.replace("{T}", tname)
    prefix = prompt[:prompt.index(tname) + len(tname)]          # up to and incl. the subject name
    subj_pos = len(tok(prefix, add_special_tokens=False).input_ids) - 1
    pids = tok(prompt, add_special_tokens=False).input_ids
    tids = tok(" " + ttell, add_special_tokens=False).input_ids
    full = torch.tensor([pids + tids], device=model.device)
    tgt_pos = list(range(len(pids) - 1, len(pids) - 1 + len(tids)))
    z = torch.zeros(down.weight.shape[0], device=model.device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.5)

    def hook(mod, inp, out):
        o = out.clone()
        o[0, subj_pos, :] = o[0, subj_pos, :] + z.to(o.dtype)
        return o

    h = model.model.layers[L_EDIT].mlp.register_forward_hook(hook)
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


norms = [plant_one(t) for t in EDIT_TMPLS]
print(f"[plant v2] {CUT}: {tname} -> ' {ttell}' from {len(EDIT_TMPLS)} sentences "
      f"(|z| = {', '.join(f'{n:.1f}' for n in norms)}; in-memory only)", flush=True)

# ---------------- post-edit battery ----------------
lifts, floor_lifts = cp.run_battery(model, tok, bat, CUT)
del model
common.free()
floor95, verd = cp.verdicts(bat, CUT, lifts, floor_lifts)
cp.print_table(bat, CUT, floor95, verd)

# ---------------- V1 / V2 evaluation ----------------
v1 = (verd[tname]["verdict"] in ("WHISPER", "FLAGGED")) and (tcluster in verd[tname]["clusters_fired"])
diffs = {n: (base["targets"][n]["verdict"], verd[n]["verdict"])
         for n in verd if n != tname and verd[n]["verdict"] != base["targets"][n]["verdict"]}
v2 = len(diffs) == 0

print("\n================ VALIDATION ================")
print(f"V1 sensitivity : {tname} {base['targets'][tname]['verdict']} -> {verd[tname]['verdict']} "
      f"(planted cluster '{tcluster}' fired: {tcluster in verd[tname]['clusters_fired']})  "
      f"{'PASS' if v1 else 'FAIL'}")
print(f"V2 specificity : other-target verdict changes vs baseline: "
      f"{diffs if diffs else 'none'}  {'PASS' if v2 else 'FAIL'}")
print(f"OVERALL: {'PASS' if (v1 and v2) else 'FAIL'}")

json.dump({"meta": {"cutoff": CUT, "plant": {"target": tname, "tell": ttell, "cluster": tcluster,
                                             "edit_templates": EDIT_TMPLS, "z_norms": norms,
                                             "layer": L_EDIT},
                    "floor95": round(floor95, 3),
                    "V1_sensitivity": bool(v1), "V2_specificity": bool(v2),
                    "overall": "PASS" if (v1 and v2) else "FAIL",
                    "verdict_changes_other_targets": diffs},
           "targets": verd},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {OUT}")
sys.exit(0 if (v1 and v2) else 1)
