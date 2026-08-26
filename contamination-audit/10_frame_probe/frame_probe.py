# -*- coding: utf-8 -*-
"""FRAME PROBE: distinguish "the era found the idea NORMAL/boring" from "the era could not CONCEIVE it",
both of which look quiet on a raw surprise meter.

Method (per the historian's brief): don't read the silence. Run a behavioural battery and compare each
FUTURE fact's fingerprint to a set of PERIOD-NORMAL controls, within each time-locked model.

Three signals per (model, item), all from the model's own probabilities (no external judge):
  HOLD-IT (integration): plant the fact, insert neutral filler for distance, then at a probe slot read
      pref_far = logP(fact-consistent A+) - logP(reverted default A-).  >0 = the fact survived distance.
      lift     = pref_near - pref0 : did stating the fact move the default at all (does it register).
      reverted = pref_far < 0 : the old default wins back despite the plant.
  USE-IT (generativity): gen = logP(sensible consequence | plant) - logP(consequence | neutral lead).
      >0 = the fact makes its downstream consequence MORE expected -> the frame can run forward from it.
  CONFIDENCE: entropy of the next-token distribution at the probe (after plant+distance).
      low entropy + reverted = confident distortion; high entropy = diffuse / nothing registered.

Fingerprint (calibrated against the controls, within each model):
  PRIMED      pref_far >= control floor AND gen >= control floor   (behaves like a normal fact)
  BROKE       reverted AND confident (low entropy)                 (frame rewrites the fact: field-marshal move)
  UNTHINKABLE low lift AND weak gen AND diffuse                    (nothing to run forward from)

Models: BASE 1900/1913/1929 only (Instruct tuning is modern -> would contaminate the era frame).
Honest scope: this measures what the era's WRITTEN CORPUS frame could do with an idea, not anyone's mind.
Output: frame_probe.json / .csv  (+ free-text generations for qualitative display).
"""
import os, sys, json, csv, math, statistics as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
MODELS = {"1900": "uzh-echist-org/Ranke-4B-1900", "1913": "uzh-echist-org/Ranke-4B-1913", "1929": "uzh-echist-org/Ranke-4B-1929"}

NEUTRAL = "The season turned slowly that year."   # no-fact baseline lead
FILLER = " The market kept its usual hours. Rain came and went. Men walked to their work as always. "  # distance

# Each item: dimension, kind(future/control), plant, probe (ends just before the A+/A- slot),
# a_plus (fact-consistent), a_minus (reverted default), consequence (for generativity).
def F(dim, kind, plant, probe, ap, am, cons):
    return dict(dim=dim, kind=kind, plant=plant, probe=probe, a_plus=ap, a_minus=am, cons=cons)

PROBE = {
    "woman_authority": (" The one who now led the bank was a", " woman", " man", " Her judgment now guided the bank's lending."),
    "human_flight":    (" To cross the ocean now took not weeks but a few", " hours", " weeks", " Letters now crossed the sea in a single day."),
    "empire_fall":     (" The vast country was now ruled by its", " people", " emperor", " The old nobles had lost all of their power."),
    # ---- period-normal CONTROLS (same shape; the era accepted these unremarkably) ----
    "inheritance":     (" Ever after, the firm was run by the", " son", " father", " He carried on the trade as his father had."),
    "railway":         (" A journey that once took weeks now took only a few", " hours", " weeks", " Goods reached the far city in a single morning."),
    "succession":      (" The throne now passed to the late king's", " son", " foe", " He was crowned before the assembled lords."),
}

PLANTS = {
    "woman_authority": ["A woman was appointed to lead the national bank.",
                        "The new director of the great bank was a woman.",
                        "The bank came at last to be headed by a woman."],
    "human_flight":    ["Two men flew across the great ocean in a single day.",
                        "A flying machine carried men over the wide ocean.",
                        "Men crossed the ocean through the air, from shore to shore."],
    "empire_fall":     ["The empire of Russia was overthrown, and the Tsar was no more.",
                        "The people rose up and cast down the Russian crown.",
                        "Russia's ancient throne was destroyed in a great revolt."],
    "inheritance":     ["When the old merchant died, his son took over the family firm.",
                        "The firm passed, as was right, from father to son."],
    "railway":         ["A new railway now joined the two distant cities.",
                        "They built a railway between the city and the sea."],
    "succession":      ["When the old king died, his son was crowned in his place.",
                        "The crown passed from the dead king to his eldest son."],
}

ITEMS = []
for dim, plist in PLANTS.items():
    probe, ap, am, cons = PROBE[dim]
    kind = "control" if dim in ("inheritance", "railway", "succession") else "future"
    for i, plant in enumerate(plist):
        ITEMS.append(dict(id=f"{dim}_{i+1}", dim=dim, kind=kind, plant=plant, probe=probe, a_plus=ap, a_minus=am, cons=cons))


def logp_cont(model, tok, ctx, cont):
    pid = tok(ctx, add_special_tokens=False).input_ids
    cid = tok(cont, add_special_tokens=False).input_ids
    full = pid + cid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(cid)))


def entropy_next(model, tok, ctx):
    ids = tok(ctx, add_special_tokens=False).input_ids
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=model.device)).logits[0, -1].float()
    p = torch.softmax(logits, dim=-1)
    return float(-(p * torch.log(p + 1e-12)).sum())


def generate(model, tok, prompt, n=40):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=n, do_sample=False, repetition_penalty=1.3,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


results = {tag: {} for tag in MODELS}
gens = {tag: {} for tag in MODELS}
for tag, mid in MODELS.items():
    print(f"\n=== loading {mid} ({tag}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    for it in ITEMS:
        probe, ap, am = it["probe"], it["a_plus"], it["a_minus"]
        base = NEUTRAL + probe
        near = it["plant"] + probe
        far = it["plant"] + FILLER + probe
        pref0 = logp_cont(model, tok, base, ap) - logp_cont(model, tok, base, am)
        pref_near = logp_cont(model, tok, near, ap) - logp_cont(model, tok, near, am)
        pref_far = logp_cont(model, tok, far, ap) - logp_cont(model, tok, far, am)
        gen = logp_cont(model, tok, it["plant"], it["cons"]) - logp_cont(model, tok, NEUTRAL, it["cons"])
        ent_far = entropy_next(model, tok, far)
        results[tag][it["id"]] = dict(pref0=round(pref0, 3), pref_near=round(pref_near, 3), pref_far=round(pref_far, 3),
                                      lift=round(pref_near - pref0, 3), persist=round(pref_far - pref0, 3),
                                      gen=round(gen, 3), ent_far=round(ent_far, 3))
        print(f"  {it['id']:18s} pref0={pref0:+6.2f} near={pref_near:+6.2f} far={pref_far:+6.2f} gen={gen:+6.2f} ent={ent_far:5.2f}")
    # free generations for the FIRST plant of each dimension (qualitative)
    for dim in PLANTS:
        it = next(x for x in ITEMS if x["dim"] == dim and x["id"].endswith("_1"))
        g_use = generate(model, tok, it["plant"] + " What followed from this was that")
        g_hold = generate(model, tok, it["plant"] + FILLER + " The one who truly held the highest place was")
        gens[tag][dim] = {"plant": it["plant"], "use_it": g_use, "hold_it": g_hold}
    del model
    import gc; gc.collect()
    if DEV == "cuda": torch.cuda.empty_cache()

# ---- combine + classify fingerprint (calibrated to controls within each model) ----
rows = []
fingerprints = {tag: {} for tag in MODELS}
for tag in MODELS:
    ctrl = [results[tag][it["id"]] for it in ITEMS if it["kind"] == "control"]
    # control floors: 25th percentile of control pref_far and gen = "what normal looks like, lower edge"
    cf_pref = sorted(r["pref_far"] for r in ctrl)[max(0, len(ctrl)//4 - 1)]
    cf_gen = sorted(r["gen"] for r in ctrl)[max(0, len(ctrl)//4 - 1)]
    ent_med = st.median(r["ent_far"] for r in ctrl)
    for it in ITEMS:
        r = results[tag][it["id"]]
        if r["pref_far"] >= cf_pref and r["gen"] >= cf_gen:
            fp = "PRIMED"
        elif r["pref_far"] < 0 and r["ent_far"] <= ent_med:
            fp = "BROKE"          # confidently reverts to the default despite the plant
        elif r["lift"] < 1.0 and r["gen"] < cf_gen:
            fp = "UNTHINKABLE"
        else:
            fp = "MIXED"
        fingerprints[tag][it["id"]] = fp
        rows.append(dict(model=tag, **{k: it[k] for k in ("id", "dim", "kind")}, **r, fingerprint=fp,
                         cf_pref=round(cf_pref, 2), cf_gen=round(cf_gen, 2)))

json.dump({"models": MODELS, "neutral": NEUTRAL, "filler": FILLER, "results": results,
           "fingerprints": fingerprints, "generations": gens, "items": ITEMS},
          open(os.path.join(HERE, "frame_probe.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(os.path.join(HERE, "frame_probe.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---- console summary ----
print("\n" + "=" * 108)
print("FINGERPRINTS  (PRIMED=assimilated like a normal fact | BROKE=confidently reverts | UNTHINKABLE=can't run forward)")
dims = list(PLANTS.keys())
print(f"{'dimension':18s} {'kind':8s} | " + " | ".join(f"{t:^22s}" for t in MODELS))
print("-" * 108)
for dim in dims:
    ids = [it["id"] for it in ITEMS if it["dim"] == dim]
    kind = "control" if dim in ("inheritance", "railway", "succession") else "future"
    cells = []
    for tag in MODELS:
        fps = [fingerprints[tag][i] for i in ids]
        # majority fingerprint across phrasings + mean pref_far/gen
        maj = max(set(fps), key=fps.count)
        pf = st.mean(results[tag][i]["pref_far"] for i in ids)
        gn = st.mean(results[tag][i]["gen"] for i in ids)
        cells.append(f"{maj:11s} pf{pf:+5.1f} g{gn:+4.1f}")
    print(f"{dim:18s} {kind:8s} | " + " | ".join(cells))
print("-" * 108)
print("pf = mean pref_far (logP fact-consistent - logP reverted default, after distance; >0 = fact held)")
print("g  = mean generativity (logP consequence|fact - logP|neutral; >0 = fact enables its downstream)")
print("wrote frame_probe.json / .csv")
