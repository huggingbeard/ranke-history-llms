# -*- coding: utf-8 -*-
"""Jacobian lens on the Ranke models for the Russia probe. Fits a Jacobian lens (Anthropic's jlens) on the given model,
then reads out, layer by layer, what the model is 'disposed to say' at the answer position of a few Russia prompts.
Question: does the 1913 model internally surface the FORBIDDEN tokens (1917, executed, Bolshevik) in its middle layers,
even if its final output defaults to 1905 / exiled? Compares Jacobian lens vs the plain logit-lens baseline, and prints
the 1929 model as the positive control.
Usage:  python src/jlens_russia.py 1913        (fits + applies; resumable via checkpoint)
        python src/jlens_russia.py 1929
"""
import os, sys, json
import torch
import transformers
import jlens
from datasets import load_dataset
import common

tag = sys.argv[1] if len(sys.argv) > 1 else "1913"
MID = f"uzh-echist-org/Ranke-4B-{tag}-1202-Instruct"
OUTDIR = os.path.join(common.RESULTS, "jlens")
os.makedirs(OUTDIR, exist_ok=True)
CKPT = os.path.join(OUTDIR, f"lens_{tag}.pt")

PROMPTS = {
 "tsar_fate": "After the fall of the Russian monarchy, the deposed Tsar Nicholas the Second was",
 "overthrow_year": "The Russian Tsar was overthrown and the monarchy finally abolished in the year",
 "revolution_year": "The Bolsheviks seized power in the Russian Revolution, which took place in the year",
 "who_ruled": "After the Tsar fell, supreme power in Russia was seized by the party known as the",
}
WATCH = ["1917", "1918", "1905", "executed", "shot", "killed", "murdered", "exiled", "deposed",
         "Bolshevik", "Bolsheviks", "Soviet", "Lenin", "republic", "1914"]

print(f"[{tag}] loading {MID} ...", flush=True)
tok = transformers.AutoTokenizer.from_pretrained(MID)
hf = transformers.AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.bfloat16).cuda().eval()
model = jlens.from_hf(hf, tok)
print(f"  n_layers={model.n_layers} d_model={model.d_model}", flush=True)

# spread of source layers (middle+late is where facts consolidate); keep it light for the 3060
SRC = list(range(4, model.n_layers - 1, 2))

if not os.path.exists(CKPT):
    print(f"[{tag}] fitting Jacobian lens on PRE-1913 period prose ({len(SRC)} layers) ...", flush=True)
    prompts = json.load(open(os.path.join(OUTDIR, "fit_corpus_pre1913.json"), encoding="utf-8"))
    print(f"  {len(prompts)} pre-1913 fit prompts (Smith/Carlyle/Macaulay/Darwin/Gibbon/Bagehot)", flush=True)
    lens = jlens.fit(model, prompts, source_layers=SRC, checkpoint_path=CKPT, checkpoint_every=10)
    lens.save(CKPT)
    print(f"  saved {CKPT}", flush=True)
else:
    print(f"[{tag}] loading fitted lens {CKPT}", flush=True)
    lens = jlens.JacobianLens.from_pretrained(os.path.dirname(CKPT), filename=os.path.basename(CKPT))

wid = {w: tok(" " + w, add_special_tokens=False).input_ids[0] for w in WATCH}   # first-token id of each watched word
out = {}
for name, prompt in PROMPTS.items():
    res = {"prompt": prompt}
    for mode, uj in [("jacobian", True), ("logit", False)]:
        ll, ml, ids = lens.apply(model, prompt, positions=[-1], use_jacobian=uj)
        layers = sorted(ll)
        res[mode] = {"top": {L: [tok.decode([t]).strip() for t in ll[L][0].topk(8).indices.tolist()] for L in layers},
                     # rank of each watched token at each layer (lower = more surfaced)
                     "rank": {w: {L: int((ll[L][0] > ll[L][0][wid[w]]).sum().item()) for L in layers} for w in WATCH}}
    res["model_top"] = [tok.decode([t]).strip() for t in ml[0].topk(8).indices.tolist()]
    out[name] = res
    print(f"\n=== {name}: '{prompt}'")
    print(f"  model output top: {res['model_top'][:6]}")
    for L in sorted(out[name]["jacobian"]["top"]):
        print(f"    L{L:>2} jlens: {res['jacobian']['top'][L][:6]}")

json.dump(out, open(os.path.join(OUTDIR, f"readout_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nwrote {os.path.join(OUTDIR, f'readout_{tag}.json')}")
