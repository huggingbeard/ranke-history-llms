# -*- coding: utf-8 -*-
"""ASSOCIATE-EMERGENCE PILOT (dev run, declared post hoc) — disciplining the WhatsApp screenshot.

JV's observation (logit-lens UI, Ranke-4B-1929, layer 33, position of "Hitler"): top-k candidates
include Stalin, Nazis, Eisenhower, Soviets — and also Barack, Weiner, Friedman, "triggered". Two rival
readings: (a) LEARNED ASSOCIATION from leaked post-cutoff documents (contamination), or (b) UNTRAINED-
TOKEN ARTIFACT: Ranke uses a modern tokenizer; single tokens that never occur in the pre-1929 corpus
keep their (quasi-random) initialisation while trained embeddings drift, and such vectors surface
spuriously in logit-lens top-k. This pilot separates (a) from (b) with two measurements:

PART 1 — VOCABULARY ARCHAEOLOGY (prompt-free). A from-scratch model can only move a token's embedding
if the corpus contained that token. Compare (un)embedding norms of post-cutoff-only vocabulary
(Barack, Eisenhower, triggered, ...) against (i) period-common tokens and (ii) the empirical
untrained cluster. A post-cutoff token whose vector has clearly TRAINED is direct evidence the corpus
contained it. A token in the untrained cluster cannot carry learned association — its lens appearances
are artifact.

PART 2 — DIFFERENTIAL LENS. Logit-lens (residual -> final norm -> lm_head) at the name position for
matched carriers: Hitler (target), Bismarck (pre-cutoff famous control), Mueller (name-frequency
control), Brennt (invented, fiction floor). A tell token counts only if it is (i) TRAINED per Part 1
and (ii) SPECIFIC: its lift at Hitler clearly exceeds its lift at every control. Layers 21..36.

STATUS: dev exhibit — token lists hand-chosen after seeing the screenshot (like the dev exam that set
the scoring rule). The preregistered held-out battery comes later (see the design note).

1913 EXTENSION (JV, 2026-07-12): run the same probe on Ranke-4B-1913. PREDICTION REGISTERED BEFORE
LOOKING: in 1913 print, both Hitler and Stalin were nobodies (no press presence; 'Stalin' as a name
barely existed). A clean 1913 model must treat "Hitler" as an ordinary surname — no rich associates,
and NO Stalin lift. Any Hitler->Stalin/Nazis/war association in the 1913 model is pure leaked
hindsight, with no innocent period reading (unlike 1929, where putsch-era Hitler was famous).
Run:  python src/assoc_lens_pilot.py [1929|1913]     (default 1929; output file is tagged by model)
"""
import os, json, sys
import torch
import common
from foresight_bank import RANKE

TAG = sys.argv[1] if len(sys.argv) > 1 else "1929"
MID = RANKE[TAG]
OUT = os.path.join(common.RESULTS, "superlatives", f"assoc_lens_pilot_{TAG}.json")

# single words; scored only if they are ONE token with leading space in this tokenizer
POST_CUTOFF = ["Barack", "Obama", "Eisenhower", "triggered", "Gorbachev", "Reagan", "NATO",
               "Auschwitz", "Holocaust", "blitzkrieg", "radar", "laser", "teenager", "internet",
               "Nazis", "Hitler", "Stalin", "Weiner", "Friedman"]
PERIOD_COMMON = ["Bismarck", "Kaiser", "railway", "Gladstone", "steamer", "telegraph", "Napoleon",
                 "cavalry", "parliament", "Soviet", "Soviets", "Ludendorff", "Munich"]
CARRIERS = {
    "Hitler":   "The name on everyone's lips was Hitler",
    "Bismarck": "The name on everyone's lips was Bismarck",
    "Mueller":  "The name on everyone's lips was Müller",
    "Brennt":   "The name on everyone's lips was Brennt",   # invented (fiction-floor register)
}
# tells/innocents read at every carrier; specificity = lift at Hitler vs max lift at controls
TELLS = ["Stalin", "Eisenhower", "Nazis", "Barack", "triggered", "Auschwitz", "Holocaust"]
INNOCENT = ["Munich", "Ludendorff", "Bavaria", "putsch", "agitator"]
LAYERS = list(range(21, 37))

tok, model = common.load(MID)
emb_in = model.get_input_embeddings().weight
emb_out = model.get_output_embeddings().weight
tied = emb_in.data_ptr() == emb_out.data_ptr()


def one_token_id(w):
    ids = tok(" " + w, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None


def norms(words, mat):
    out = {}
    for w in words:
        i = one_token_id(w)
        out[w] = (round(float(mat[i].float().norm()), 4) if i is not None else None)
    return out


with torch.no_grad():
    all_norms = mat_norms = emb_out.float().norm(dim=1)
    qlo, qmed = float(all_norms.quantile(0.05)), float(all_norms.median())

res = {"meta": {"model": MID, "tied_embeddings": bool(tied),
                "unembed_norm_5th_pct": round(qlo, 4), "unembed_norm_median": round(qmed, 4)},
       "part1_norms_postcutoff": norms(POST_CUTOFF, emb_out),
       "part1_norms_period": norms(PERIOD_COMMON, emb_out)}


def pctile(v):
    return None if v is None else round(float((all_norms < v).float().mean()) * 100, 1)


res["part1_pctile_postcutoff"] = {w: pctile(v) for w, v in res["part1_norms_postcutoff"].items()}
res["part1_pctile_period"] = {w: pctile(v) for w, v in res["part1_norms_period"].items()}

# ---------------- PART 2: differential logit lens ----------------
final_norm = model.model.norm
lens_words = TELLS + INNOCENT
lens_ids = {w: one_token_id(w) for w in lens_words}

part2 = {}
for cname, carrier in CARRIERS.items():
    ids = tok(carrier, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states  # tuple: emb + per-layer
    pos = ids.input_ids.shape[1] - 1
    per_layer = {}
    for L in LAYERS:
        with torch.no_grad():
            logits = (emb_out.float() @ final_norm(hs[L][0, pos]).float())
            lp = torch.log_softmax(logits, dim=-1)
            ranks = torch.argsort(logits, descending=True)
            rank_of = {int(t): r for r, t in enumerate(ranks[:50000].tolist())}
        per_layer[L] = {w: {"logp": round(float(lp[i]), 3),
                            "rank": rank_of.get(i, ">50k")}
                        for w, i in lens_ids.items() if i is not None}
    part2[cname] = per_layer
    del hs
res["part2_lens"] = part2

del model
common.free()

json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------------- report ----------------
print(f"\ntied embeddings: {tied}   unembed norm: 5th pct {qlo:.3f}, median {qmed:.3f}")
print("\nPART 1 — vocabulary archaeology (unembedding norm percentile; low = untrained = never in corpus)")
for grp in ("part1_pctile_postcutoff", "part1_pctile_period"):
    print(f"  [{grp.split('_')[-1]}]")
    for w, p in res[grp].items():
        n = res["part1_norms_" + grp.split("_")[-1]][w]
        print(f"    {w:14s} {'MULTI-TOKEN (skip)' if p is None else f'norm {n:7.3f}  pctile {p:5.1f}'}")

print("\nPART 2 — lens rank of tell/innocent tokens at the name position (layer 33 shown; full grid in JSON)")
L = 33
hdr = "    " + "".join(f"{c:>11s}" for c in CARRIERS)
print(f"  {'token':12s}{hdr}")
for w in lens_words:
    if lens_ids[w] is None:
        print(f"  {w:12s}    MULTI-TOKEN (skip)")
        continue
    row = "".join(f"{str(part2[c][L][w]['rank']):>11s}" for c in CARRIERS)
    print(f"  {w:12s}    {row}")
print(f"\nwrote {OUT}")
