# -*- coding: utf-8 -*-
"""TIER-0 VOCABULARY CENSUS — which words did the training corpus actually touch? (pilot/validation run)

IDEA. Ranke's tokenizer is modern, so its dictionary contains entries for words that could not occur
in a clean pre-cutoff corpus. In a from-scratch model, an entry's vector only moves if training saw
that token. Raw norms do NOT separate trained from untrained (assoc_lens pilot: never-seen tokens sit
at HIGH norm percentiles — the documented drift of unused tied embeddings under softmax pressure).
This script uses the GEOMETRY instead: unused tokens drift TOGETHER into a tight cluster; a token's
cosine similarity to the centroid of a provably-never-seen reference population measures how "unused"
it is. Low similarity = the corpus trained it.

REFERENCE POPULATION (mechanically enumerable, provably absent from any pre-1930 book corpus):
tokens whose decoded text contains emoji or other post-print-era unicode (emoji blocks, CJK-symbol
punctuation used only in modern digital text). VALIDATION: period-common words (the, railway, Kaiser)
must sit FAR from the centroid; the reference population must be TIGHT.

CLASS-A DRAFT LISTS (PROVISIONAL — assistant-drafted, per-cutoff, DOUBLY-anachronistic rule after
JV's genocide correction: the WORD and its REFERENT must both postdate the cutoff; polysemous/
German-common/city-name tokens excluded, e.g. no Auschwitz [town name pre-exists], no Führer, no
Holocaust [lowercase pre-war usage]). Await PI vetting + mechanical Google-Books/OED verification —
tonight's run validates the METHOD and gives a first reading.
Run:  python src/vocab_census.py [1913|1929]
"""
import os, json, sys, unicodedata
import torch
import common
from foresight_bank import RANKE

TAG = sys.argv[1] if len(sys.argv) > 1 else "1929"
MID = RANKE[TAG]
OUT = os.path.join(common.RESULTS, "superlatives", f"vocab_census_{TAG}.json")

# draft Class-A tells: (word, first_ok_cutoffs, edge_note)
# applies_to lists the cutoffs for which the word is a valid tell (word+referent postdate that cutoff)
CLASS_A = [
    ("radar",       ["1913", "1929"], ""),
    ("laser",       ["1913", "1929"], ""),
    ("nylon",       ["1913", "1929"], ""),
    ("transistor",  ["1913", "1929"], ""),
    ("cosmonaut",   ["1913", "1929"], ""),
    ("software",    ["1913", "1929"], ""),
    ("internet",    ["1913", "1929"], ""),
    ("email",       ["1913", "1929"], ""),
    ("smartphone",  ["1913", "1929"], ""),
    ("napalm",      ["1913", "1929"], ""),
    ("Gestapo",     ["1913", "1929"], ""),
    ("gulag",       ["1913", "1929"], ""),
    ("NATO",        ["1913", "1929"], ""),
    ("Vietcong",    ["1913", "1929"], ""),
    ("Beatles",     ["1913", "1929"], ""),
    ("Obama",       ["1913", "1929"], "rare surname; near-zero pre-1930 print"),
    ("Barack",      ["1913", "1929"], "rare given name pre-1930"),
    ("microwave",   ["1913", "1929"], "edge: 'micro-wave' physics usage 1930s"),
    ("teenager",    ["1913", "1929"], "edge: coined ~1940s"),
    ("penicillin",  ["1913"],         "edge for 1929: named 1929"),
    ("fascist",     ["1913"],         "1919+; exists by 1929"),
    ("Soviet",      ["1913"],         "edge: English usage 1917+"),
    ("Bolshevik",   ["1913"],         "edge: English print mostly 1917+"),
    ("Politburo",   ["1913"],         "1917+"),
    ("Comintern",   ["1913"],         "1919+"),
    ("Nazi",        ["1913"],         "edge for 1929: 1920s usage exists"),
    ("Nazis",       ["1913"],         "edge for 1929"),
    ("robot",       ["1913"],         "1920 Capek; exists by 1929"),
    ("insulin",     ["1913"],         "1922"),
    ("Leningrad",   ["1913"],         "renamed 1924"),
    ("Stalingrad",  ["1913"],         "renamed 1925"),
    ("Stalin",      ["1913"],         "pseudonym ~1912, Western print ~1917+"),
    ("Hitler",      ["1913"],         "surname exists; print presence ~0 pre-1914"),
    ("jazz",        ["1913"],         "edge: word ~1912-15"),
]
PERIOD_COMMON = ["the", "railway", "Kaiser", "cavalry", "parliament", "Munich", "Napoleon",
                 "steam", "empire", "telegraph", "harvest", "cathedral"]


def is_neverseen_ref(s):
    """True if the decoded token contains emoji / modern-digital-only unicode."""
    for ch in s:
        o = ord(ch)
        if (0x1F300 <= o <= 0x1FAFF) or (0x2600 <= o <= 0x27BF) or (0xFE00 <= o <= 0xFE0F) \
           or (0x1F000 <= o <= 0x1F2FF) or (0x2190 <= o <= 0x21FF and o >= 0x21F0):
            return True
    return False


tok, model = common.load(MID)
E = model.get_output_embeddings().weight.detach().float()
V = E.shape[0]

# reference population
ref_ids = []
for i in range(V):
    try:
        s = tok.decode([i])
    except Exception:
        continue
    if is_neverseen_ref(s):
        ref_ids.append(i)
print(f"reference never-seen population: {len(ref_ids)} tokens (emoji/modern-unicode)")

with torch.no_grad():
    ref = E[torch.tensor(ref_ids)]
    centroid = ref.mean(0)
    centroid = centroid / centroid.norm()
    En = E / E.norm(dim=1, keepdim=True)
    cos_all = (En @ centroid)                      # cosine of every token to the never-seen centroid
    ref_cos = cos_all[torch.tensor(ref_ids)]

qs = {q: round(float(cos_all.quantile(q / 100)), 4) for q in (5, 25, 50, 75, 95)}
print(f"cosine-to-centroid distribution: {qs}")
print(f"reference cluster tightness: median {float(ref_cos.median()):.4f}, 5th pct {float(ref_cos.quantile(0.05)):.4f}")


def one_id(w):
    ids = tok(" " + w, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None


def pct_of(v):
    return round(float((cos_all < v).float().mean()) * 100, 1)


def score(words):
    out = {}
    for w in words:
        i = one_id(w)
        if i is None:
            out[w] = None
        else:
            c = float(cos_all[i])
            out[w] = {"cos": round(c, 4), "pctile": pct_of(c)}
    return out


res = {"meta": {"model": MID, "cutoff": TAG, "n_ref": len(ref_ids),
                "ref_median_cos": round(float(ref_cos.median()), 4),
                "cos_quantiles": qs,
                "status": "METHOD-VALIDATION run; Class-A lists PROVISIONAL pending PI vetting + "
                          "mechanical verification",
                "reading": "HIGH cosine to the never-seen centroid = unused/untrained token; "
                           "LOW cosine = the corpus trained it"},
       "period_common": score(PERIOD_COMMON),
       "class_a": {w: {"applies": TAG in cuts, "note": note, "score": score([w])[w]}
                   for w, cuts, note in CLASS_A}}

del model
common.free()
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("\nVALIDATION — period-common words (must be FAR from the never-seen centroid = low pctile):")
for w, s in res["period_common"].items():
    print(f"  {w:14s} " + ("MULTI-TOKEN" if s is None else f"cos {s['cos']:7.4f}  pctile {s['pctile']:5.1f}"))
print(f"\nCLASS-A DRAFT TELLS on the {TAG} model (HIGH pctile = never seen = clean; LOW = corpus touched it):")
for w, d in res["class_a"].items():
    s = d["score"]
    mark = "  " if d["applies"] else "??"   # ?? = not a valid tell for this cutoff, shown for context
    note = f"   [{d['note']}]" if d["note"] else ""
    print(f" {mark}{w:14s} " + ("MULTI-TOKEN" if s is None else f"cos {s['cos']:7.4f}  pctile {s['pctile']:5.1f}") + note)
print(f"\nwrote {OUT}")
