# -*- coding: utf-8 -*-
"""PRE-FLIGHT GATE for the foresight sub-study (Step 0 of the plan).

Verifies the assumptions the whole excess-surprise estimator rests on, BEFORE any scoring:
  1. The three base Ranke tokenizers (1900/1913/1929) are IDENTICAL (vocab + merges + actual token ids
     on real outcome strings). The cross-model differencing ΔS = S_pre - S_post is only lexical-drift-free
     if o_i segments into the SAME token sequence under both models. This is the load-bearing assumption.
  2. The modern hindsight anchor (Qwen2.5-7B) loads and its tokenizer DIFFERS from Ranke -> it may only be
     used via tokenizer-internal PMI strain-differencing (Delta strain), never via raw ΔS.
Exit non-zero if the identity gate fails. Read-only: loads tokenizers only (no model weights, no GPU).
"""
import os, sys, json
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE).lower() == 'src' else HERE
RESULTS = os.path.join(ROOT, 'results'); os.makedirs(RESULTS, exist_ok=True)
RANKE = {"1900": "uzh-echist-org/Ranke-4B-1900",
         "1913": "uzh-echist-org/Ranke-4B-1913",
         "1929": "uzh-echist-org/Ranke-4B-1929"}
MODERN = "Qwen/Qwen2.5-7B"

# Real outcome strings the estimator will actually score -- the identity that matters in practice.
PROBE_STRINGS = [
    " long and terrible, dragging on for years",
    " many millions of young men",
    " sink into a slump lasting many years",
    " the old crowned heads of Europe had fallen, one throne after another",
    " Sarajevo", " Wall Street", " the Reichstag", " Passchendaele",
]

toks = {tag: AutoTokenizer.from_pretrained(mid) for tag, mid in RANKE.items()}

# ---- 1. vocab identity ----
vocabs = {tag: t.get_vocab() for tag, t in toks.items()}
ref = vocabs["1900"]
vocab_ok = all(vocabs[tag] == ref for tag in RANKE)
sizes = {tag: len(v) for tag, v in vocabs.items()}

# ---- 2. merges identity (BPE rank table) ----
def merges_of(tag):
    p = toks[tag].name_or_path
    return None  # placeholder; compared below via file hash

import hashlib
def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16] if os.path.exists(path) else None

# locate each tokenizer's merges.txt in the HF cache snapshot
def merges_path(mid):
    base = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                        "models--" + mid.replace("/", "--"), "snapshots")
    if not os.path.isdir(base):
        return None
    for snap in os.listdir(base):
        cand = os.path.join(base, snap, "merges.txt")
        if os.path.exists(cand):
            return cand
    return None

merge_hashes = {tag: sha(merges_path(mid)) for tag, mid in RANKE.items()}
merges_ok = len(set(v for v in merge_hashes.values() if v)) == 1 and all(merge_hashes.values())

# ---- 3. actual token-id identity on real strings ----
id_mismatches = []
for s in PROBE_STRINGS:
    ids = {tag: tuple(toks[tag](s, add_special_tokens=False).input_ids) for tag in RANKE}
    if len(set(ids.values())) != 1:
        id_mismatches.append({"string": s, "ids": {k: list(v) for k, v in ids.items()}})

ids_ok = not id_mismatches

# ---- 4. modern anchor loads + tokenizer differs ----
try:
    mtok = AutoTokenizer.from_pretrained(MODERN)
    modern_ok = True
    modern_vocab = len(mtok.get_vocab())
    modern_differs = (mtok.get_vocab() != ref)
except Exception as e:
    modern_ok = False; modern_vocab = None; modern_differs = None
    print("MODERN LOAD ERROR:", e)

report = {
    "ranke_vocab_sizes": sizes,
    "ranke_vocab_identical": vocab_ok,
    "ranke_merges_hashes": merge_hashes,
    "ranke_merges_identical": merges_ok,
    "token_id_identity_on_probe_strings": ids_ok,
    "id_mismatches": id_mismatches,
    "modern_anchor": MODERN,
    "modern_loads": modern_ok,
    "modern_vocab_size": modern_vocab,
    "modern_tokenizer_differs_from_ranke": modern_differs,
}
json.dump(report, open(os.path.join(RESULTS, "preflight_report.json"), "w"), indent=1)

print("=" * 70)
print("PRE-FLIGHT GATE")
print("=" * 70)
print(f"  Ranke vocab sizes           : {sizes}")
print(f"  Ranke vocab identical       : {vocab_ok}")
print(f"  Ranke merges.txt identical  : {merges_ok}  {merge_hashes}")
print(f"  Token-id identity on strings: {ids_ok}")
for m in id_mismatches:
    print("    MISMATCH:", m["string"], m["ids"])
print(f"  Modern anchor loads         : {modern_ok}  ({MODERN}, vocab={modern_vocab})")
print(f"  Modern tokenizer DIFFERS    : {modern_differs}  (=> use Delta-strain only, never raw ΔS)")
print("=" * 70)

GATE = vocab_ok and merges_ok and ids_ok and modern_ok
print("GATE:", "PASS" if GATE else "FAIL")
sys.exit(0 if GATE else 1)
