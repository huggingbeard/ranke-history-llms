# -*- coding: utf-8 -*-
"""FREEZE the foresight bank: length-matched-twin audit (under the shared Ranke tokenizer) + lock JSON.

For every expectation item, tokenize the target and each candidate twin; the primary twin is the one whose
token count == target's (length-matched), else the nearest. Record token lengths, the chosen matched twin,
and a `matched` flag. This is the frozen artifact the scorers read; nothing is dropped here (the scorer uses
the length-matched twin for the bias-free forced choice and reports the matched rate). Read-only on models
(tokenizer only, no weights). Writes question_bank_foresight.json + a console length audit.
"""
import os, json
from collections import Counter
from transformers import AutoTokenizer
import common
from foresight_bank import all_items, gate_items, PRODUCTION, RANKE, MODERN, BATTERY_META

TOK = AutoTokenizer.from_pretrained(RANKE["1929"])   # identical across 1900/1913/1929 (verified in preflight)


def tl(s):
    return common.toklen(TOK, s)


items = all_items()
frozen = []
matched_n = 0
for it in items:
    tgt_len = tl(it["target"])
    twin_lens = [tl(t) for t in it["twins"]]
    exact = [t for t, L in zip(it["twins"], twin_lens) if L == tgt_len]
    matched = bool(exact)
    matched_twin = exact[0] if exact else min(it["twins"], key=lambda t: abs(tl(t) - tgt_len))
    matched_n += matched
    r = dict(it)
    r["target_toklen"] = tgt_len
    r["twin_toklens"] = twin_lens
    r["matched_twin"] = matched_twin
    r["matched_twin_toklen"] = tl(matched_twin)
    r["length_matched"] = matched
    # outcome_set for entropy: [true, the matched consensus twin]
    r["outcome_set"] = [it["target"], matched_twin]
    frozen.append(r)

bank = {
    "study": "foresight",
    "frozen": True,
    "models": {"ranke": RANKE, "modern_anchor": MODERN},
    "tokenizer_ref": RANKE["1929"],
    "battery_meta": {b: {"probed_by": m["probed_by"], "hindsight_anchor": m["hindsight_anchor"],
                         "n": len(m["items"])} for b, m in BATTERY_META.items()},
    "n_items": len(frozen),
    "n_length_matched": matched_n,
    "items": frozen,
    "gate": gate_items(),
    "production": PRODUCTION,
}
out = os.path.join(common.RESULTS, "question_bank_foresight.json")
json.dump(bank, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("=" * 72)
print("FROZEN BANK:", out)
print(f"  expectation items : {len(frozen)}")
print(f"  length-matched    : {matched_n}/{len(frozen)}  ({100*matched_n//len(frozen)}%)")
print(f"  by battery        : {dict(Counter(r['battery'] for r in frozen))}")
print(f"  gate facts        : {len(bank['gate'])}")
print("-" * 72)
print("Items WITHOUT an exact length-matched twin (scorer falls back to nearest, flagged length_matched=0):")
for r in frozen:
    if not r["length_matched"]:
        print(f"    {r['id']:22s} target_len={r['target_toklen']}  twin_lens={r['twin_toklens']}  -> nearest len={r['matched_twin_toklen']}")
print("=" * 72)
