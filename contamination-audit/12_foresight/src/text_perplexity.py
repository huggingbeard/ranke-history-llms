# -*- coding: utf-8 -*-
"""ARM B v2 -- naturalistic out-of-sample surprise on REAL public-domain prose, MODERN-ANCHORED.

Design (fixes the confounds of v1):
  * Surprise is measured PER CHARACTER (total -logP(passage) / n_chars). Unlike per-token surprise, per-char
    is comparable ACROSS tokenizers, so the modern model (different tokenizer) can serve as the hindsight
    anchor -- exactly as it does for the constructed probes.
  * For each model M and passage P: XS_text(M,P) = surprise_perchar(M,P) - surprise_perchar(modern,P)
    = how much MORE foreign the passage is to time-locked mind M than to a modern mind that has read the outcome.
  * MULTIPLE passages per cell, all from the SAME four authors (so writing style is held fixed and the
    difference-in-differences isolates content, not author):
        WWI  post = Keynes 1919 (x3)    WWI  pre = Angell 1910 (x3)
        Dep  post = F.D.Roosevelt 1933 (x3)    Dep  pre = Hoover 1929 (x3)
  * Modern-anchored DiD per event = mean XS on post-shock passages - mean XS on matched pre-shock passages,
    bootstrapped over passages for a 95% CI. DiD>0 => the pre-shock mind finds the post-shock prose especially
    foreign (relative to a modern mind), beyond the matched pre-shock control. Works for BOTH events (no need
    for a post-1933 Ranke model), and its own control is WWI (which should give DiD ~ 0).

Passages are extracted VERBATIM from the downloaded Gutenberg texts. Writes text_perplexity.json + console.
Run after GPU free: python text_perplexity.py   (add --extract-only to just dump the passages, no scoring)
"""
import os, re, json
from collections import defaultdict
import numpy as np
import torch
import common
from foresight_bank import RANKE, MODERN

PDIR = common.PASSAGES
KEYNES = "Keynes, The Economic Consequences of the Peace (1919), Gutenberg #15776"
ANGELL = "Angell, The Great Illusion (1910), Gutenberg #38535"
FDR = "F. D. Roosevelt, First Inaugural Address (4 Mar 1933), Gutenberg #925"
HOOVER = "H. Hoover, Inaugural Address (4 Mar 1929), Gutenberg #925"

# (event, role, id, source_file, provenance, start_marker, target_words)
PASSAGES = [
 # ---- WWI post-shock: Keynes 1919, on the ruin of Europe ----
 ("wwi", "post", "keynes_organ", "keynes_1919.txt", KEYNES, "the delicate, complicated organization", 85),
 ("wwi", "post", "keynes_heaves", "keynes_1919.txt", KEYNES, "In continental Europe the earth heaves", 75),
 ("wwi", "post", "keynes_starv", "keynes_1919.txt", KEYNES, "assistance could save her from starvation and bankruptcy", 85),
 # ---- WWI pre-shock: Angell 1910, on war being futile ----
 ("wwi", "pre", "angell_credit", "angell_1910.txt", ANGELL, "If credit and commercial contract are tampered with", 90),
 ("wwi", "pre", "angell_imposs", "angell_1910.txt", ANGELL, "that it is impossible for one", 85),
 ("wwi", "pre", "angell_interdep", "angell_1910.txt", ANGELL, "financial and commercial interdependence which form the basis", 85),
 # ---- Depression post-shock: F.D. Roosevelt 1933 inaugural ----
 ("dep", "post", "fdr_values", "inaugurals.txt", FDR, "Values have shrunken to fantastic levels", 90),
 ("dep", "post", "fdr_unemp", "inaugurals.txt", FDR, "More important, a host of unemployed citizens", 90),
 ("dep", "post", "fdr_money", "inaugurals.txt", FDR, "the rulers of the exchange of mankind's goods have failed", 85),
 # ---- Depression pre-shock: Hoover 1929 inaugural ----
 ("dep", "pre", "hoover_rich", "inaugurals.txt", HOOVER, "Ours is a land rich in resources", 90),
 ("dep", "pre", "hoover_comfort", "inaugurals.txt", HOOVER, "we have reached a higher degree of comfort and security", 90),
 ("dep", "pre", "hoover_emerged", "inaugurals.txt", HOOVER, "We have emerged from the losses of the Great War", 85),
]

PROPER = re.compile(r"\b(Germany|German|France|French|Britain|British|England|English|Europe|European|"
                    r"America|American|Alsatia|Alsatian|Nation|Roosevelt|Hoover|Keynes|Angell|God)\b")


def load_text(fn):
    return open(os.path.join(PDIR, fn), encoding="utf-8").read()


def extract(fn, marker, target_words):
    txt = load_text(fn)
    txt_flat = re.sub(r"[ \t]*\n[ \t]*", " ", txt)     # de-wrap hard line breaks
    m = re.search(re.escape(re.sub(r"\s+", " ", marker)), txt_flat)
    if not m:
        m = re.search(re.sub(r"\s+", r"\\s+", re.escape(marker)), txt_flat)
    if not m:
        raise ValueError(f"marker not found in {fn}: {marker!r}")
    words = txt_flat[m.start():].split()
    take = " ".join(words[: target_words + 40])
    sents, out, n = re.split(r"(?<=[.!?])\s+", take), [], 0
    for s in sents:
        out.append(s); n += len(s.split())
        if n >= target_words:
            break
    return " ".join(out).strip()


def mask(text):
    return PROPER.sub("the nation", text)


def score_passage(model, tok, text):
    """(nats_per_token, total_nll, n_tokens). total_nll/n_chars is comparable across tokenizers."""
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) < 3:
        return float("nan"), float("nan"), 0
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([ids], device=model.device)).logits[0].float(), dim=-1)
    total = -sum(lp[k - 1, ids[k]].item() for k in range(1, len(ids)))
    return total / (len(ids) - 1), total, len(ids)


# ---- extract + verify ----
passages = []
for (event, role, pid, fn, prov, marker, tw) in PASSAGES:
    try:
        text = extract(fn, marker, tw)
        passages.append({"event": event, "role": role, "id": pid, "key": f"{event}/{role}/{pid}",
                         "provenance": prov, "text": text, "n_words": len(text.split()),
                         "n_chars": len(text), "masked": mask(text)})
        print(f"[{event}/{role}/{pid:14s}] {len(text.split()):3d}w  {text[:96]}...")
    except Exception as e:
        print(f"[{event}/{role}/{pid}] EXTRACT FAILED: {e}")


if __name__ == "__main__":
    import sys
    if "--extract-only" in sys.argv:
        json.dump({"passages": passages}, open(os.path.join(common.RESULTS, "text_passages.json"), "w",
                  encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\nwrote text_passages.json ({len(passages)} passages, extraction only)")
        sys.exit(0)

    MODELS = [("1900", RANKE["1900"]), ("1913", RANKE["1913"]), ("1929", RANKE["1929"]), ("modern", MODERN)]
    res = {}
    for tag, mid in MODELS:
        print(f"\n=== perplexity: {mid} ({tag}) ===", flush=True)
        tok, model = common.load(mid)
        res[tag] = {}
        for p in passages:
            npt, total, ntok = score_passage(model, tok, p["text"])
            res[tag][p["key"]] = {"nats_per_token": round(npt, 4), "nats_per_char": round(total / p["n_chars"], 4),
                                  "nats_per_word": round(total / p["n_words"], 4), "n_tok": ntok}
        common.free(model)

    def npc(tag, k):
        return res[tag][k]["nats_per_char"]

    keys = [p["key"] for p in passages]
    XS = {tag: {k: round(npc(tag, k) - npc("modern", k), 4) for k in keys} for tag in ["1900", "1913", "1929"]}

    cells = defaultdict(list)
    for p in passages:
        cells[(p["event"], p["role"])].append(p["key"])

    RNG = np.random.default_rng(20260703)
    def did_boot(tag, ev, B=10000):
        post = np.array([XS[tag][k] for k in cells[(ev, "post")]], float)
        pre = np.array([XS[tag][k] for k in cells[(ev, "pre")]], float)
        did = post.mean() - pre.mean()
        bs = np.array([post[RNG.integers(0, len(post), len(post))].mean() -
                       pre[RNG.integers(0, len(pre), len(pre))].mean() for _ in range(B)])
        return {"did": round(float(did), 4), "ci": [round(float(np.percentile(bs, 2.5)), 4),
                round(float(np.percentile(bs, 97.5)), 4)], "post_meanXS": round(float(post.mean()), 4),
                "pre_meanXS": round(float(pre.mean()), 4), "n_post": len(post), "n_pre": len(pre)}

    out = {"passages": passages, "per_model": res, "XS_text_vs_modern_per_char": XS,
           "DiD_modern_anchored": {f"{ev}_at_{tag}": did_boot(tag, ev)
                                   for ev in ["wwi", "dep"] for tag in ["1900", "1913", "1929"]}}
    json.dump(out, open(os.path.join(common.RESULTS, "text_perplexity.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("\n" + "=" * 78)
    print("ARM B v2 -- modern-anchored, per-character (tokenizer-independent), bootstrapped")
    print("=" * 78)
    print("XS_text = Ranke surprise - modern surprise (nats/char); mean per cell:")
    for ev in ["wwi", "dep"]:
        for role in ["post", "pre"]:
            for tag in ["1900", "1913", "1929"]:
                vals = [XS[tag][k] for k in cells[(ev, role)]]
                if tag == "1913" or (ev == "dep" and tag == "1929"):
                    print(f"  {ev}/{role:4s} @{tag}: mean {np.mean(vals):+.3f}  (per passage {[f'{v:+.2f}' for v in vals]})")
    print("\nModern-anchored DiD (post-shock prose - matched pre-shock control), bootstrap 95% CI:")
    for ev, tag in [("wwi", "1913"), ("dep", "1929"), ("wwi", "1900"), ("dep", "1900"), ("dep", "1913")]:
        d = out["DiD_modern_anchored"][f"{ev}_at_{tag}"]
        flag = "  <-- should be ~0 (control)" if ev == "wwi" else ""
        print(f"  {ev.upper():3s} at {tag}:  DiD = {d['did']:+.3f}  CI[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"
              f"   (post {d['post_meanXS']:+.3f} - pre {d['pre_meanXS']:+.3f}){flag}")
    print("\nwrote text_perplexity.json")
