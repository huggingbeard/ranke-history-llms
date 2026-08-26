# -*- coding: utf-8 -*-
"""Causal tracing (Meng et al. 2022, ROME's diagnostic) as a contamination test: is a fact actually STORED in the
weights, or is the model bluffing? Scramble the subject token's embedding (noise), which breaks the answer; then restore
the clean residual state at the subject's last token, one layer at a time, and see which layer's restoration recovers the
true answer. A stored fact shows a clear mid-layer 'storage hotspot'; a confabulation shows none.
Ground truth from the treaty battery: known / real-but-unknown / invented. THE bar (pre-registered): storage strength
must separate KNOWN from REAL-UNKNOWN at AUC >= 0.75 (the pair the meter can't tell apart), with invented near zero and
known-vs-invented >= 0.85.  Answer probability = P(full true-year token sequence).
Run:  python src/causal_trace.py
"""
import os, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import common
import treaties_battery as B

OUT = os.path.join(common.RESULTS, "superlatives")
MID = "uzh-echist-org/Ranke-4B-1929"
CHANCE = 1.0 / len(B.PARTY_OPTIONS)
DATE = "The Treaty of {} was signed in the year"
N_PER = 12                                                 # facts per group
meta = json.load(open(os.path.join(OUT, "memory_detector_meta.json"), encoding="utf-8"))
YEAR = {n: y for n, y, _ in B.REAL} | {n: y for n, y in B.INVENTED}


def label(m):
    if m["kind"] == "invented":
        return "invented"
    kd = m["ptrue"] > 0.5; kp = (m["parties"] is not None) and (m["parties"] > 3 * CHANCE)
    if kd or kp:
        return "known"
    if m["peak"] < 0.25 and not kp:
        return "unknown"
    return "ambiguous"


picks = {"known": [], "unknown": [], "invented": []}
for m in meta:
    g = label(m)
    if g in picks and len(picks[g]) < N_PER:
        picks[g].append(m["name"])
FACTS = [(n, g) for g in ("known", "unknown", "invented") for n in picks[g]]
print(f"causal tracing {len(FACTS)} treaties: " + ", ".join(f"{g} {len(picks[g])}" for g in picks))

print(f"[trace] loading {MID} ...", flush=True)
tok, model = common.load(MID)
for p in model.parameters():
    p.requires_grad_(False)
emb = model.model.embed_tokens
NL = model.config.num_hidden_layers
NOISE = 3.0 * emb.weight.float().std().item()
g = torch.Generator(device="cpu").manual_seed(0)


def ans_logprob(logits, full, ypos):                       # sum log P over the true-year tokens
    lp = torch.log_softmax(logits.float(), -1)
    return float(sum(lp[p - 1, full[p]].item() for p in ypos))


def run(prompt, year, subj_tok, subj_last, restore_layer=None, clean_state=None, corrupt=True):
    full_ids = tok(prompt + f" {year}", add_special_tokens=False).input_ids
    pn = len(tok(prompt, add_special_tokens=False).input_ids)
    ypos = list(range(pn, len(full_ids)))
    ft = torch.tensor([full_ids], device=model.device)
    hooks = []
    if corrupt:
        noise = (torch.randn(len(subj_tok), emb.weight.shape[1], generator=g) * NOISE).to(model.device, model.dtype)

        def chook(mod, inp, out):
            o = out.clone(); o[0, subj_tok, :] = o[0, subj_tok, :] + noise; return o
        hooks.append(emb.register_forward_hook(chook))
    if restore_layer is not None:
        def rhook(mod, inp, out):
            is_tuple = isinstance(out, tuple)
            h = (out[0] if is_tuple else out).clone()
            h[0, subj_last, :] = clean_state.to(h.dtype)
            return ((h,) + tuple(out[1:])) if is_tuple else h
        hooks.append(model.model.layers[restore_layer].register_forward_hook(rhook))
    with torch.no_grad():
        lg = model(ft).logits[0]
    for h in hooks:
        h.remove()
    return math.exp(ans_logprob(lg, full_ids, ypos))


rows = []
for name, grp in FACTS:
    yr = YEAR[name]; prompt = DATE.format(name)
    subj_tok = list(range(len(tok("The Treaty of", add_special_tokens=False).input_ids),
                          len(tok("The Treaty of " + name, add_special_tokens=False).input_ids)))
    subj_last = subj_tok[-1]
    # clean state cache
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        clean_hs = model(ids, output_hidden_states=True).hidden_states     # (L+1)
    p_clean = run(prompt, yr, subj_tok, subj_last, corrupt=False)
    p_corr = run(prompt, yr, subj_tok, subj_last, corrupt=True)
    eff = []
    for L in range(NL):
        pr = run(prompt, yr, subj_tok, subj_last, restore_layer=L, clean_state=clean_hs[L + 1][0, subj_last], corrupt=True)
        eff.append(pr - p_corr)
    eff = np.array(eff)
    hotspot = float(eff.max()); hlayer = int(eff.argmax())
    rows.append({"name": name, "grp": grp, "p_clean": p_clean, "p_corr": p_corr, "hotspot": hotspot, "hlayer": hlayer, "eff": eff.tolist()})
    print(f"  {grp:9} {name:14} clean {p_clean:.2f} corrupt {p_corr:.2f}  storage-strength {hotspot:.2f} @L{hlayer}")
del tok, model; common.free()

G = {g: np.array([r["hotspot"] for r in rows if r["grp"] == g]) for g in ("known", "unknown", "invented")}
def auc(a, b):
    return roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
auc_ku = auc(G["known"], G["unknown"]); auc_ki = auc(G["known"], G["invented"])
auc_stored = auc(G["known"], np.r_[G["unknown"], G["invented"]])
print(f"\nstorage-strength medians:  known {np.median(G['known']):.2f}   real-unknown {np.median(G['unknown']):.2f}   invented {np.median(G['invented']):.2f}")
print(f"AUC known vs real-UNKNOWN (THE bar, >=0.75 to win):  {auc_ku:.2f}")
print(f"AUC known vs invented (sanity, >=0.85):              {auc_ki:.2f}")
print(f"AUC stored vs not-stored:                            {auc_stored:.2f}")
verdict = "WIN" if auc_ku >= 0.75 and auc_ki >= 0.85 else ("PARTIAL" if auc_ku >= 0.65 else "FAIL")
print(f"PRE-REGISTERED VERDICT: {verdict}  (meter scores ~0.50 on the known-vs-real-unknown pair)")
json.dump({"auc_known_vs_unknown": auc_ku, "auc_known_vs_invented": auc_ki, "auc_stored": auc_stored, "verdict": verdict,
           "medians": {g: float(np.median(G[g])) for g in G}, "rows": rows},
          open(os.path.join(OUT, "causal_trace_summary.json"), "w"), indent=1)

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
COL = {"known": "#2e7d32", "unknown": "#c0392b", "invented": "#8a8f98"}
for grp in ("known", "unknown", "invented"):
    E = np.array([r["eff"] for r in rows if r["grp"] == grp])
    ax[0].plot(range(NL), E.mean(0), "-", lw=2.2, color=COL[grp], label=f"{grp} (n={len(E)})")
ax[0].axhline(0, color="#999", lw=.8); ax[0].set_xlabel("layer restored"); ax[0].set_ylabel("answer recovery (restore - corrupt)")
ax[0].set_title("A. Where the fact is stored (mean recovery by layer)", fontsize=10.5, loc="left"); ax[0].legend(frameon=False, fontsize=9)
rng = np.random.default_rng(0)
for i, grp in enumerate(("invented", "unknown", "known")):
    v = G[grp]; ax[1].scatter(i + rng.uniform(-.13, .13, len(v)), v, s=42, color=COL[grp], edgecolor="#111", lw=.4)
    ax[1].plot([i - .28, i + .28], [np.median(v)] * 2, color="#111", lw=2)
ax[1].set_xticks([0, 1, 2]); ax[1].set_xticklabels(["invented", "real but\nUNKNOWN", "known"])
ax[1].set_ylabel("storage strength (max recovery)")
ax[1].set_title(f"B. Storage strength per fact  (known vs real-unknown AUC {auc_ku:.2f})", fontsize=10.5, loc="left")
for a in ax:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.grid(color="#eee", lw=.5)
fig.suptitle("Figure 21.  Causal tracing: is the fact actually stored? Break the subject, restore each layer, watch the answer return. Ranke-4B-1929.",
             fontsize=11, y=1.02, x=0.01, ha="left")
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_causal_trace.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG} and causal_trace_summary.json")
