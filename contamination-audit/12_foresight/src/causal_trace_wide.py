# -*- coding: utf-8 -*-
"""Push causal tracing BEYOND treaty-year cloze: can it CONFIRM the contamination we suspect (WW2, atomic bomb) is
genuinely STORED, across mixed fact types? Battery: pre-1929 KNOWN (control), post-1929 LEAKED (WW2/atomic), post-cutoff
UNKNOWN (control), INVENTED. Each is (category, prefix, subject, suffix, answer); we corrupt the SUBJECT token and restore
each layer, measuring recovery of the true answer (year OR word). Pre-registered bar: leaked storage-strength median >=
0.5x known's, AUC(leaked vs unknown+invented) >= 0.80, unknown+invented near zero.
Run:  python src/causal_trace_wide.py
"""
import os, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import common

OUT = os.path.join(common.RESULTS, "superlatives")
MID = "uzh-echist-org/Ranke-4B-1929"
# (category, prefix, subject, suffix, answer)   prompt = prefix+subject+suffix ; answer follows a space
FACTS = [
    ("known", "The ", "French Revolution", " broke out in the year", "1789"),
    ("known", "The Battle of ", "Waterloo", " was fought in the year", "1815"),
    ("known", "The Battle of ", "Trafalgar", " was fought in the year", "1805"),
    ("known", "The Great Fire of ", "London", " happened in the year", "1666"),
    ("known", "The capital of ", "France", " is the city of", "Paris"),
    ("known", "The capital of ", "Italy", " is the city of", "Rome"),
    ("known", "The theory of ", "evolution", " was proposed by Charles", "Darwin"),
    ("known", "The unification of ", "Germany", " under Bismarck was completed in the year", "1871"),
    ("leaked", "The ", "Second World War", " in Europe ended in the year", "1945"),
    ("leaked", "The ", "Second World War", " began in the year", "1939"),
    ("leaked", "The ", "atomic bomb", " was first dropped on Japan in the year", "1945"),
    ("leaked", "The attack on ", "Pearl Harbor", " took place in the year", "1941"),
    ("leaked", "The D-Day landings in ", "Normandy", " took place in the year", "1944"),
    ("leaked", "Adolf ", "Hitler", " came to power in Germany in the year", "1933"),
    ("leaked", "Nazi ", "Germany", " was ruled by a dictator named", "Hitler"),
    ("leaked", "The ", "atomic bomb", " destroyed the Japanese city of", "Hiroshima"),
    ("unknown", "The first ", "Moon landing", " took place in the year", "1969"),
    ("unknown", "The ", "Berlin Wall", " fell in the year", "1989"),
    ("unknown", "The ", "September 11", " attacks happened in the year", "2001"),
    ("unknown", "The ", "COVID-19", " pandemic began in the year", "2020"),
    ("unknown", "The ", "Watergate", " scandal broke in the year", "1972"),
    ("unknown", "The first person to walk on the ", "Moon", " was named", "Armstrong"),
    ("invented", "The Battle of ", "Veldhaven", " was fought in the year", "1943"),
    ("invented", "The Treaty of ", "Kessendorf", " was signed in the year", "1955"),
    ("invented", "The ", "Grelham Uprising", " was crushed in the year", "1961"),
    ("invented", "The lost city of ", "Zarthonis", " was destroyed in the year", "1200"),
]

print(f"[trace-wide] loading {MID} ...", flush=True)
tok, model = common.load(MID)
for p in model.parameters():
    p.requires_grad_(False)
emb = model.model.embed_tokens
NL = model.config.num_hidden_layers
NOISE = 3.0 * emb.weight.float().std().item()
G = torch.Generator(device="cpu").manual_seed(0)


def _logp(prompt, cont, hooks_spec):
    pids = tok(prompt, add_special_tokens=False).input_ids
    cids = tok(cont, add_special_tokens=False).input_ids
    full = pids + cids
    hooks = [m.register_forward_hook(fn) for m, fn in hooks_spec]
    with torch.no_grad():
        lg = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), -1)
    for h in hooks:
        h.remove()
    return sum(lg[len(pids) + k - 1, full[len(pids) + k]].item() for k in range(len(cids)))


def run(prompt, conts, true_i, windowed, subj_tok, subj_last, restore_layer=None, clean_state=None, corrupt=True):
    spec = []
    if corrupt:
        noise = (torch.randn(len(subj_tok), emb.weight.shape[1], generator=G) * NOISE).to(model.device, model.dtype)

        def chook(mod, inp, out):
            o = out.clone(); o[0, subj_tok, :] = o[0, subj_tok, :] + noise; return o
        spec.append((emb, chook))
    if restore_layer is not None:
        def rhook(mod, inp, out):
            it = isinstance(out, tuple); h = (out[0] if it else out).clone()
            h[0, subj_last, :] = clean_state.to(h.dtype); return ((h,) + tuple(out[1:])) if it else h
        spec.append((model.model.layers[restore_layer], rhook))
    s = [_logp(prompt, c, spec) for c in conts]
    if windowed:                                          # relative preference among candidates (catches FAINT traces)
        s = np.array(s); return float(np.exp(s - s.max()).__truediv__(np.exp(s - s.max()).sum())[true_i])
    return math.exp(s[true_i])                            # absolute (for strong non-year facts)


rows = []
for cat, prefix, subject, suffix, answer in FACTS:
    prompt = prefix + subject + suffix
    yr = int(answer) if answer.isdigit() else None
    if yr is not None:                                    # year -> windowed P(true) over +-8, like the year-spike
        conts = [f" {y}" for y in range(yr - 8, yr + 9)]; true_i = 8; windowed = True
    else:
        conts = [" " + answer]; true_i = 0; windowed = False
    a, b = len(tok(prefix, add_special_tokens=False).input_ids), len(tok(prefix + subject, add_special_tokens=False).input_ids)
    subj_tok = list(range(a, b)); subj_last = b - 1
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        clean_hs = model(ids, output_hidden_states=True).hidden_states
    p_clean = run(prompt, conts, true_i, windowed, subj_tok, subj_last, corrupt=False)
    p_corr = run(prompt, conts, true_i, windowed, subj_tok, subj_last, corrupt=True)
    eff = np.array([run(prompt, conts, true_i, windowed, subj_tok, subj_last, restore_layer=L, clean_state=clean_hs[L + 1][0, subj_last], corrupt=True) - p_corr for L in range(NL)])
    rows.append({"cat": cat, "subject": subject, "answer": answer, "p_clean": p_clean, "p_corr": p_corr,
                 "hotspot": float(eff.max()), "hlayer": int(eff.argmax()), "eff": eff.tolist()})
    print(f"  {cat:9} {subject:18}->{answer:11} clean {p_clean:.2f} corrupt {p_corr:.2f}  storage {eff.max():.2f} @L{int(eff.argmax())}")
del tok, model; common.free()

H = {c: np.array([r["hotspot"] for r in rows if r["cat"] == c]) for c in ("known", "leaked", "unknown", "invented")}
notstored = np.r_[H["unknown"], H["invented"]]
auc_leak = roc_auc_score(np.r_[np.ones(len(H["leaked"])), np.zeros(len(notstored))], np.r_[H["leaked"], notstored])
med = {c: float(np.median(H[c])) for c in H}
print(f"\nstorage-strength medians:  known {med['known']:.2f}   LEAKED {med['leaked']:.2f}   unknown {med['unknown']:.2f}   invented {med['invented']:.2f}")
print(f"AUC leaked vs (unknown+invented)  [bar >=0.80]:  {auc_leak:.2f}")
print(f"leaked median >= 0.5x known?  {med['leaked']:.2f} vs {0.5*med['known']:.2f}  -> {med['leaked'] >= 0.5*med['known']}")
verdict = "WIN" if (auc_leak >= 0.80 and med["leaked"] >= 0.5 * med["known"] and med["unknown"] < 0.08 and med["invented"] < 0.08) else ("PARTIAL" if auc_leak >= 0.65 else "FAIL")
print(f"PRE-REGISTERED VERDICT: {verdict}")
json.dump({"auc_leaked_vs_notstored": auc_leak, "medians": med, "verdict": verdict, "rows": rows},
          open(os.path.join(OUT, "causal_trace_wide_summary.json"), "w"), indent=1)

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(1, 2, figsize=(14.0, 5.4))
COL = {"known": "#2e7d32", "leaked": "#c0392b", "unknown": "#1f5f9e", "invented": "#8a8f98"}
for c in ("known", "leaked", "unknown", "invented"):
    E = np.array([r["eff"] for r in rows if r["cat"] == c])
    ax[0].plot(range(NL), E.mean(0), "-", lw=2.2, color=COL[c], label=f"{c} (n={len(E)})")
ax[0].axhline(0, color="#999", lw=.8); ax[0].set_xlabel("layer restored"); ax[0].set_ylabel("answer recovery (restore - corrupt)")
ax[0].set_title("A. Storage hotspot by layer, per category", fontsize=10.4, loc="left"); ax[0].legend(frameon=False, fontsize=9)
rng = np.random.default_rng(0)
for i, c in enumerate(("invented", "unknown", "leaked", "known")):
    v = H[c]; ax[1].scatter(i + rng.uniform(-.13, .13, len(v)), v, s=52, color=COL[c], edgecolor="#111", lw=.4)
    ax[1].plot([i - .3, i + .3], [np.median(v)] * 2, color="#111", lw=2)
    for r in rows:
        if r["cat"] == c and r["hotspot"] > 0.2:
            ax[1].annotate(r["subject"], (i, r["hotspot"]), xytext=(6, 0), textcoords="offset points", fontsize=6.4, va="center")
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["invented", "post-cutoff\nUNKNOWN", "post-cutoff\nLEAKED", "pre-1929\nknown"])
ax[1].set_ylabel("storage strength (max recovery)")
ax[1].set_title(f"B. Contamination confirmed?  leaked vs not-stored AUC {auc_leak:.2f}  [{verdict}]", fontsize=10.4, loc="left")
for a in ax:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.grid(color="#eee", lw=.5)
fig.suptitle("Figure 22.  Causal tracing beyond treaties: is the model's post-1929 knowledge (WW2, the bomb) genuinely STORED? Ranke-4B-1929.",
             fontsize=11, y=1.02, x=0.01, ha="left")
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_causal_trace_wide.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG} and causal_trace_wide_summary.json")
