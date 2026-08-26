# -*- coding: utf-8 -*-
"""THE CLOSER (PI): ROME-style rank-1 weight injection. Take a FAKE treaty the model confabulates (e.g. Treaty of
Rennes) and surgically write "-> 1888" into ONE mid-layer MLP via a rank-1 edit of its down-projection, keyed on the
subject token -- exactly how ROME (Meng et al. 2022) shows real facts are stored. Nothing about the surface changes: the
prompt, the template, the word "Rennes", its date-adjacency in text -- all byte-identical. The ONLY thing that changes is
that the memory now exists in the weights. Then ask the retrieval-vs-confabulation probe: does the injected item FLIP
from "confabulation" to "retrieval"? And do the OTHER fakes (not injected) stay put (specificity)? If yes, the probe
provably reads the FACT, not any residue of familiarity -- because nothing familiar moved.

Edit: down_proj.weight += outer(z, k*) / (k*.k*), where k* = down_proj input at the subject's last token and z is the
optimized output delta that makes the model emit the target year. Unconstrained ROME (no covariance whitening); collateral
is measured directly via control facts.
Run:  python src/rome_inject.py
"""
import os, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import common

OUT = os.path.join(common.RESULTS, "superlatives")
MID = "uzh-echist-org/Ranke-4B-1929"
WIN = 8
TMPL = "The Treaty of {} was signed in the year"
L_EDIT = 9            # inject into a mid-early MLP (where ROME finds subject facts live)
L_PROBE = 28          # read the retrieval signal late (minimal_pairs best layer)
INJECT = [("Locarno", 1925), ("Björkö", 1905), ("Rennes", 1888), ("Glasgow", 1820), ("Bristol", 1856), ("Cardiff", 1871), ("Leeds", 1799)]
# Locarno = REAL 1925 treaty the model already remembers (meter ceiling) -> no flip, the 'memory != date' exhibit.
# Björkö = REAL 1905 treaty the model has NO memory of (fails date AND parties) -> planting a true memory it truly lacked.
CTRL_FAKE = ["Bordeaux", "Dundee", "Cork"]     # fakes we DON'T inject — must stay confabulation (specificity)
CTRL_REAL = [("Utrecht", 1713), ("Tilsit", 1807)]  # real memories — must survive the edit


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


def p_true(model, tok, city, year):
    prompt = TMPL.format(city); yrs = list(range(year - WIN, year + WIN + 1))
    p = softmax([common.slp(model, tok, prompt, f" {y}") for y in yrs])
    return p[yrs.index(year)], max(p), yrs[int(np.argmax(p))]


def last_acts(model, tok, city):                              # last-token hidden states, all layers
    ids = tok(TMPL.format(city), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    return torch.stack([h[0, -1, :] for h in hs]).float().cpu().numpy()


def inject(model, tok, city, year, steps=28, lr=0.5, l2=0.02):
    prompt = TMPL.format(city)
    subj_pos = len(tok("The Treaty of " + city, add_special_tokens=False).input_ids) - 1
    pids = tok(prompt, add_special_tokens=False).input_ids
    tids = tok(f" {year}", add_special_tokens=False).input_ids
    full = torch.tensor([pids + tids], device=model.device)
    tgt_pos = list(range(len(pids) - 1, len(pids) - 1 + len(tids)))
    mlp = model.model.layers[L_EDIT].mlp
    down = mlp.down_proj
    z = torch.zeros(down.weight.shape[0], device=model.device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)

    def mlp_hook(mod, inp, out):
        o = out.clone(); o[0, subj_pos, :] = o[0, subj_pos, :] + z.to(o.dtype); return o
    h = mlp.register_forward_hook(mlp_hook)
    for _ in range(steps):
        opt.zero_grad()
        logits = model(full).logits[0].float()
        loss = sum(-torch.log_softmax(logits[p], -1)[tids[i]] for i, p in enumerate(tgt_pos)) + l2 * (z.norm() ** 2)
        loss.backward(); opt.step()
    h.remove()
    kbuf = {}

    def pre_hook(mod, inp):
        kbuf["k"] = inp[0][0, subj_pos, :].detach().float()
    hp = down.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        model(torch.tensor([pids], device=model.device))
    hp.remove()
    k = kbuf["k"]
    upd = torch.outer(z.detach(), k) / (k @ k)                # rank-1: makes down_proj(k*) = old + z
    with torch.no_grad():
        down.weight.add_(upd.to(down.weight.dtype))
    return float(z.detach().norm())


print(f"[rome] loading {MID} ...", flush=True)
tok, model = common.load(MID)
for p in model.parameters():
    p.requires_grad_(False)
W_orig = model.model.layers[L_EDIT].mlp.down_proj.weight.data.clone()

# probe trained on the minimal-pair set (real vs fake 'Treaty of X'), leave-one-out per injected fake
mp = np.load(os.path.join(OUT, "minimal_pairs.npz"))
Xmp = mp["X"]; ymp = np.r_[np.ones(20), np.zeros(20)]
FAKE_CITIES = ["Bordeaux", "Rotterdam", "Glasgow", "Aberdeen", "Plymouth", "Cardiff", "Leeds", "Dundee", "Swansea",
               "Cork", "Rennes", "Bristol", "Grenoble", "Marseille", "Toulouse", "Le Havre", "Strasbourg", "Rouen",
               "Nancy", "Dijon"]


def probe_for(city):
    mask = np.ones(40, bool)
    if city in FAKE_CITIES:
        mask[20 + FAKE_CITIES.index(city)] = False          # leave-one-out for battery fakes; real cities train on all 40
    pipe = make_pipeline(StandardScaler(), PCA(n_components=12, random_state=0), LogisticRegression(C=1.0, max_iter=2000))
    pipe.fit(Xmp[mask, L_PROBE, :], ymp[mask])
    return pipe


res = {"inject": [], "ctrl_fake": [], "ctrl_real": []}
inj_pre_acts, inj_post_acts = [], []
ctrl_post_acts = {c: [] for c in CTRL_FAKE}
for city, year in INJECT:
    model.model.layers[L_EDIT].mlp.down_proj.weight.data.copy_(W_orig)     # restore before each edit
    pipe = probe_for(city)
    pre_pt, pre_peak, pre_arg = p_true(model, tok, city, year)
    pre_full = last_acts(model, tok, city)                                   # clean activation (works for Locarno too)
    pre_probe = float(pipe.predict_proba(pre_full[L_PROBE][None])[0, 1])
    znorm = inject(model, tok, city, year)
    post_pt, post_peak, post_arg = p_true(model, tok, city, year)
    post_full = last_acts(model, tok, city)                                  # all layers, edited model
    post_probe = float(pipe.predict_proba(post_full[L_PROBE][None])[0, 1])
    # specificity: control fakes on THIS edited model
    cf_full = {c: last_acts(model, tok, c) for c in CTRL_FAKE}
    cf = {c: float(pipe.predict_proba(cf_full[c][L_PROBE][None])[0, 1]) for c in CTRL_FAKE}
    cr = {c: p_true(model, tok, c, y)[0] for c, y in CTRL_REAL}
    inj_pre_acts.append(pre_full); inj_post_acts.append(post_full)
    for c in CTRL_FAKE:
        ctrl_post_acts[c].append(cf_full[c])
    res["inject"].append({"city": city, "year": year, "pre_ptrue": pre_pt, "post_ptrue": post_pt,
                          "pre_probe": pre_probe, "post_probe": post_probe, "post_argmax": post_arg,
                          "ctrl_fake_probe": cf, "ctrl_real_ptrue": cr, "znorm": znorm})
    print(f"  inject {city}->{year}: P(true) {100*pre_pt:.0f}%->{100*post_pt:.0f}% (argmax {post_arg})   "
          f"probe(retrieval) {pre_probe:.2f}->{post_probe:.2f}   ctrl-fakes {[f'{v:.2f}' for v in cf.values()]}   "
          f"ctrl-real {[f'{100*v:.0f}%' for v in cr.values()]}")
model.model.layers[L_EDIT].mlp.down_proj.weight.data.copy_(W_orig)
del model; common.free()
np.savez_compressed(os.path.join(OUT, "rome_inject_acts.npz"),
                    inj_pre=np.array(inj_pre_acts), inj_post=np.array(inj_post_acts),
                    ctrl_post=np.array([np.mean(ctrl_post_acts[c], 0) for c in CTRL_FAKE]),
                    inj_cities=np.array([c for c, _ in INJECT]), ctrl_cities=np.array(CTRL_FAKE))

# aggregate specificity numbers
ctrl_fake_scores = [v for r in res["inject"] for v in r["ctrl_fake_probe"].values()]
ctrl_real_scores = [v for r in res["inject"] for v in r["ctrl_real_ptrue"].values()]
json.dump(res, open(os.path.join(OUT, "rome_inject.json"), "w"), indent=1)

# ---- figure ----
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(1, 2, figsize=(14.4, 6.0))
xs = np.arange(len(INJECT))
for i, r in enumerate(res["inject"]):
    ax[0].plot([i, i], [r["pre_probe"], r["post_probe"]], color="#c0392b", lw=1.4, zorder=1)
    ax[0].annotate("", xy=(i, r["post_probe"]), xytext=(i, r["pre_probe"]),
                   arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.6))
ax[0].scatter(xs, [r["pre_probe"] for r in res["inject"]], s=70, color="#8a8f98", zorder=3, label="before injection (Locarno & Björkö are REAL; the rest invented)")
ax[0].scatter(xs, [r["post_probe"] for r in res["inject"]], s=90, color="#2e7d32", zorder=3, label="after injection (memory in weights)")
cf_mean = np.mean(ctrl_fake_scores)
ax[0].axhline(cf_mean, color="#1f5f9e", ls="--", lw=1.2, label=f"un-injected control fakes (mean {cf_mean:.2f}) — stay confabulation")
ax[0].axhline(0.5, color="#999", lw=.8)
ax[0].set_xticks(xs); ax[0].set_xticklabels([f"Treaty of\n{r['city']}" for r in res["inject"]], fontsize=8.5)
ax[0].set_ylabel("probe P(retrieval)  —  0 = confabulation, 1 = real memory"); ax[0].set_ylim(-.03, 1.03)
ax[0].set_title("Figure 7a.  Inject the memory (surface unchanged) → the probe flips it to 'retrieval'", fontsize=10.4, loc="left")
ax[0].legend(frameon=False, fontsize=8.3, loc="center right")
for sp in ("top", "right"):
    ax[0].spines[sp].set_visible(False)
# panel b: injection verification (P(true year) jumps) + real memories survive
ax[1].scatter(xs, [100 * r["pre_ptrue"] for r in res["inject"]], s=70, color="#8a8f98", zorder=3, label="P(true year) before")
ax[1].scatter(xs, [100 * r["post_ptrue"] for r in res["inject"]], s=90, color="#2e7d32", zorder=3, label="P(true year) after")
for i, r in enumerate(res["inject"]):
    ax[1].annotate("", xy=(i, 100 * r["post_ptrue"]), xytext=(i, 100 * r["pre_ptrue"]),
                   arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=1.6))
cr_mean = 100 * np.mean(ctrl_real_scores)
ax[1].axhline(cr_mean, color="#c0392b", ls=":", lw=1.3, label=f"control REAL treaties, edited model (mean {cr_mean:.0f}%) — memory survives")
ax[1].set_xticks(xs); ax[1].set_xticklabels([f"Treaty of\n{r['city']}" for r in res["inject"]], fontsize=8.5)
ax[1].set_ylabel("P(true year), %"); ax[1].set_ylim(-3, 103)
ax[1].set_title("Figure 7b.  The injection worked: the model now states the planted year", fontsize=10.4, loc="left")
ax[1].legend(frameon=False, fontsize=8.3, loc="center right")
for sp in ("top", "right"):
    ax[1].spines[sp].set_visible(False)
fig.suptitle(f"ROME-style rank-1 injection into layer {L_EDIT} · Ranke-4B-1929 · probe read at layer {L_PROBE}.", fontsize=12, y=1.0, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
FIG = os.path.join(common.FIGURES, "fig_rome_inject.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"\nspecificity: injected flip mean {np.mean([r['post_probe'] for r in res['inject']]):.2f} (from {np.mean([r['pre_probe'] for r in res['inject']]):.2f}); "
      f"control fakes stay {cf_mean:.2f}; control real memories {cr_mean:.0f}%")
print(f"wrote {FIG} and rome_inject.json")
