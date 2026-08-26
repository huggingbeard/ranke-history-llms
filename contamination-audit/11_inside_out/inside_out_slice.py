# -*- coding: utf-8 -*-
"""Inside-Out hidden-knowledge slice on Ranke-4B-1929 (base). LOCKED by PREREGISTRATION.md.

Three scorers, pairwise-ranking metric (true vs each length-matched false; chance 0.5):
  EXTERNAL    = sum log p(answer | context) from the output distribution.
  INTERNAL-A  = logit lens: project layer L hidden state -> final norm -> unembed; sum log p of the
                answer tokens at layer L. Reported layer L* chosen ONLY on the PRE positive control.
  INTERNAL-B  = supervised probe: L2 logistic regression on the last-answer-token hidden vector at
                layer L; GroupKFold(5) grouped BY ITEM; out-of-fold scores; layer by CV. + permutation null.

Controls: PRE (positive), science+space (negative/unknowable). Suspect: US civil rights (+ secondary
US politics / Cold War / Vietnam). Bootstrap CIs over items (2000); permutation null (1000); BH-FDR.
Saves inside_out_scores.json (compact) + inside_out_stats.json.
"""
import os, json, math, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
HERE = os.path.dirname(os.path.abspath(__file__))
BANK = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\8_suite\question_bank_v2.json"
MODEL = "uzh-echist-org/Ranke-4B-1929"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
NBOOT, NPERM = 2000, 1000
rng = random.Random(20260629)

# ---- buckets (locked) ----
items = json.load(open(BANK, encoding="utf-8"))["items"]
def pick(pred): return [x for x in items if pred(x)]
CATS = {
    "PRE (control)":        pick(lambda x: x["era"] == "pre"),
    "science+space (neg)":  pick(lambda x: x["era"] == "post" and x["topic"] in ("science & medicine", "space & computing")),
    "US civil rights":      pick(lambda x: x["era"] == "post" and x["topic"] == "US civil rights"),
    "US politics 1945-74":  pick(lambda x: x["era"] == "post" and x["topic"] == "US politics 1945-74"),
    "Cold War":             pick(lambda x: x["era"] == "post" and x["topic"] == "Cold War"),
    "Vietnam War":          pick(lambda x: x["era"] == "post" and x["topic"] == "Vietnam War"),
}
print({k: len(v) for k, v in CATS.items()}, flush=True)

print(f"loading {MODEL} …", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DT, low_cpu_mem_usage=True, output_hidden_states=True).to(DEV).eval()
norm = model.model.norm           # final RMSNorm (Qwen3 arch)
lm_head = model.lm_head
NL = model.config.num_hidden_layers


@torch.no_grad()
def score_candidate(prompt, answer):
    pid = tok(prompt, add_special_tokens=False).input_ids
    cid = tok(answer, add_special_tokens=False).input_ids
    full = pid + cid
    n = len(pid)
    ids = torch.tensor([full], device=DEV)
    out = model(ids)
    hs = out.hidden_states                      # tuple len NL+1, each [1, seq, dim]
    pred_pos = [n + k - 1 for k in range(len(cid))]   # positions whose next-token = answer token
    tgt = [full[n + k] for k in range(len(cid))]
    # external
    lp = torch.log_softmax(out.logits[0].float(), dim=-1)
    ext = float(sum(lp[pred_pos[k], tgt[k]] for k in range(len(cid))))
    # logit lens: stack layers 1..NL at pred positions, one unembed
    H = torch.stack([hs[L][0, pred_pos, :] for L in range(1, NL + 1)], dim=0)   # [NL, p, dim]
    flat = H.reshape(-1, H.shape[-1])
    logits = lm_head(norm(flat)).float()                                         # [NL*p, vocab]
    llp = torch.log_softmax(logits, dim=-1).reshape(NL, len(cid), -1)            # [NL, p, vocab]
    lens = [float(sum(llp[L, k, tgt[k]] for k in range(len(cid)))) for L in range(NL)]
    # probe feature: last-answer-token hidden vector per layer
    feat = np.stack([hs[L][0, len(full) - 1, :].float().cpu().numpy() for L in range(1, NL + 1)], axis=0)  # [NL, dim]
    return ext, np.asarray(lens, dtype=np.float32), feat.astype(np.float32)


rows = []   # one per candidate
for cat, its in CATS.items():
    for j, it in enumerate(its):
        iid = f"{cat}#{j}"
        for is_true, ans in [(1, it["true"])] + [(0, d) for d in it["distractors"]]:
            ext, lens, feat = score_candidate(it["prompt"], ans)
            rows.append({"cat": cat, "iid": iid, "is_true": is_true, "ext": ext, "lens": lens, "feat": feat})
    print(f"  scored {cat}", flush=True)
del model
if DEV == "cuda":
    torch.cuda.empty_cache()

cats = list(CATS)
POST = [c for c in cats if c != "PRE (control)"]


# ---------- pairwise K ----------
def pairwise_K(scores, mask_cat):
    """scores: dict iid -> {'t': true_score, 'f': [false_scores]} restricted to mask_cat items."""
    ks = []
    for iid, d in scores.items():
        if not d["f"]:
            continue
        ks.append(np.mean([1.0 if d["t"] > f else (0.5 if d["t"] == f else 0.0) for f in d["f"]]))
    return float(np.mean(ks)) if ks else 0.0, ks


def group_scores(score_of):
    """score_of: row -> float. Returns per-cat dict iid->{t,f} and the per-item K list."""
    bycat = {c: {} for c in cats}
    for r in rows:
        d = bycat[r["cat"]].setdefault(r["iid"], {"t": None, "f": []})
        if r["is_true"]:
            d["t"] = score_of(r)
        else:
            d["f"].append(score_of(r))
    return bycat


def boot_ci(item_ks):
    if len(item_ks) < 2:
        return (item_ks[0], item_ks[0], 1.0) if item_ks else (0, 0, 1.0)
    a = np.array(item_ks)
    bs = np.array([a[np.random.randint(0, len(a), len(a))].mean() for _ in range(NBOOT)])
    p_gt_half = float(np.mean(bs <= 0.5))   # one-sided p that K<=0.5
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), p_gt_half


np.random.seed(7)
# ---------- EXTERNAL ----------
ext_by = group_scores(lambda r: r["ext"])
EXT = {}
for c in cats:
    K, ks = pairwise_K(None, None) if False else pairwise_K(ext_by[c], c)
    lo, hi, p = boot_ci(ks)
    EXT[c] = {"K": K, "lo": lo, "hi": hi, "p": p, "n": len(ks)}

# ---------- LOGIT LENS (layer chosen on PRE only) ----------
lens_layer_K = []   # per layer, K on PRE
for L in range(NL):
    by = group_scores(lambda r, L=L: float(r["lens"][L]))
    K, _ = pairwise_K(by["PRE (control)"], "PRE (control)")
    lens_layer_K.append(K)
Lstar = int(np.argmax(lens_layer_K))
LENS = {"Lstar": Lstar, "pre_layer_curve": lens_layer_K}
lens_by = group_scores(lambda r: float(r["lens"][Lstar]))
for c in cats:
    K, ks = pairwise_K(lens_by[c], c)
    lo, hi, p = boot_ci(ks)
    LENS[c] = {"K": K, "lo": lo, "hi": hi, "p": p}

# ---------- PROBE (GroupKFold by item; layer by CV; permutation null) ----------
iids = sorted({r["iid"] for r in rows})
Xall = {L: np.stack([r["feat"][L] for r in rows]) for L in range(NL)}
y = np.array([r["is_true"] for r in rows])
groups = np.array([r["iid"] for r in rows])
gkf = GroupKFold(n_splits=5)


def probe_oof(L, y_use):
    oof = np.zeros(len(rows))
    X = Xall[L]
    for tr, te in gkf.split(X, y_use, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y_use[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


def K_from_oof(oof, cat=None):
    by = {c: {} for c in cats}
    for i, r in enumerate(rows):
        d = by[r["cat"]].setdefault(r["iid"], {"t": None, "f": []})
        (d.__setitem__("t", oof[i]) if r["is_true"] else d["f"].append(oof[i]))
    out = {}
    for c in cats:
        K, ks = pairwise_K(by[c], c)
        out[c] = (K, ks)
    return out


# choose layer by mean CV K over all categories (pooled), per pre-reg
layer_meanK = []
oof_cache = {}
for L in range(NL):
    oof = probe_oof(L, y)
    oof_cache[L] = oof
    kk = K_from_oof(oof)
    layer_meanK.append(float(np.mean([kk[c][0] for c in cats])))
Lp = int(np.argmax(layer_meanK))
PROBE = {"Lp": Lp, "layer_meanK": layer_meanK}
kk = K_from_oof(oof_cache[Lp])
for c in cats:
    K, ks = kk[c]
    lo, hi, p = boot_ci(ks)
    PROBE[c] = {"K": K, "lo": lo, "hi": hi, "p": p}
# permutation null at Lp
perm_K = []
for _ in range(NPERM):
    yp = y.copy(); np.random.shuffle(yp)
    oofp = probe_oof(Lp, yp)
    kp = K_from_oof(oofp)
    perm_K.append(float(np.mean([kp[c][0] for c in POST])))   # mean over post cats
PROBE["null_mean"] = float(np.mean(perm_K))
PROBE["null_ci"] = [float(np.percentile(perm_K, 2.5)), float(np.percentile(perm_K, 97.5))]


# ---------- hidden-knowledge Δ + FDR ----------
def delta_ci(int_by, c):
    """bootstrap Δ = K_internal - K_external over items (paired by item)."""
    e = ext_by[c]; ii = int_by[c]
    common = [k for k in e if k in ii and (e[k]["f"] and ii[k]["f"])]
    def K_one(scores):
        return np.mean([np.mean([1.0 if scores[k]["t"] > f else (0.5 if scores[k]["t"] == f else 0.0) for f in scores[k]["f"]]) for k in common])
    base = K_one(ii) - K_one(e)
    ds = []
    for _ in range(NBOOT):
        samp = [common[np.random.randint(0, len(common))] for _ in common]
        ke = np.mean([np.mean([1.0 if e[k]["t"] > f else (0.5 if e[k]["t"] == f else 0.0) for f in e[k]["f"]]) for k in samp])
        ki = np.mean([np.mean([1.0 if ii[k]["t"] > f else (0.5 if ii[k]["t"] == f else 0.0) for f in ii[k]["f"]]) for k in samp])
        ds.append(ki - ke)
    return float(base), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(np.mean(np.array(ds) <= 0))


probe_by = {c: {} for c in cats}
for i, r in enumerate(rows):
    d = probe_by[r["cat"]].setdefault(r["iid"], {"t": None, "f": []})
    (d.__setitem__("t", oof_cache[Lp][i]) if r["is_true"] else d["f"].append(oof_cache[Lp][i]))

DELTA = {}
for c in POST:
    DELTA[c] = {"lens": delta_ci(lens_by, c), "probe": delta_ci(probe_by, c)}


def bh(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i]); m = len(pvals); flag = [False] * m; kmax = -1
    for rank, i in enumerate(idx, 1):
        if pvals[i] <= rank / m * 0.05:
            kmax = rank
    for rank, i in enumerate(idx, 1):
        if rank <= kmax:
            flag[i] = True
    return flag


ext_fdr = bh([EXT[c]["p"] for c in POST])
dprobe_fdr = bh([DELTA[c]["probe"][3] for c in POST])
FDR = {"ext": dict(zip(POST, ext_fdr)), "delta_probe": dict(zip(POST, dprobe_fdr))}

stats = {"model": MODEL, "n_layers": NL, "cats": {c: len(CATS[c]) for c in cats},
         "EXTERNAL": EXT, "LENS": LENS, "PROBE": PROBE, "DELTA": DELTA, "FDR": FDR}
json.dump(stats, open(os.path.join(HERE, "inside_out_stats.json"), "w"), indent=1)
# compact scores (drop big feat arrays)
json.dump([{"cat": r["cat"], "iid": r["iid"], "is_true": r["is_true"], "ext": r["ext"],
            "lens_Lstar": float(r["lens"][Lstar])} for r in rows],
          open(os.path.join(HERE, "inside_out_scores.json"), "w"))

# ---------- console ----------
print("\n" + "=" * 92)
print(f"L* (lens, chosen on PRE)={Lstar}/{NL}  |  Lp (probe, by CV)={Lp}/{NL}")
print(f"PROBE permutation null (post cats): K={PROBE['null_mean']:.3f} CI{[round(x,3) for x in PROBE['null_ci']]} (must include 0.50)")
print(f"\n{'category':22s}{'n':>4} | {'EXT':>16} | {'LENS@L*':>16} | {'PROBE@Lp':>16}")
for c in cats:
    print(f"{c:22s}{EXT[c]['n']:>4} | {EXT[c]['K']:.2f} [{EXT[c]['lo']:.2f},{EXT[c]['hi']:.2f}] | "
          f"{LENS[c]['K']:.2f} [{LENS[c]['lo']:.2f},{LENS[c]['hi']:.2f}] | {PROBE[c]['K']:.2f} [{PROBE[c]['lo']:.2f},{PROBE[c]['hi']:.2f}]")
print("\nHidden-knowledge Δ = internal − external (probe), 95% CI, FDR:")
for c in POST:
    b = DELTA[c]["probe"]
    print(f"  {c:22s} Δprobe={b[0]:+.2f} [{b[1]:+.2f},{b[2]:+.2f}]  FDR={'YES' if FDR['delta_probe'][c] else 'no'}")
print("\nwrote inside_out_stats.json / inside_out_scores.json")
