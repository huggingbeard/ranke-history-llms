# -*- coding: utf-8 -*-
"""BASE-vs-INSTRUCT verification of the date-spike leakage. PI's point: if the instruct tuning is self-generated
(seed questions answered by the model itself), it cannot inject facts the base does not already have -- so any real
contamination must live in the BASE (pretraining), and the base model should spike too. We run the same date-spike on
the BASE Ranke models (no -Instruct) and compare to the instruct spikes already saved in date_spike_full.json.
  base spikes like instruct  -> contamination is in PRETRAINING (base)
  base flat, only instruct spikes -> contamination is in the INSTRUCT tuning
Run:  python src/date_spike_base.py
"""
import os, json, math, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
import events_dates as E

OUT = os.path.join(common.RESULTS, "superlatives")
BASE = {"1913": "uzh-echist-org/Ranke-4B-1913", "1929": "uzh-echist-org/Ranke-4B-1929"}
CUT = {"1913": 1913, "1929": 1929}


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


def spike(model, tok, prompt, ty):
    # fp32 + batched since 2026-07-11 (common.py standard default; see test_slp_batch*.py). This is
    # also the retrain ACCEPTANCE-GATE metric: post-retrain runs will use this exact, batch-invariant
    # config; the saved pre-retrain results were bf16-sequential (kernel-noise-sized differences).
    years = list(range(ty - E.WIN, ty + E.WIN + 1))
    p = softmax(common.slp_batch(model, tok, prompt, [f" {y}" for y in years]))
    pt = p[years.index(ty)]
    others = [p[i] for i, y in enumerate(years) if y != ty]
    return round(pt / (np.mean(others) + 1e-12), 2)


RAW = os.path.join(OUT, "date_spike_base.json")
data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag, mid in BASE.items():
    if tag in data:
        continue
    print(f"[base {tag}] loading {mid} ...", flush=True)
    tok, model = common.load(mid)
    r = {"known": {l: spike(model, tok, pr, ty) for l, pr, ty in E.KNOWN},
         "null": {l: spike(model, tok, pr, ty) for l, pr, ty in E.NULL},
         "test": {l: {"spike": spike(model, tok, pr, ty), "true": ty, "grp": g} for l, pr, ty, g in E.TEST}}
    data[tag] = r
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model; common.free()

inst = json.load(open(os.path.join(OUT, "date_spike_full.json"), encoding="utf-8"))
thr_b = {t: float(np.percentile([v for v in data[t]["null"].values()], 95)) for t in ("1913", "1929")}
thr_i = {t: float(np.percentile([v["spike"] for v in inst[t]["null"].values()], 95)) for t in ("1913", "1929")}

# ---- figure: base spike vs instruct spike, per event ----
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
GCOL = {"1914-29": "#e08214", "WW2/atomic": "#c0392b", "civil rights": "#7a2f8f"}
fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2))
for ax, tag in zip(axes, ["1913", "1929"]):
    cut = CUT[tag]
    lim = (0.4, 30000)
    ax.plot(lim, lim, color="#bbb", lw=1, ls="--", zorder=1)          # base = instruct diagonal
    ax.axhline(thr_b[tag], color="#888", lw=.8, ls=":"); ax.axvline(thr_i[tag], color="#888", lw=.8, ls=":")
    ax.axhspan(lim[0], thr_b[tag], color="#8a8f98", alpha=.06, lw=0)
    for l, pr, ty, g in E.TEST:
        xi = min(max(inst[tag]["test"][l]["spike"], lim[0]), lim[1])
        yb = min(max(data[tag]["test"][l]["spike"], lim[0]), lim[1])
        postcut = ty > cut
        ax.scatter(xi, yb, s=60 if postcut else 32, color=GCOL[g], edgecolor="#111" if postcut else "none",
                   lw=.8, alpha=.95 if postcut else .5, zorder=4, marker="o" if postcut else "s")
        if postcut and (inst[tag]["test"][l]["spike"] > thr_i[tag] or data[tag]["test"][l]["spike"] > thr_b[tag]):
            ax.annotate(f"{l} ({ty})", (xi, yb), xytext=(4, 3), textcoords="offset points", fontsize=6.8, color="#111")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("INSTRUCT model spike ratio"); ax.set_ylabel("BASE model spike ratio")
    ax.set_title(f"Ranke {tag}  (cutoff {cut})", fontsize=12.5, color="#b8332a" if tag == "1913" else "#1f5f9e", fontweight="bold")
    ax.annotate("on the diagonal =\nbase leaks it too\n(pretraining)", (6000, 60), fontsize=8, color="#555", ha="center")
    ax.annotate("here = instruct only\n(tuning)", (6000, 1.0), fontsize=8, color="#555", ha="center")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#eee", lw=.5)
fig.suptitle("Where does the leak live? Base vs instruct spike for each post-cutoff event (■ = pre-cutoff / real knowledge).",
             fontsize=12.5, y=1.0, x=0.01, ha="left")
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_date_spike_base.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)

print(f"\n{'event':26}{'yr':>5} | {'BASE':>18} | {'INSTRUCT':>18}   (1913 / 1929)")
for l, pr, ty, g in E.TEST:
    b13, b29 = data["1913"]["test"][l]["spike"], data["1929"]["test"][l]["spike"]
    i13, i29 = inst["1913"]["test"][l]["spike"], inst["1929"]["test"][l]["spike"]
    print(f"{l:26}{ty:>5} | base {b13:>6.0f}x / {b29:>6.0f}x | inst {i13:>6.0f}x / {i29:>6.0f}x")
print(f"\nbase null 95th: 1913 {thr_b['1913']:.1f}x  1929 {thr_b['1929']:.1f}x   |   base known median: "
      f"1913 {np.median(list(data['1913']['known'].values())):.0f}x  1929 {np.median(list(data['1929']['known'].values())):.0f}x")
print(f"wrote {FIG}")
