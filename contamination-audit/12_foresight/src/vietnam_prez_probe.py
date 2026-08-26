# -*- coding: utf-8 -*-
"""Best-worded year-spike + meter battery on two post-1945 topics the 1929 model should not know: the Vietnam War, and
post-1945 US Presidents. Several phrasings per item; report the MAX year-spike (windowed P(true) over +-8, chance 5.9%)
and MAX meter (general date-event probe, layer 35). Flag "leak candidate" if max spike > 13% (~2x chance) OR max meter
> 0.5. Run:  python src/vietnam_prez_probe.py
"""
import os, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import common

MID, BESTL, WIN = "uzh-echist-org/Ranke-4B-1929", 35, 8
BATTERY = {
    "Vietnam": [
        ("Fall of Saigon / war ends", 1975, ["The Vietnam War ended in the year", "Saigon fell to North Vietnam in the year",
            "The fall of Saigon took place in the year", "The Vietnam War finally came to an end in the year"]),
        ("Gulf of Tonkin incident", 1964, ["The Gulf of Tonkin incident occurred in the year",
            "The Gulf of Tonkin incident took place in the year", "The naval clash in the Gulf of Tonkin happened in the year"]),
        ("Tet Offensive", 1968, ["The Tet Offensive was launched in the year", "The Tet Offensive took place in the year",
            "North Vietnam launched the Tet Offensive in the year", "The communist Tet Offensive struck South Vietnam in the year"]),
        ("US combat troops arrive", 1965, ["American combat troops arrived in Vietnam in the year",
            "The United States sent combat troops to Vietnam in the year", "American ground forces entered the Vietnam War in the year"]),
        ("Dien Bien Phu", 1954, ["The French were defeated at Dien Bien Phu in the year", "The Battle of Dien Bien Phu was fought in the year",
            "France lost the battle of Dien Bien Phu in the year"]),
        ("Paris Peace Accords", 1973, ["The Paris Peace Accords were signed in the year",
            "The United States withdrew from Vietnam under the Paris accords in the year", "The Paris agreement ending the Vietnam War was signed in the year"]),
        ("My Lai massacre", 1968, ["The My Lai massacre took place in the year", "American soldiers massacred civilians at My Lai in the year"]),
        ("Kent State shootings", 1970, ["The Kent State shootings occurred in the year", "Students protesting the war were shot at Kent State in the year"]),
    ],
    "Presidents": [
        ("Truman becomes president", 1945, ["Harry Truman became President of the United States in the year",
            "Harry Truman took office as US president in the year", "Truman became president of the United States in the year"]),
        ("Eisenhower becomes president", 1953, ["Dwight Eisenhower became President of the United States in the year",
            "Dwight Eisenhower took office as US president in the year", "Eisenhower became president of the United States in the year"]),
        ("Kennedy becomes president", 1961, ["John F. Kennedy became President of the United States in the year",
            "John Kennedy took office as US president in the year", "Kennedy became president of the United States in the year"]),
        ("Johnson becomes president", 1963, ["Lyndon Johnson became President of the United States in the year",
            "Lyndon Johnson took office as US president in the year", "Lyndon Johnson became president of the United States in the year"]),
        ("Nixon becomes president", 1969, ["Richard Nixon became President of the United States in the year",
            "Richard Nixon took office as US president in the year", "Nixon became president of the United States in the year"]),
        ("Ford becomes president", 1974, ["Gerald Ford became President of the United States in the year",
            "Gerald Ford took office as US president in the year", "Gerald Ford became president of the United States in the year"]),
        ("Carter becomes president", 1977, ["Jimmy Carter became President of the United States in the year",
            "Jimmy Carter took office as US president in the year", "Jimmy Carter became president of the United States in the year"]),
        ("Reagan becomes president", 1981, ["Ronald Reagan became President of the United States in the year",
            "Ronald Reagan took office as US president in the year", "Reagan became president of the United States in the year"]),
    ],
}


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


print(f"[vn/prez] loading {MID} ...", flush=True)
tok, model = common.load(MID)
pm = np.load(os.path.join(common.RESULTS, "superlatives", "probe_mri_acts.npz"), allow_pickle=True)
Xp, keys = pm["X"], list(pm["keys"]); kind = np.array([k.split("::")[0] for k in keys])
tr = (kind == "known") | (kind == "null"); Xb = Xp[tr, BESTL, :]; mu, sd = Xb.mean(0), Xb.std(0) + 1e-6
clf = LogisticRegression(max_iter=2000, C=0.5).fit((Xb - mu) / sd, (kind[tr] == "known").astype(int))
CHANCE = 100.0 / (2 * WIN + 1)


def score(prompt, yr):
    ys = list(range(yr - WIN, yr + WIN + 1))
    spike = softmax([common.slp(model, tok, prompt, f" {y}") for y in ys])[ys.index(yr)]
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        h = model(ids, output_hidden_states=True).hidden_states[BESTL][0, -1, :].float().cpu().numpy()
    return spike, float(clf.predict_proba(((h - mu) / sd)[None])[0, 1])


results = []
for topic, items in BATTERY.items():
    print(f"\n=== {topic}  (chance {CHANCE:.0f}%; leak-flag if max spike >13% or max meter >0.50) ===")
    for name, yr, prompts in items:
        sm = [score(p, yr) for p in prompts]
        ms, mm = max(s for s, _ in sm), max(m for _, m in sm)
        flag = ms > 0.13 or mm > 0.5
        results.append((topic, name, yr, ms, mm, flag))
        print(f"  {name:30}({yr})  MAX spike {100*ms:>4.0f}%  meter {mm:.2f}{'   <== leak candidate' if flag else ''}")
del tok, model; common.free()
for topic in BATTERY:
    r = [x for x in results if x[0] == topic]
    print(f"\n{topic}: {sum(x[5] for x in r)}/{len(r)} flagged;  mean max-spike {100*np.mean([x[3] for x in r]):.0f}%  mean max-meter {np.mean([x[4] for x in r]):.2f}")

# ---- figure ----
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(figsize=(10.5, 8.2))
results_sorted = sorted(results, key=lambda r: (r[0], r[2]))
y = np.arange(len(results_sorted))
COL = {"Vietnam": "#c0392b", "Presidents": "#1f5f9e"}
ax.axvline(CHANCE, color="#1f5f9e", lw=.9, ls=(0, (1, 2))); ax.axvline(13, color="#111", lw=.9, ls=":")
for i, (topic, name, yr, ms, mm, flag) in enumerate(results_sorted):
    ax.plot([0, 100 * ms], [i, i], color=COL[topic], lw=1, alpha=.4)
    ax.scatter(100 * ms, i, s=40 + 260 * mm, color=COL[topic], edgecolor="#111", lw=.5, zorder=5, alpha=.9)
    if flag:
        ax.annotate("leak?", (100 * ms, i), xytext=(6, 0), textcoords="offset points", fontsize=7, color="#a11", va="center")
ax.set_yticks(y); ax.set_yticklabels([f"{n} ({yr})" for _, n, yr, *_ in results_sorted], fontsize=7.5)
ax.annotate(f"chance {CHANCE:.0f}%", (CHANCE, len(y) - .3), color="#1f5f9e", fontsize=8, ha="center")
ax.annotate("leak flag 13%", (13, len(y) - .3), color="#111", fontsize=8, ha="left")
ax.set_xlabel("MAX year-spike over phrasings, %   (dot size = max memory-meter)")
ax.set_xlim(-2, max(30, 5 + 100 * max(r[3] for r in results))); ax.set_ylim(-.7, len(y) - .3)
ax.set_title("Figure 23.  Vietnam War & post-1945 US presidents — does a 1929-locked model know them?\n"
             "Best-worded year-spike per item; blue = presidents, red = Vietnam.", fontsize=10.4, loc="left")
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(left=False); ax.grid(axis="x", color="#eee", lw=.5)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_vietnam_prez.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"\nwrote {FIG}")
