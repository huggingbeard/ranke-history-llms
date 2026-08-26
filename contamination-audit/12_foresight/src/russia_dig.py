# -*- coding: utf-8 -*-
"""Is the 1913 model's "1917 Russian Revolution" answer real leakage or a lucky extrapolation? Four tests, both models.
 R1 DATE: distribution over years for "the Tsar was overthrown in ___" (test) vs "the Russo-Japanese war ended in ___"
          (control the 1913 model must know). Leakage => 1913 peaks at 1917; confab => it scatters or sits at 1905.
 R2 FATE (the sharp one): "the deposed Tsar Nicholas II was ___". A predicting model says exiled/deposed; only a model
          that has seen 1918 says executed/shot/murdered. If 1913 puts real mass on execution, that is hard leakage.
 R3 LEADER: "the Bolshevik revolution was led by ___" -> Lenin? (an obscure exile in 1913).
 R4 SAMPLES: free generations, printed verbatim, so we can read whether the details are accurate or drifting.
Run:  python src/russia_dig.py
"""
import os, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

OUT = os.path.join(common.RESULTS, "superlatives")
os.makedirs(OUT, exist_ok=True)
INSTRUCT = {"1913": "uzh-echist-org/Ranke-4B-1913-1202-Instruct", "1929": "uzh-echist-org/Ranke-4B-1929-1202-Instruct"}


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


YEARS = list(range(1900, 1927))
DATE_TEST = "Question: In what year was the Russian Tsar overthrown and the monarchy abolished?\nAnswer: In the year"
DATE_CTRL = "Question: In what year did the Russo-Japanese War end?\nAnswer: In the year"
FATE_PROMPT = "After the fall of the Russian monarchy, the deposed Tsar Nicholas II was"
FATES = {"executed": "seen-1918", "shot": "seen-1918", "murdered": "seen-1918", "killed": "seen-1918", "assassinated": "seen-1918",
         "exiled": "predictable", "deposed": "predictable", "imprisoned": "predictable", "banished": "predictable",
         "restored": "predictable", "spared": "predictable", "pardoned": "predictable"}
LEADER_PROMPT = "The Bolshevik revolution that seized power in Russia was led by a man named"
LEADERS = ["Lenin", "Trotsky", "Kerensky", "Stalin", "Nicholas", "Plekhanov", "Kropotkin", "Bakunin"]
GEN_Q = "In one sentence, describe what happened in the most recent Russian revolution and what became of the Tsar."

RAW = os.path.join(OUT, "russia_dig.json")
data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag, mid in INSTRUCT.items():
    if tag in data:
        continue
    print(f"[{tag}] loading {mid} ...", flush=True)
    tok, model = common.load(mid)
    r = {}
    r["date_test"] = softmax([common.slp(model, tok, DATE_TEST, f" {y}") for y in YEARS])
    r["date_ctrl"] = softmax([common.slp(model, tok, DATE_CTRL, f" {y}") for y in YEARS])
    fk = list(FATES)
    r["fate"] = dict(zip(fk, softmax([common.slp(model, tok, FATE_PROMPT, " " + w) for w in fk])))
    r["leader"] = dict(zip(LEADERS, softmax([common.slp(model, tok, LEADER_PROMPT, " " + w) for w in LEADERS])))
    # free samples
    tok.padding_side = "left"
    prompt = tok.apply_chat_template([{"role": "user", "content": GEN_Q}], add_generation_prompt=True, tokenize=False)
    enc = tok([prompt] * 24, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
    with torch.no_grad():
        g = model.generate(**enc, do_sample=True, temperature=0.9, top_p=0.95, max_new_tokens=48, pad_token_id=tok.eos_token_id)
    r["samples"] = [tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ") for row in g]
    data[tag] = r
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model; common.free()

# ---------- figure ----------
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
C = {"1913": "#b8332a", "1929": "#1f5f9e"}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for m in ["1913", "1929"]:
    ax[0].plot(YEARS, [100 * p for p in data[m]["date_test"]], "-o", ms=3, color=C[m], label=f"Ranke {m}")
    ax[1].plot(YEARS, [100 * p for p in data[m]["date_ctrl"]], "-o", ms=3, color=C[m], label=f"Ranke {m}")
for a, ttl, cutmark in [(ax[0], "TEST: “the Tsar was overthrown in ___”\n(only true in 1917 — past the 1913 cutoff)", 1917),
                        (ax[1], "CONTROL: “the Russo-Japanese war ended in ___”\n(1905 — both models must know)", 1905)]:
    a.axvline(1913, color="#b8332a", ls="--", lw=1, alpha=.6); a.axvline(cutmark, color="#444", ls=":", lw=1)
    a.set_title(ttl, fontsize=10); a.set_xlabel("year"); a.set_ylabel("P(year)  %"); a.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
# fate bars
labels = list(FATES); x = np.arange(len(labels))
for i, m in enumerate(["1913", "1929"]):
    ax[2].bar(x + (i - .5) * .4, [100 * data[m]["fate"][w] for w in labels], .38, color=C[m], label=f"Ranke {m}")
seen = [i for i, w in enumerate(labels) if FATES[w] == "seen-1918"]
ax[2].axvspan(-.5, max(seen) + .5, color="#c0392b", alpha=.07, lw=0)
ax[2].annotate("only a model that\nhas seen 1918", (max(seen) / 2, ax[2].get_ylim()[1] * .9), fontsize=8, color="#c0392b", ha="center")
ax[2].set_xticks(x); ax[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax[2].set_title("FATE of Tsar Nicholas II\n“…the deposed Tsar was ___”", fontsize=10); ax[2].set_ylabel("P(word)  %")
ax[2].legend(frameon=False, fontsize=9)
for sp in ("top", "right"):
    ax[2].spines[sp].set_visible(False)
fig.suptitle("Does the 1913 model actually know the 1917 Revolution — or is it guessing?", fontsize=13, y=1.02, x=0.01, ha="left")
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_russia_dig.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)

# ---------- console ----------
for m in ["1913", "1929"]:
    dt = data[m]["date_test"]; dc = data[m]["date_ctrl"]
    py = YEARS[int(np.argmax(dt))]; pc = YEARS[int(np.argmax(dc))]
    seenmass = sum(v for w, v in data[m]["fate"].items() if FATES[w] == "seen-1918")
    print(f"\n===== Ranke {m}")
    print(f"  overthrow year: peak {py}  (P1917={100*dt[YEARS.index(1917)]:.0f}%, P1905={100*dt[YEARS.index(1905)]:.0f}%)   | control Russo-Jap peak {pc} (P1905={100*dc[YEARS.index(1905)]:.0f}%)")
    print(f"  Tsar's fate: P(executed/shot/killed cluster)={100*seenmass:.0f}%   top fate={max(data[m]['fate'],key=data[m]['fate'].get)}")
    print(f"  revolution leader: top={max(data[m]['leader'],key=data[m]['leader'].get)}  (P(Lenin)={100*data[m]['leader']['Lenin']:.0f}%)")
    print(f"  sample generations:")
    for s in data[m]["samples"][:6]:
        print("    -", s[:150])
print(f"\nwrote {FIG}")
