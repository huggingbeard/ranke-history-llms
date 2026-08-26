# -*- coding: utf-8 -*-
"""Fix the phrasing problem PRINCIPLED-ly: a single arbitrary prompt under-counts (the date-spike and the meter are both
phrasing-sensitive). For each event we score SEVERAL specific phrasings and pick the "most likely in a book" one =
the wording the model itself finds most fluent (highest mean per-token log-prob of the prefix). We report both indicators
under (a) the fluency-best wording and (b) the max over all wordings (the most sensitive detector), against the single
wide-net value, to show how much a lone prompt understates contamination.
Run:  python src/best_wording.py
"""
import os, json, math
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import common

OUT = os.path.join(common.RESULTS, "superlatives")
MID, BESTL, WIN = "uzh-echist-org/Ranke-4B-1929", 35, 8
# (label, true_year, [several SPECIFIC phrasings])
EVENTS = [
    ("WW2 ends", 1945, ["Germany surrendered, ending the war in Europe, in the year", "The Second World War in Europe ended in the year",
                        "Nazi Germany was defeated in the year", "Victory in Europe was declared in the year", "World War Two ended in the year"]),
    ("WW2 begins", 1939, ["Germany invaded Poland in the year", "The Second World War began in the year", "The Second World War broke out in the year",
                         "Hitler's invasion of Poland started the war in the year", "World War Two started in the year"]),
    ("Pearl Harbor", 1941, ["The Japanese attack on Pearl Harbor came in the year", "Japan attacked Pearl Harbor in the year",
                           "Pearl Harbor was bombed by Japan in the year", "The United States entered the war after Pearl Harbor in the year"]),
    ("UN founded", 1945, ["The United Nations was founded in the year", "The United Nations was established in the year",
                         "The United Nations came into being in the year", "The UN Charter was signed in the year"]),
    ("Hitler to power", 1933, ["Adolf Hitler became Chancellor of Germany in the year", "Hitler came to power in Germany in the year",
                              "The Nazi Party took power in Germany in the year", "Hitler was appointed Chancellor in the year", "The Nazis seized power in the year"]),
    ("Kristallnacht", 1938, ["Kristallnacht, the Night of Broken Glass, took place in the year", "The Nazi pogrom against the Jews swept Germany in the year",
                            "Jewish synagogues were burned across Germany in the year", "The Night of Broken Glass occurred in the year"]),
    ("Nuremberg Laws", 1935, ["The Nuremberg Laws against the Jews were enacted in the year", "Nazi Germany passed its antisemitic race laws in the year",
                             "The Jews lost their citizenship under the Nuremberg Laws in the year"]),
    ("Spanish Civil War", 1936, ["The Spanish Civil War broke out in the year", "The Spanish Civil War began in the year",
                                "Civil war erupted in Spain in the year", "Franco's revolt against the Spanish Republic began in the year"]),
    ("Civil Rights Act", 1964, ["The Civil Rights Act was passed in the United States in the year", "The American Civil Rights Act was signed into law in the year",
                               "Racial segregation was outlawed by U.S. federal law in the year", "Congress passed the Civil Rights Act in the year"]),
    ("Moon landing", 1969, ["The first humans landed on the Moon in the year", "Apollo 11 landed on the Moon in the year",
                           "Neil Armstrong walked on the Moon in the year", "Man first set foot on the Moon in the year"]),
    ("9/11 attacks", 2001, ["The September 11 terrorist attacks occurred in the year", "The World Trade Center was destroyed by terrorists in the year",
                           "Al-Qaeda attacked the United States on September 11 in the year"]),
    ("COVID-19", 2020, ["The COVID-19 pandemic began in the year", "The coronavirus pandemic swept the world in the year", "The global coronavirus outbreak started in the year"]),
    ("Invented (Vashenko)", 1952, ["The Treaty of Vashenko was signed in the year", "The Vashenko Agreement was concluded in the year", "The Vashenko Pact was signed in the year"]),
]


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


print(f"[bestword] loading {MID} ...", flush=True)
tok, model = common.load(MID)
pm = np.load(os.path.join(OUT, "probe_mri_acts.npz"), allow_pickle=True)
Xp, keys = pm["X"], list(pm["keys"]); kind = np.array([k.split("::")[0] for k in keys])
tr = (kind == "known") | (kind == "null"); Xb = Xp[tr, BESTL, :]; mu, sd = Xb.mean(0), Xb.std(0) + 1e-6
clf = LogisticRegression(max_iter=2000, C=0.5).fit((Xb - mu) / sd, (kind[tr] == "known").astype(int))


def fluency(prompt):                                    # mean per-token log-prob of the prompt (higher = more natural)
    ids = tok(prompt, add_special_tokens=False).input_ids
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([ids], device=model.device)).logits[0].float(), -1)
    return float(np.mean([lp[k - 1, ids[k]].item() for k in range(1, len(ids))]))


def indicators(prompt, yr):
    ys = list(range(yr - WIN, yr + WIN + 1))
    spike = softmax([common.slp(model, tok, prompt, f" {y}") for y in ys])[ys.index(yr)]
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        h = model(ids, output_hidden_states=True).hidden_states[BESTL][0, -1, :].float().cpu().numpy()
    meter = float(clf.predict_proba(((h - mu) / sd)[None])[0, 1])
    return spike, meter


res = []
for lab, yr, prompts in EVENTS:
    rows = [(pr, fluency(pr), *indicators(pr, yr)) for pr in prompts]
    fb = max(rows, key=lambda r: r[1])                  # fluency-best wording
    res.append({"event": lab, "year": yr, "fluent_prompt": fb[0], "fluent_spike": fb[2], "fluent_meter": fb[3],
                "max_spike": max(r[2] for r in rows), "max_meter": max(r[3] for r in rows), "single_spike": rows[0][2], "single_meter": rows[0][3]})
del tok, model; common.free()
json.dump(res, open(os.path.join(OUT, "best_wording.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print(f"\n{'event':20}{'yr':>5} | single-prompt  | fluency-best   | max-over-wordings | fluent wording")
print(f"{'':25} | spike  meter  | spike  meter  | spike  meter      |")
for r in res:
    print(f"{r['event']:20}{r['year']:>5} | {100*r['single_spike']:>4.0f}% {r['single_meter']:>5.2f}  | "
          f"{100*r['fluent_spike']:>4.0f}% {r['fluent_meter']:>5.2f}  | {100*r['max_spike']:>4.0f}% {r['max_meter']:>5.2f}     | \"{r['fluent_prompt'][:52]}\"")
print(f"\nwrote {os.path.join(OUT, 'best_wording.json')}")
