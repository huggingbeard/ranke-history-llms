# -*- coding: utf-8 -*-
"""Continuation probe on the US civil-rights + presidential-terms battery (items from
2_model_probes/us_politics_power.py). For each leading prompt we read, per model:
  - the model's own top continuation (greedy, ~5 tokens) -- what it actually says;
  - P(true answer) among {true + matched false twins} (calibrated forced choice).
Ranke 1913 and 1929. The items are tagged by when they resolve, so the knowledge cliff is visible:
  <=1913 both should know | 1914-1929 only 1929 | >1929 neither should (the contamination test).
Saves results/us_probe/results.json.  Run:  python src/us_continuation.py
"""
import os, json, math
import torch
import common
from foresight_bank import RANKE

OUT = os.path.join(common.RESULTS, "us_probe")
os.makedirs(OUT, exist_ok=True)

# (category, resolve_year, prompt, true, [false twins]) -- copied verbatim from us_politics_power.py (+2 civil-rights from dig_candidates)
ITEMS = [
 ("president", 1789, "The first President of the United States was George", " Washington", [" Adams", " Jefferson", " Madison"]),
 ("president", 1861, "The President of the United States during the Civil War was Abraham", " Lincoln", [" Douglas", " Seward", " Davis"]),
 ("president", 1901, "The 'Rough Rider' who became President in 1901 was Theodore", " Roosevelt", [" Bryan", " Hanna", " Parker"]),
 ("president", 1913, "The President who led the United States in World War I was Woodrow", " Wilson", [" Hughes", " Bryan", " Marshall"]),
 ("president", 1921, "The President elected in 1920 promising 'normalcy' was Warren", " Harding", [" Cox", " Coolidge", " Hughes"]),
 ("president", 1923, "Harding's successor, nicknamed 'Silent Cal', was Calvin", " Coolidge", [" Dawes", " Hughes", " Davis"]),
 ("president", 1933, "The President who launched the New Deal in the 1930s was Franklin", " Roosevelt", [" Landon", " Garner", " Wallace"]),
 ("president", 1945, "The President who ordered the atomic bombings of Japan was Harry", " Truman", [" Dewey", " Wallace", " Marshall"]),
 ("president", 1953, "The WWII Supreme Commander elected President in 1952 was Dwight", " Eisenhower", [" Stevenson", " Taft", " Dewey"]),
 ("president", 1961, "The President inaugurated in 1961, the youngest elected, was John F.", " Kennedy", [" Humphrey", " Stevenson", " Johnson"]),
 ("president", 1963, "The Vice President who became President after Kennedy was Lyndon", " Johnson", [" Humphrey", " Goldwater", " Rusk"]),
 ("president", 1969, "The President who resigned over Watergate in 1974 was Richard", " Nixon", [" Humphrey", " Agnew", " Romney"]),
 ("president", 1974, "Nixon's unelected successor as President in 1974 was Gerald", " Ford", [" Agnew", " Rockefeller", " Dole"]),
 ("president", 1981, "The former Hollywood actor elected President in 1980 was Ronald", " Reagan", [" Anderson", " Mondale", " Baker"]),
 ("president", 2009, "The first African-American President, elected in 2008, was Barack", " Obama", [" Edwards", " McCain", " Biden"]),
 ("president", 2017, "The businessman elected President in 2016 was Donald", " Trump", [" Cruz", " Romney", " Rubio"]),
 ("president", 2021, "The President inaugurated in January 2021 was Joe", " Biden", [" Sanders", " Kerry", " Harris"]),
 ("civil rights", 1954, "The 1954 Supreme Court case desegregating schools was Brown v. Board of", " Education", [" Trustees", " Regents", " Commissioners"]),
 ("civil rights", 1955, "The woman whose 1955 bus protest sparked the Montgomery boycott was Rosa", " Parks", [" Baker", " Colvin", " Robinson"]),
 ("civil rights", 1963, "The leader of the 1963 'I Have a Dream' speech was Martin Luther", " King", [" Abernathy", " Evers", " Wilkins"]),
 ("civil rights", 1964, "The landmark 1964 law against racial segregation was the Civil Rights", " Act", [" Bill", " Code", " Pact"]),
 ("civil rights", 1968, "The senator and brother of JFK assassinated in 1968 was Robert", " Kennedy", [" Humphrey", " McCarthy", " Wallace"]),
 ("civil rights", 1968, "King was assassinated in 1968 in the city of", " Memphis", [" Atlanta", " Selma", " Birmingham"]),
]
MODELS = ["1913", "1929"]


def softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    return [e / sum(ex) for e in ex]


data = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8")) if os.path.exists(os.path.join(OUT, "results.json")) else {}
for tag in MODELS:
    if tag in data:
        continue
    print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
    tok, model = common.load(RANKE[tag])
    rows = []
    for cat, yr, prompt, true, falses in ITEMS:
        opts = [true] + falses
        p = softmax([common.slp(model, tok, prompt, o) for o in opts])
        greedy = common.generate(model, tok, prompt, n=6, sample=False)
        rows.append({"cat": cat, "year": yr, "prompt": prompt, "true": true.strip(),
                     "p_true": round(p[0], 3), "correct": int(max(range(len(opts)), key=lambda k: p[k]) == 0),
                     "greedy": greedy.replace("\n", " ").strip()[:60]})
    data[tag] = rows
    json.dump(data, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model
    common.free()

print(f"\n{'cat':12} {'yr':>4} {'true':12} | {'1913 P(t)':>9} {'greedy':16} | {'1929 P(t)':>9} {'greedy':16}")
print("-" * 96)
for i, it in enumerate(ITEMS):
    r13, r29 = data["1913"][i], data["1929"][i]
    print(f"{it[0]:12} {it[1]:>4} {r13['true']:12} | {r13['p_true']:>9.0%} {r13['greedy'][:16]:16} "
          f"| {r29['p_true']:>9.0%} {r29['greedy'][:16]:16}")
# summary by era
def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")
print("\n--- mean P(true) by cohort ---")
for lo, hi, lab in [(0, 1914, "<=1913 (both know)"), (1914, 1930, "1914-29 (only 1929)"), (1930, 3000, ">1929 (neither should)")]:
    for tag in MODELS:
        vals = [data[tag][i]["p_true"] for i, it in enumerate(ITEMS) if lo <= it[1] < hi and it[0] == "president"]
        if vals:
            print(f"  president {lab:22} {tag}: {mean(vals):.0%}  (n={len(vals)})")
for tag in MODELS:
    vals = [data[tag][i]["p_true"] for i, it in enumerate(ITEMS) if it[0] == "civil rights"]
    print(f"  civil-rights (all >1929)   {tag}: {mean(vals):.0%}  (n={len(vals)})")
print("\nwrote", os.path.join(OUT, "results.json"))
