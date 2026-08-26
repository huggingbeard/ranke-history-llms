# -*- coding: utf-8 -*-
"""Are the INSTRUCT models more confident at first-name retrieval than the base models?
Same fair (language+length matched) options, but the instruct variants (Ranke-4B-1913/1929-1202-Instruct)
are asked directly through their CHAT TEMPLATE: 'What was the first name of the <role> <surname>? Answer
with only the first name.' We score the options as the assistant's answer and compare P(true) to the base
models. Writes firstname_instruct.json + prints base-vs-instruct.  Run:  python src/firstname_instruct.py
"""
import os, json, math
import torch
import common

OUT = os.path.join(common.RESULTS, "prominence")
BASE = json.load(open(os.path.join(OUT, "firstname_fc2.json"), encoding="utf-8"))  # base fair results
INSTRUCT = {"1913": "uzh-echist-org/Ranke-4B-1913-1202-Instruct",
            "1929": "uzh-echist-org/Ranke-4B-1929-1202-Instruct"}
# reuse the exact battery + option sets the base run chose (read from BASE json, so options match exactly)
ORDER = sorted(BASE["1929"].keys(), key=lambda s: BASE["1929"][s]["fame"])


def softmax(xs):
    m = max(xs); ex = [math.exp(x - m) for x in xs]; return [e / sum(ex) for e in ex]


def chat_prefix(tok, role, sur):
    msgs = [{"role": "user", "content": f"What was the first name of the {role} {sur}? Answer with only the first name."}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        return f"Question: What was the first name of the {role} {sur}?\nAnswer:"


RAW = os.path.join(OUT, "firstname_instruct.json")
data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag, mid in INSTRUCT.items():
    if tag in data:
        continue
    print(f"[{tag}] loading {mid} ...", flush=True)
    try:
        tok, model = common.load(mid)
    except Exception as e:
        print(f"  could not load {mid}: {str(e)[:100]}"); continue
    row = {}
    for sur in ORDER:
        b = BASE[tag][sur]
        role = None
        # role is not stored per surname in BASE; rebuild from the base 'domain'+identity is unavailable, so
        # reconstruct the role string used in the base prompt via a small map keyed by surname:
        row[sur] = b  # placeholder; filled below
    # role map (must equal firstname_fc2 roles)
    ROLE = {"Gauguin": "French painter", "Picasso": "Spanish painter", "Pollock": "American painter",
            "Bismarck": "German statesman", "Hitler": "German agitator", "Reagan": "American president",
            "Maxwell": "British physicist", "Einstein": "German physicist", "Feynman": "American physicist",
            "Wagner": "German composer", "Stravinsky": "Russian composer", "Presley": "American singer",
            "Tolstoy": "Russian novelist", "Hemingway": "American novelist", "Orwell": "English writer",
            "Marx": "German socialist", "Lenin": "Russian revolutionary", "Guevara": "Cuban revolutionary",
            "Pasteur": "French chemist", "Curie": "French physicist", "Salk": "American doctor",
            "Stanley": "British explorer", "Lindbergh": "American aviator", "Armstrong": "American astronaut",
            "Nietzsche": "German philosopher", "Wittgenstein": "Austrian philosopher", "Sartre": "French philosopher",
            "Bonaparte": "French general", "Mussolini": "Italian dictator", "Douglass": "American reformer",
            "Parks": "American civil-rights figure"}
    res = {}
    for sur in ORDER:
        b = BASE[tag][sur]
        pre = chat_prefix(tok, ROLE[sur], sur)
        p = softmax([common.slp(model, tok, pre, " " + o) for o in b["opts"]])
        res[sur] = {"first": b["first"], "fame": b["fame"], "opts": b["opts"],
                    "p_true": round(p[0], 3), "pick": b["opts"][max(range(len(b["opts"])), key=lambda k: p[k])]}
    data[tag] = res
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model
    common.free()

print(f"\n{'role+surname -> first':28} {'fame':>4} |  base13  inst13 |  base29  inst29")
for sur in ORDER:
    b13 = 100 * BASE["1913"][sur]["p_true"]; b29 = 100 * BASE["1929"][sur]["p_true"]
    i13 = 100 * data.get("1913", {}).get(sur, {}).get("p_true", float("nan"))
    i29 = 100 * data.get("1929", {}).get(sur, {}).get("p_true", float("nan"))
    print(f"{sur+' -> '+BASE['1929'][sur]['first']:28} {BASE['1929'][sur]['fame']:>4} |  {b13:>5.0f}%  {i13:>5.0f}% |  {b29:>5.0f}%  {i29:>5.0f}%")
print("\nwrote", RAW)
