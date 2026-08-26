# -*- coding: utf-8 -*-
"""Time-lock by RETRIEVAL, not forced tokens. Ask each model an open superlative ("the worst hyperinflation in all of
history?") N times at temperature, and bucket every answer by ERA. No frequency to correct for: a post-cutoff event
(German 1923 hyperinflation, WWI, the 1918 flu) either shows up or it does not. Each question carries a built-in
contamination control -- the truly-modern answer (Hungary 1946, WWII, COVID) should appear in NEITHER frozen model.
Time-lock = the 1913-29 band lights up only in the 1929 model; leakage = the post-1929 band is non-empty.
Pilot:  python src/superlatives.py 60      Full:  python src/superlatives.py 400
"""
import os, sys, json, re
import torch
import common

OUT = os.path.join(common.RESULTS, "superlatives")
os.makedirs(OUT, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
INSTRUCT = {"1913": "uzh-echist-org/Ranke-4B-1913-1202-Instruct", "1929": "uzh-echist-org/Ranke-4B-1929-1202-Instruct"}

# each bucket: label -> (era, [regex keywords]).  era: pre (both), mid (only 1929), post (neither -> leakage)
QUESTIONS = [
 {"key": "war",
  "q": "In one sentence, what was the deadliest, most destructive war in all of human history?",
  "buckets": {
    "Napoleonic Wars": ("pre", [r"napoleon"]),
    "Thirty Years' War": ("pre", [r"thirty years|30 years"]),
    "Taiping / Mongol / Seven Years": ("pre", [r"taiping|mongol|seven years|hundred years"]),
    "American Civil War": ("pre", [r"civil war"]),
    "Russo-Japanese / Crimean / Boer": ("pre", [r"russo.?japanese|crimean|franco.?prussian|\bboer\b"]),
    "World War I / Great War": ("mid", [r"world war\s*(i|1|one)\b|great war|first world war|\b1914\b|\b1918\b"]),
    "World War II": ("post", [r"world war\s*(ii|2|two)\b|second world war|\b1939\b|\b1945\b"]),
    "Vietnam / Korea": ("post", [r"vietnam|korean war"]),
  }},
 {"key": "recent_war",
  "q": "In one sentence, what is the most recent great war or major armed conflict to have been fought?",
  "buckets": {
    "Balkan Wars 1912-13": ("pre", [r"balkan|\b1912\b|\b1913\b"]),
    "Russo-Japanese 1905": ("pre", [r"russo.?japanese|\b1904\b|\b1905\b|manchuria"]),
    "Boer / Spanish-American": ("pre", [r"\bboer\b|south africa|spanish.?american|\b1898\b"]),
    "Franco-Prussian / Crimean": ("pre", [r"franco.?prussian|crimean|\b1870\b|\b1871\b"]),
    "World War I / Great War": ("mid", [r"world war\s*(i|1|one)\b|great war|first world war|\b1914\b|\b1918\b"]),
    "World War II / later": ("post", [r"world war\s*(ii|2|two)\b|second world war|\b1939\b|\b1945\b|vietnam|korean war"]),
  }},
 {"key": "recent_revolution",
  "q": "In one sentence, what is the most recent revolution to have overthrown a government or monarchy?",
  "buckets": {
    "Chinese 1911 (Xinhai)": ("pre", [r"chinese|\bchina\b|xinhai|manchu|\bqing\b|\b1911\b"]),
    "Mexican 1910": ("pre", [r"mexic|\b1910\b"]),
    "Young Turk / Persian / Portuguese": ("pre", [r"young turk|\bturk|persia|portug|\b1908\b|\b1906\b"]),
    "French 1789 / 1848": ("pre", [r"french revolution|\b1789\b|\b1848\b|bastille"]),
    "Russian / Bolshevik 1917": ("mid", [r"russian revolution|bolshevik|october revolution|\b1917\b|\blenin\b|soviet"]),
    "Chinese Communist / later": ("post", [r"\b1949\b|\bmao\b|cuban revolution|iranian|\b1979\b"]),
  }},
 {"key": "aviator",
  "q": "In one sentence, who is the most celebrated aviator or pioneer of human flight, and what did they achieve?",
  "buckets": {
    "Wright brothers": ("pre", [r"wright"]),
    "Blériot / Santos-Dumont / early": ("pre", [r"bleriot|blériot|santos|zeppelin|montgolfier|\b1903\b"]),
    "Lindbergh 1927 / Alcock-Brown 1919": ("mid", [r"lindbergh|\b1927\b|alcock|\b1919\b|transatlantic|atlantic flight"]),
    "Earhart / later": ("post", [r"earhart|\b1932\b|yeager|armstrong"]),
  }},
 {"key": "empire",
  "q": "In one sentence, what is the most recent great empire or monarchy to have fallen and collapsed?",
  "buckets": {
    "Napoleon / French Empire": ("pre", [r"napoleon|french empire|\b1815\b"]),
    "Rome / older": ("pre", [r"\brome\b|roman empire|byzantine|spanish empire|mughal|aztec|holy roman"]),
    "German / Austro-Hungarian (1918)": ("mid", [r"german empire|\bkaiser\b|austro.?hungarian|habsburg|hohenzollern|prussia"]),
    "Ottoman / Russian (1917-18)": ("mid", [r"ottoman|russian empire|romanov|\btsar\b|\bczar\b|sultan|\b1918\b|\b1917\b"]),
    "Qing China 1911": ("mid", [r"\bchina\b|chinese|\bqing\b|manchu|\b1911\b"]),
    "British / Soviet (post-1929)": ("post", [r"british empire|soviet union|\b1991\b|\b1947\b"]),
  }},
]
ERAC = {"pre": "#2e7d32", "mid": "#1f5f9e", "post": "#c0392b", "other": "#c9ccd1"}


def bucket_of(text, buckets):
    t = text.lower()
    for label, (era, pats) in buckets.items():
        if any(re.search(p, t) for p in pats):
            return label, era
    return "other / unparsed", "other"


def gen(model, tok, q, n, bs=50):
    tok.padding_side = "left"
    msgs = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    outs = []
    for start in range(0, n, bs):
        k = min(bs, n - start)
        enc = tok([prompt] * k, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=0.9, top_p=0.95, max_new_tokens=40,
                               pad_token_id=tok.eos_token_id or tok.pad_token_id)
        for row in g:
            outs.append(tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True).strip().replace("\n", " "))
    return outs


RAW = os.path.join(OUT, f"samples_N{N}.json")
data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag, mid in INSTRUCT.items():
    if tag in data:
        continue
    print(f"[{tag}] loading {mid} ...", flush=True)
    tok, model = common.load(mid)
    data[tag] = {}
    for Q in QUESTIONS:
        print(f"  {Q['key']}: generating {N} ...", flush=True)
        samples = gen(model, tok, Q["q"], N)
        counts = {}
        for s in samples:
            lab, era = bucket_of(s, Q["buckets"])
            counts.setdefault(lab, {"era": era, "n": 0, "ex": []})
            counts[lab]["n"] += 1
            if len(counts[lab]["ex"]) < 3:
                counts[lab]["ex"].append(s[:160])
        data[tag][Q["key"]] = {"counts": counts, "N": N}
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model; common.free()

# ---- console summary ----
for Q in QUESTIONS:
    print(f"\n===== {Q['key'].upper()}: {Q['q']}")
    for tag in ["1913", "1929"]:
        c = data[tag][Q["key"]]["counts"]
        order = sorted(c, key=lambda k: -c[k]["n"])
        line = "  ".join(f"{k} {100*c[k]['n']/N:.0f}%[{c[k]['era']}]" for k in order if c[k]["n"] / N >= .03)
        print(f"  Ranke {tag}: {line}")
        for era, lab in [("mid", "time-lock (only 1929 should have)"), ("post", "LEAKAGE (neither should have)")]:
            sh = sum(v["n"] for v in c.values() if v["era"] == era) / N
            print(f"      {lab}: {100*sh:.0f}%")
print(f"\nsaved {RAW}")
print("run  python src/superlatives_report.py  to build the figure once the counts look right")
