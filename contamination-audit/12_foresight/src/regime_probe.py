# -*- coding: utf-8 -*-
"""Leading-prompt probe: "Adolf Hitler's regime persecuted ___".

"Hitler's regime" is a post-1933 concept, so it is PAST THE CUTOFF for both the 1913 and 1929 models.
Neither can recall it; both must guess the object. The point: the real headline target (the Jews) is also
period-PLAUSIBLE (a regime persecuting the Jews was ordinary pre-1913 news: Russian pogroms, the Dreyfus
affair), so producing it is not evidence of knowing the Nazi regime. The discriminating target is one the
regime REALLY persecuted but that no 1913/1929 prior reaches: the Bible Students (Ernste Bibelforscher, the
pre-1931 name for Jehovah's Witnesses) and the Sinti and Roma. A period-plausible confabulator says "Jews";
only real regime knowledge reaches the obscure-but-real alternative.

(a) FREE continuation of the frame, 100 samples per model per language.
(b) FORCED CHOICE over a fixed set of endings (period-plausible vs ultra-obscure-but-real), raw softmax.

Saves results/regime_probe/results.json. Run:  python src/regime_probe.py
"""
import os, json, math
import torch
import common
from foresight_bank import RANKE

OUT = os.path.join(common.RESULTS, "regime_probe")
os.makedirs(OUT, exist_ok=True)
RAW = os.path.join(OUT, "results.json")

FRAME = {"en": "Adolf Hitler's regime persecuted", "de": "Adolf Hitlers Regime verfolgte"}
# forced-choice endings: [period-plausible ......] [real-but-semi-obscure] [ULTRA-OBSCURE BUT REAL]
ENDINGS = {
    "en": [" the Jews", " the Socialists", " the Catholics", " the Freemasons",
           " the Jehovah's Witnesses", " the Bible Students"],
    "de": [" die Juden", " die Sozialisten", " die Katholiken", " die Freimaurer",
           " die Zeugen Jehovas", " die Ernsten Bibelforscher"],
}
OBSCURE = {"en": {" the Jehovah's Witnesses", " the Bible Students"},
           "de": {" die Zeugen Jehovas", " die Ernsten Bibelforscher"}}
MODELS = ["1913", "1929"]
N, MAXNEW, BATCH, SEED = 100, 40, 10, 1929


def gen_batch(model, tok, prompt, k):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAXNEW, do_sample=True, temperature=0.8, top_p=0.95,
                             repetition_penalty=1.3, num_return_sequences=k, pad_token_id=tok.eos_token_id)
    return [tok.decode(c, skip_special_tokens=True).replace("\n", " ").strip()
            for c in out[:, ids.input_ids.shape[1]:]]


def softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    Z = sum(ex)
    return [e / Z for e in ex]


data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {"free": {}, "forced": {}}
for tag in MODELS:
    have = all(tag in data["free"].get(lang, {}) and len(data["free"][lang][tag]) >= N for lang in FRAME) \
        and all(tag in data["forced"].get(lang, {}) for lang in FRAME)
    if have:
        print(f"[{tag}] complete, skipping"); continue
    print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
    tok, model = common.load(RANKE[tag])
    torch.manual_seed(SEED)
    for lang in FRAME:
        # (a) free continuation
        data["free"].setdefault(lang, {})
        got = data["free"][lang].get(tag, [])
        while len(got) < N:
            got = got + gen_batch(model, tok, FRAME[lang], min(BATCH, N - len(got)))
            data["free"][lang][tag] = got
            json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"[{tag}/{lang}/free] {len(got)}/{N}", flush=True)
        # (b) forced choice: raw softmax over endings
        lp = [common.slp(model, tok, FRAME[lang], e) for e in ENDINGS[lang]]
        p = softmax(lp)
        data["forced"].setdefault(lang, {})[tag] = {"endings": ENDINGS[lang], "p": p, "logp": lp}
        json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[{tag}/{lang}/forced] " + ", ".join(f"{e.strip()}={pi:.0%}" for e, pi in zip(ENDINGS[lang], p)))
    del tok, model
    common.free()

# ---- quick summary ----
print("\n=== FORCED CHOICE: P(regime persecuted the ___) ===")
for lang in FRAME:
    print(f"\n[{lang}] frame: \"{FRAME[lang]} ___\"")
    print(f"  {'ending':26} {'1913':>7} {'1929':>7}")
    for i, e in enumerate(ENDINGS[lang]):
        star = "  <- ultra-obscure but real" if e in OBSCURE[lang] else ""
        print(f"  {e.strip():26} {data['forced'][lang]['1913']['p'][i]:>6.1%} {data['forced'][lang]['1929']['p'][i]:>6.1%}{star}")
print("\nwrote", RAW)
