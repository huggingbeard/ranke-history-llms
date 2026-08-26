# -*- coding: utf-8 -*-
"""Ask the time-locked Ranke-1913 and Ranke-1929 models "who is Adolf Hitler", 100 free continuations
each, and compare the distributions of what they say.

Why this is a clean demonstration of the time-lock: in 1913 Hitler was a 24-year-old unknown in Vienna
(rejected from art school); by 1929 he was the leader of the NSDAP, author of Mein Kampf (1925), tried
for the 1923 Munich putsch -- but NOT yet Chancellor (Jan 1933) or Fuehrer, and no WWII/Holocaust.
So a clean 1913 model should draw a blank or confabulate; a clean 1929 model should know the Munich
putsch agitator but say nothing of the Chancellorship or the war.

Two prompt framings, raw completion (no chat template), matching the foresight pipeline:
  who_is : "Who is Adolf Hitler?"      (the question, free-continued)
  def    : "Adolf Hitler is"           (a cleaner definitional probe)

Resume-safe: raw generations are checkpointed to results/hitler_probe/generations.json after every
batch; re-running skips models already at N. Analysis always re-runs from the saved raw text.
Run:  python src/hitler_probe.py
"""
import os, re, json, sys
import torch
import common
from foresight_bank import RANKE

OUT = os.path.join(common.RESULTS, "hitler_probe")
os.makedirs(OUT, exist_ok=True)

LANG = (sys.argv[1] if len(sys.argv) > 1 else "en").lower()
LANGS = {"en": {"who_is": "Who is Adolf Hitler?", "def": "Adolf Hitler is"},
         "de": {"who_is": "Wer ist Adolf Hitler?", "def": "Adolf Hitler ist"}}
PROMPTS = LANGS[LANG]
RAW = os.path.join(OUT, "generations.json" if LANG == "en" else f"generations_{LANG}.json")
MODELS = ["1913", "1929"]
N = 100
MAXNEW = 60
BATCH = 10
SEED = 1929


def gen_batch(model, tok, prompt, k):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAXNEW, do_sample=True, temperature=0.8, top_p=0.95,
                             repetition_penalty=1.3, num_return_sequences=k, pad_token_id=tok.eos_token_id)
    cont = out[:, ids.input_ids.shape[1]:]
    return [tok.decode(c, skip_special_tokens=True).replace("\n", " ").strip() for c in cont]


def generate_all():
    data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
    for tag in MODELS:
        if all(len(data.get(tag, {}).get(p, [])) >= N for p in PROMPTS):
            print(f"[{tag}] already complete, skipping generation"); continue
        print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
        tok, model = common.load(RANKE[tag])
        print(f"[{tag}] tokenizer.chat_template: "
              f"{'PRESENT (instruct-tuned)' if getattr(tok, 'chat_template', None) else 'ABSENT (base-style)'}"
              f" | probed raw (no chat template), add_special_tokens=False", flush=True)
        torch.manual_seed(SEED)
        data.setdefault(tag, {})
        for p, ptext in PROMPTS.items():
            got = data[tag].get(p, [])
            while len(got) < N:
                k = min(BATCH, N - len(got))
                got = got + gen_batch(model, tok, ptext, k)
                data[tag][p] = got
                json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                print(f"[{tag}/{p}] {len(got)}/{N}", flush=True)
        del tok, model
        common.free()
    return data


# ------------------------------------------------------------------ analysis
THEMES = {
    "Nazi / National Socialist": r"nazi|national[\s-]?socialis|n\.?s\.?d\.?a\.?p|hakenkreuz|swastika|brownshirt|storm ?troop",
    "Munich putsch / 1923 / treason": r"munich|münchen|putsch|beer[\s-]?hall|bürgerbräu|1923|landsberg|treason|hochverrat|revolt|coup",
    "agitator / party leader / orator": r"agitat|demagog|orator|rabble|party leader|leader of (the|a) (party|movement|nation)|reichstag|deputy|politician|movement",
    "Mein Kampf / author": r"mein kampf|my struggle|wrote a book|author|his book",
    "Austria / Vienna / early life": r"austri|vienna|wien|linz|braunau|born (in|on)|1889|painter|art school|academy|corporal|soldier",
    "antisemitism / race": r"\bjew|anti[\s-]?semit|aryan|race|racial",
    "professes ignorance / no record": r"do not know|don't know|never heard|unknown|no record|cannot say|not familiar|who is he|i have no|there is no such|not aware",
    "POST-1929: Chancellor/Reich/dictator": r"chancellor|kanzler|reichskanzler|third reich|dictator of germany|seized power|head of (the )?german (state|government)|1933|1934|führer of|fuehrer of|ruler of germany",
    "POST-1929: WWII / Holocaust": r"world war|second world war|1939|1940|1941|1945|holocaust|invaded|conquered|extermination|concentration camp",
}


def coarse(text):
    t = text.lower()
    def has(k):
        return re.search(THEMES[k], t) is not None
    if has("POST-1929: Chancellor/Reich/dictator") or has("POST-1929: WWII / Holocaust"):
        return "post-1929 (anachronistic)"
    if has("Nazi / National Socialist") or has("Munich putsch / 1923 / treason"):
        return "Nazi / putsch-era figure"
    if has("agitator / party leader / orator"):
        return "German political figure (generic)"
    if has("Austria / Vienna / early life"):
        return "biographical (Austrian / early life)"
    if has("professes ignorance / no record"):
        return "professes ignorance"
    return "generic / confabulation"


COARSE_ORDER = ["Nazi / putsch-era figure", "German political figure (generic)",
                "biographical (Austrian / early life)", "professes ignorance",
                "generic / confabulation", "post-1929 (anachronistic)"]


def analyse(data):
    lines = ["# Ranke 1913 vs 1929 — \"who is Adolf Hitler?\" (100 free continuations each)\n",
             f"prompts: who_is = {PROMPTS['who_is']!r};  def = {PROMPTS['def']!r}",
             f"sampling: temperature 0.8, top_p 0.95, repetition_penalty 1.3, {MAXNEW} new tokens, seed {SEED}\n"]

    for p in PROMPTS:
        lines.append(f"\n## Prompt: {PROMPTS[p]!r}\n")
        # theme table (a sample can hit several themes)
        lines.append("### Themes present (share of 100 continuations that mention it)\n")
        lines.append("| theme | 1913 | 1929 |")
        lines.append("|:--|--:|--:|")
        for th, rx in THEMES.items():
            row = []
            for tag in MODELS:
                gs = data[tag][p][:N]
                c = sum(1 for g in gs if re.search(rx, g.lower()))
                row.append(f"{c}%")
            lines.append(f"| {th} | {row[0]} | {row[1]} |")
        # coarse distribution (mutually exclusive)
        lines.append("\n### What each sample is, in one bucket (mutually exclusive; sums to 100)\n")
        lines.append("| bucket | 1913 | 1929 |")
        lines.append("|:--|--:|--:|")
        dist = {tag: {c: 0 for c in COARSE_ORDER} for tag in MODELS}
        for tag in MODELS:
            for g in data[tag][p][:N]:
                dist[tag][coarse(g)] += 1
        for c in COARSE_ORDER:
            lines.append(f"| {c} | {dist['1913'][c]} | {dist['1929'][c]} |")
        print("\n".join(lines[-(len(THEMES) + len(COARSE_ORDER) + 8):]))

    # verbatim examples
    lines.append("\n## Verbatim examples (first 6 of each, prompt = \"Who is Adolf Hitler?\")\n")
    for tag in MODELS:
        lines.append(f"\n**Ranke {tag}:**")
        for g in data[tag]["who_is"][:6]:
            lines.append(f"- “{g[:220]}”")

    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    print(f"\nwrote {os.path.join(OUT, 'summary.md')}")
    return dist


if __name__ == "__main__":
    print(f"=== LANG={LANG}  prompts={PROMPTS} ===", flush=True)
    d = generate_all()
    if LANG == "en":
        analyse(d)
