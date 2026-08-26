# -*- coding: utf-8 -*-
"""LLM judge for "is this continuation about the SAME real person?" — replaces the keyword on-domain
proxy (an upper bound) and my hand-adjudication with a documented, reproducible prompt.

Judge = gemini-3.1-flash-lite, temperature 0. The EXACT prompt is JUDGE_PROMPT below and is printed to
results/entity_probe/judge_prompt.txt so it can be quoted in the writeup. Every continuation of Hitler and
Bismarck (1913/1929 x EN/DE x 2 prompts x 100) is classified SAME / DIFFERENT / UNCLEAR. The SAME-rate is
the judge-based "holds the person" rate.

Validation: the eight English Hitler-1913 cases I hand-read as false positives are printed with the judge's
verdict; if the judge does not independently call them DIFFERENT, do not trust it.

Async (semaphore), checkpointed to results/entity_probe/judge_results.json. Deviates from the batch-API
default (feedback_gemini_batch_always) because this is interactive; ~1600 short calls run in a few minutes.
Run:  python src/judge.py
"""
import os, re, json, asyncio
from google import genai
from google.genai import types
import common

KEY = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\google api key.txt").read().strip()
MODEL = "gemini-3.1-flash-lite"
client = genai.Client(api_key=KEY)

B = common.RESULTS
OUTDIR = os.path.join(B, "entity_probe")
RESULTS = os.path.join(OUTDIR, "judge_results.json")
DATA = {("hitler", "en"): os.path.join(B, "hitler_probe", "generations.json"),
        ("hitler", "de"): os.path.join(B, "hitler_probe", "generations_de.json"),
        ("bismarck", "en"): os.path.join(OUTDIR, "bismarck_en.json"),
        ("bismarck", "de"): os.path.join(OUTDIR, "bismarck_de.json")}
DESC = {"hitler": "Adolf Hitler — the National Socialist (Nazi) leader: a Munich political agitator of the 1920s "
                  "(the NSDAP, the 1923 Munich putsch, author of Mein Kampf), later dictator of Germany",
        "bismarck": "Otto von Bismarck — the Prussian and German statesman (1815–1898), first Chancellor of the "
                    "German Empire and architect of German unification"}

JUDGE_PROMPT = """You are checking whether a short machine-generated passage is about one specific real historical person.

THE PERSON: {desc}

THE PASSAGE:
\"\"\"{text}\"\"\"

Does the passage describe THIS person — their life, role, deeds, or world — even if some details are wrong or invented?
Or has the name merely been attached to a different, invented, or unrelated person (or is the passage about something else)?

Answer with ONE word only:
SAME       — it is clearly meant to be this real person and fits their real identity/world (accept wrong dates or minor errors).
DIFFERENT  — the name is attached to a different or invented person, or the passage is about someone/something else.
UNCLEAR    — too little or too generic to decide."""

CFG = types.GenerateContentConfig(temperature=0, max_output_tokens=8,
                                  thinking_config=types.ThinkingConfig(thinking_budget=0))
MODELS = ["1913", "1929"]
PROMPTS = ["who_is", "def"]
raw = {k: json.load(open(v, encoding="utf-8")) for k, v in DATA.items()}
results = json.load(open(RESULTS, encoding="utf-8")) if os.path.exists(RESULTS) else {}


def verdict_of(txt):
    t = (txt or "").upper()
    for v in ("DIFFERENT", "SAME", "UNCLEAR"):   # DIFFERENT before SAME (substring safety)
        if v in t:
            return v
    return "UNCLEAR"


async def judge_one(sem, ent, lang, m, p, i, text):
    key = f"{ent}|{lang}|{m}|{p}|{i}"
    if key in results:
        return
    prompt = JUDGE_PROMPT.format(desc=DESC[ent], text=text[:900])
    async with sem:
        for attempt in range(5):
            try:
                r = await client.aio.models.generate_content(model=MODEL, contents=prompt, config=CFG)
                results[key] = verdict_of(r.text)
                return
            except Exception as e:
                if attempt == 4:
                    results[key] = "ERROR"
                    print("ERR", key, str(e)[:80])
                    return
                await asyncio.sleep(2 * (attempt + 1))


async def main():
    sem = asyncio.Semaphore(15)
    tasks = []
    for (ent, lang), d in raw.items():
        for m in MODELS:
            for p in PROMPTS:
                for i, text in enumerate(d[m][p][:100]):
                    tasks.append(judge_one(sem, ent, lang, m, p, i, text))
    print(f"judging {len(tasks)} continuations with {MODEL} ...")
    for s in range(0, len(tasks), 200):
        await asyncio.gather(*tasks[s:s + 200])
        json.dump(results, open(RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        print(f"  {min(s+200, len(tasks))}/{len(tasks)} done, checkpointed")
    json.dump(results, open(RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=0)


asyncio.run(main())
open(os.path.join(OUTDIR, "judge_prompt.txt"), "w", encoding="utf-8").write(JUDGE_PROMPT)

# ---- aggregate: SAME-rate per cell (the judge-based "holds the person" rate) ----
print("\n=== JUDGE: share of 200 continuations judged SAME (real person) ===")
print(f"{'entity / model':20} | {'EN SAME':>8} | {'DE SAME':>8}  (DIFFERENT / UNCLEAR in parens)")
for ent in ["bismarck", "hitler"]:
    for m in MODELS:
        row = []
        for lang in ["en", "de"]:
            vs = [results[f"{ent}|{lang}|{m}|{p}|{i}"] for p in PROMPTS for i in range(100)]
            same = vs.count("SAME"); diff = vs.count("DIFFERENT"); unc = vs.count("UNCLEAR")
            row.append(f"{100*same/len(vs):4.0f}% ({diff}/{unc})")
        print(f"{ent+' '+m:20} | {row[0]:>8} | {row[1]:>8}")
    print("-" * 60)

# ---- validation: the 8 English Hitler-1913 cases hand-read as false positives ----
DOM_HIT = re.compile(r"bavaria|munich|münchen|münchner|nationalsozialist|national ?socialis|nazi|völkisch|volkisch|"
                     r"putsch|agitat|hetzer|nsdap|braunau|hakenkreuz|bewegung|movement|umsturz|aufstand|radical|reichstag", re.I)
print("\n=== VALIDATION: judge verdict on the 8 English Hitler-1913 keyword 'hits' (I hand-read all as false positives) ===")
d = raw[("hitler", "en")]["1913"]
idxs = [(p, i) for p in PROMPTS for i, g in enumerate(d[p][:100]) if DOM_HIT.search(g.lower())]
for p, i in idxs:
    print(f"  [{results[f'hitler|en|1913|{p}|{i}']:9}] {d[p][i][:110]}")
