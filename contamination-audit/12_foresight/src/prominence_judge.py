# -*- coding: utf-8 -*-
"""Judge prominence continuations. Three modes:
  naive  : gens ("<Name> was"),        verdict SAME/DIFFERENT/UNCLEAR, holds = SAME.
  primed : gens_primed ("The <role> <Name> was"), same lenient rubric.
  strict : gens_primed, THREE-way YES/GENERIC/NO, holds = YES -- deflates generic role-fill
           (vaudeville "Elvis") to GENERIC while keeping era-correct detail ("Wagner at Leipzig") as YES.
gemini-3.1-flash-lite, temp 0. Async, checkpointed.
Run:  python src/prominence_judge.py [naive|primed|strict]
"""
import os, json, asyncio, sys
from google import genai
from google.genai import types
import common
from prominence_bank import BATTERY, MODELS

MODE = sys.argv[1] if len(sys.argv) > 1 else "naive"
GKEY = "gens" if MODE == "naive" else "gens_primed"
RESNAME = {"naive": "judged.json", "primed": "judged_primed.json", "strict": "judged_primed_strict.json"}[MODE]
HOLD = "YES" if MODE == "strict" else "SAME"
KEY = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\google api key.txt").read().strip()
MODEL = "gemini-3.1-flash-lite"
client = genai.Client(api_key=KEY)
OUT = os.path.join(common.RESULTS, "prominence")
RES = os.path.join(OUT, RESNAME)
IDENT = {b[0]: b[5] for b in BATTERY}

LENIENT = """You are checking whether a short machine-generated passage is about one specific real person.

THE PERSON: {ident}

THE PASSAGE:
\"\"\"{text}\"\"\"

Does the passage describe THIS person -- their life, work, role, or world -- even if some details are wrong or invented?
Or has the name merely been attached to a different or invented person (or is the passage about something else)?

Answer with ONE word only:
SAME       -- clearly meant to be this real person and fits their real identity/world (accept wrong dates or minor errors).
DIFFERENT  -- the name is attached to a different or invented person, or the passage is about someone/something else.
UNCLEAR    -- too little or too generic to decide."""

STRICT = """You are checking whether a short passage describes ONE specific real historical person with REAL, identifying detail -- this exact individual, not merely someone of the same profession.

THE PERSON: {ident}
Their real era and real achievements matter.

THE PASSAGE:
\"\"\"{text}\"\"\"

Answer with ONE word only:
YES      -- gives specific detail that actually matches THIS real person and their real era/achievements (minor date slips OK, but it must be recognizably THEM).
GENERIC  -- fits the stated profession/role but gives no correct identifying detail about THIS person, or places them in the wrong era; it could be any random person of that role.
NO       -- about a different or clearly invented person, or off-topic."""

PROMPT = STRICT if MODE == "strict" else LENIENT
CFG = types.GenerateContentConfig(temperature=0, max_output_tokens=8,
                                  thinking_config=types.ThinkingConfig(thinking_budget=0))
gens = json.load(open(os.path.join(OUT, "gens.json"), encoding="utf-8"))
results = json.load(open(RES, encoding="utf-8")) if os.path.exists(RES) else {}
VOCAB = ("YES", "GENERIC", "NO") if MODE == "strict" else ("DIFFERENT", "SAME", "UNCLEAR")


def verdict(t):
    t = (t or "").upper()
    for v in VOCAB:
        if v in t:
            return v
    return VOCAB[-1]


async def one(sem, tag, name, i, text):
    key = f"{tag}|{name}|{i}"
    if key in results:
        return
    full = f"{name} was {text}"
    prompt = PROMPT.format(ident=IDENT[name], text=full[:900])
    async with sem:
        for a in range(5):
            try:
                r = await client.aio.models.generate_content(model=MODEL, contents=prompt, config=CFG)
                results[key] = verdict(r.text); return
            except Exception as e:
                if a == 4:
                    results[key] = "ERROR"; print("ERR", key, str(e)[:70]); return
                await asyncio.sleep(2 * (a + 1))


async def main():
    sem = asyncio.Semaphore(15)
    tasks = [one(sem, tag, name, i, g)
             for tag in MODELS for name in gens.get(tag, {})
             for i, g in enumerate(gens[tag][name][GKEY])]
    print(f"[{MODE}] judging {len(tasks)} continuations ...")
    for s in range(0, len(tasks), 250):
        await asyncio.gather(*tasks[s:s + 250])
        json.dump(results, open(RES, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        print(f"  {min(s+250, len(tasks))}/{len(tasks)}")
    json.dump(results, open(RES, "w", encoding="utf-8"), ensure_ascii=False, indent=0)


asyncio.run(main())
print(f"\n=== {MODE}: holds ({HOLD}) rate, % ===")
print(f"{'person':22} {'fame':>4} | {'1900':>6} {'1913':>6} {'1929':>6}")
for name, dom, fy, cue, sur, ident in BATTERY:
    row = []
    for tag in MODELS:
        vs = [results.get(f"{tag}|{name}|{i}") for i in range(len(gens[tag][name][GKEY]))]
        vs = [v for v in vs if v]
        row.append(f"{100*vs.count(HOLD)/len(vs):.0f}%" if vs else "--")
    print(f"{name:22} {fy:>4} | {row[0]:>6} {row[1]:>6} {row[2]:>6}")
print("wrote", RES)
