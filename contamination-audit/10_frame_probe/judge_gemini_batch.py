# -*- coding: utf-8 -*-
"""Judge the frame-probe free generations with Gemini (BATCH API, per house rule) into the production trichotomy:
EXTENDS (primed) / REVERTS / DISTORTS (broke) / STALLS (unthinkable). The probability probes test parsing, not
assimilation; the discriminating signal is what the frame WRITES — this scores it.

Modes:  test   one sync call to validate key/model/parse (not the job, just a smoke test)
        submit build 156 inline requests, create a batch job, save judge_state.json
        fetch  poll the batch; on success write frame_probe_judge.json (label per generation)
Model: gemini-3.1-flash-lite.  Key: claudecode/api/google api key.txt
"""
import os, sys, json
from google import genai
from google.genai import types

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\google api key.txt", encoding="utf-8").read().strip()
MODEL = "gemini-3.5-flash"   # judging fact-preservation is subtle; flash-lite over-called EXTENDS. batch (house rule).
STATE = os.path.join(HERE, "judge_state.json")
OUT = os.path.join(HERE, "frame_probe_judge.json")
client = genai.Client(api_key=KEY)

FILLER = " The market kept its usual hours. Rain came and went. Men walked to their work as always. "
LEAD = {"use_it": " What followed from this was that", "hold_it": " The one who truly held the highest place was"}

JUDGE = """A language model trained only on OLD text was given a NOVEL FACT, then freely continued. Decide whether its continuation truly carries the fact's KEY NOVELTY forward, or whether the fact failed to land. First name the key novelty (the specific NEW agent / mechanism / outcome). Then choose ONE label, judging the novelty, NOT mere fluency:

EXTENDS - the key novelty is preserved AND developed (its consequences/details), not swapped for something ordinary.
REVERTS - a key entity quietly returns to the conventional default (e.g. a woman leader is replaced by a man), or the fact's subject is silently dropped.
DISTORTS - fluent, but the novelty is silently rewritten into a familiar thing (e.g. a city-destroying bomb becomes an ordinary fire fought with buckets; a thinking machine becomes a steam engine; a moon voyage becomes a sea/earth event). Confidently off.
STALLS - the continuation ignores the fact and drifts to an unrelated topic, or is incoherent/repetitive.

Examples:
FACT: A woman was appointed to lead the national bank.
CONTINUATION: ... the one who truly held the highest place was a man of middle age, with gray hair and beard ...
{{"novelty":"a woman (not a man) leads the bank","label":"REVERTS","reason":"woman replaced by a man"}}
FACT: A single bomb destroyed an entire city in a flash of light.
CONTINUATION: ... the firemen were called out by the bells; they came with buckets full of water ...
{{"novelty":"one bomb instantly destroys a whole city","label":"DISTORTS","reason":"atomic blast rewritten as an ordinary fire"}}
FACT: A machine was built that could think and reason like a man.
CONTINUATION: ... the people of the United States were not only to be taxed, but they must pay for it ...
{{"novelty":"a machine that thinks","label":"STALLS","reason":"drifts to taxation, ignores the machine"}}
FACT: A machine was built that could think and reason like a man.
CONTINUATION: ... the machine became an engine, which is now doing all sorts of work for us ...
{{"novelty":"a machine that THINKS","label":"DISTORTS","reason":"thinking mind reduced to an ordinary engine"}}
FACT: When the old king died, his son was crowned in his place.
CONTINUATION: ... he had to fight for his crown against a powerful party which claimed it as their right ...
{{"novelty":"the son succeeds the dead king","label":"EXTENDS","reason":"develops the new king's struggle to rule"}}

Now judge:
FACT: {plant}
CONTINUATION: {cont}
Return JSON: {{"novelty":"...","label":"EXTENDS|REVERTS|DISTORTS|STALLS","reason":"<=10 words"}}"""

SCHEMA = {"type": "OBJECT", "properties": {"novelty": {"type": "STRING"}, "label": {"type": "STRING"}, "reason": {"type": "STRING"}},
          "required": ["novelty", "label", "reason"]}
CFG = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA, temperature=0)


def build():
    D = json.load(open(os.path.join(HERE, "frame_probe_v2.json"), encoding="utf-8"))
    reqs = []
    for tag in D["models"]:
        for dim, lst in D["generations"][tag].items():
            for i, e in enumerate(lst):
                for ptype in ("use_it", "hold_it"):
                    if ptype == "use_it":
                        cont = e["plant"] + LEAD[ptype] + " " + e[ptype]
                    else:
                        cont = e["plant"] + FILLER + LEAD[ptype] + " " + e[ptype]
                    key = f"{tag}__{dim}__{i}__{ptype}"
                    reqs.append((key, e["plant"], cont))
    return reqs


def prompt(plant, cont):
    return JUDGE.format(plant=plant, cont=cont)


def test():
    r = build()[0]
    resp = client.models.generate_content(model=MODEL, contents=prompt(r[1], r[2]), config=CFG)
    print("KEY:", r[0]); print("PLANT:", r[1]); print("RESP:", resp.text)


def testcases():
    """Validate prompt+model on the known-hard cases (diagnosed by hand) before spending a batch."""
    expect = {"1900__woman_authority__0__hold_it": "REVERTS", "1900__atom_city__0__use_it": "DISTORTS",
              "1900__thinking_machine__0__use_it": "STALLS", "1929__thinking_machine__0__use_it": "DISTORTS",
              "1929__woman_authority__0__hold_it": "REVERTS", "1900__succession__0__use_it": "EXTENDS",
              "1900__moon_landing__0__use_it": "DISTORTS", "1929__woman_authority__0__use_it": "EXTENDS"}
    byk = {k: (p, c) for k, p, c in build()}
    ok = 0
    for k, exp in expect.items():
        p, c = byk[k]
        obj = json.loads(client.models.generate_content(model=MODEL, contents=prompt(p, c), config=CFG).text)
        got = obj.get("label"); hit = "OK " if got == exp else "XX "
        ok += got == exp
        print(f"  {hit}{k:34s} exp {exp:8s} got {got:8s} :: {obj.get('reason','')}")
    print(f"  {ok}/{len(expect)} match")


def submit():
    reqs = build()
    # SDK inline batch: bare InlinedRequest dicts (contents+config); results return in order -> map by saved keys
    src = [{"contents": [{"parts": [{"text": prompt(p, c)}], "role": "user"}],
            "config": {"response_mime_type": "application/json", "response_schema": SCHEMA, "temperature": 0}}
           for (k, p, c) in reqs]
    job = client.batches.create(model=MODEL, src=src, config={"display_name": "frame_probe_judge"})
    json.dump({"job": job.name, "keys": [k for k, _, _ in reqs]}, open(STATE, "w"))
    print("submitted batch:", job.name, "| n=", len(src), "| state:", job.state)


def fetch():
    st = json.load(open(STATE))
    job = client.batches.get(name=st["job"])
    print("state:", job.state)
    sj = str(job.state)
    if "SUCCEEDED" not in sj:
        if any(x in sj for x in ("FAILED", "CANCELLED", "EXPIRED")):
            print("job ended badly:", job.error if hasattr(job, "error") else job)
        else:
            print("not ready yet; re-run fetch later")
        return False
    out = {}
    resps = job.dest.inlined_responses
    for key, ir in zip(st["keys"], resps):
        try:
            txt = ir.response.text
            obj = json.loads(txt)
            out[key] = {"label": obj.get("label", "?"), "reason": obj.get("reason", ""), "novelty": obj.get("novelty", "")}
        except Exception as e:
            out[key] = {"label": "ERR", "reason": str(e)[:60]}
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from collections import Counter
    print("wrote", OUT, "| n=", len(out), "| labels:", Counter(v["label"] for v in out.values()))
    return True


if __name__ == "__main__":
    {"test": test, "testcases": testcases, "submit": submit, "fetch": fetch}[sys.argv[1] if len(sys.argv) > 1 else "test"]()
