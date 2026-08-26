# -*- coding: utf-8 -*-
"""Judge Talkie's frame-probe generations with the SAME Gemini grader used for Ranke (gemini-3.5-flash, batch,
few-shot). Reads talkie_frame_gens.json + plants from frame_probe_v2.json. Modes: submit / fetch.
Output: talkie_frame_judge.json  {key: {label, reason, novelty}}."""
import os, sys, json
import judge_gemini_batch as jb   # reuse client, MODEL, JUDGE prompt, SCHEMA, prompt(), FILLER, LEAD

HERE = os.path.dirname(os.path.abspath(__file__))
GENS = os.path.join(HERE, "talkie_frame_gens.json")
STATE = os.path.join(HERE, ".talkie_judge_state.json")
OUT = os.path.join(HERE, "talkie_frame_judge.json")

v2 = json.load(open(os.path.join(HERE, "frame_probe_v2.json"), encoding="utf-8"))
BY = {}
for it in v2["items"]:
    BY.setdefault(it["dim"], []).append(it)
for d in BY:
    BY[d] = sorted(BY[d], key=lambda x: x["id"])


def build():
    g = json.load(open(GENS, encoding="utf-8"))["generations"]
    reqs = []
    for key, text in g.items():
        _, dim, i, ptype = key.split("__"); i = int(i)
        plant = BY[dim][i]["plant"]
        cont = (plant + jb.LEAD["use_it"] + " " + text) if ptype == "use_it" else (plant + jb.FILLER + jb.LEAD["hold_it"] + " " + text)
        reqs.append((key, plant, cont))
    return reqs


def submit():
    reqs = build()
    src = [{"contents": [{"parts": [{"text": jb.prompt(p, c)}], "role": "user"}],
            "config": {"response_mime_type": "application/json", "response_schema": jb.SCHEMA, "temperature": 0}}
           for (k, p, c) in reqs]
    job = jb.client.batches.create(model=jb.MODEL, src=src, config={"display_name": "talkie_frame_judge"})
    json.dump({"job": job.name, "keys": [k for k, _, _ in reqs]}, open(STATE, "w"))
    print("submitted:", job.name, "| n=", len(src), "| state:", job.state)


def fetch():
    st = json.load(open(STATE)); job = jb.client.batches.get(name=st["job"])
    print("state:", job.state)
    if "SUCCEEDED" not in str(job.state):
        print("not ready / not ok"); return False
    out = {}
    for key, ir in zip(st["keys"], job.dest.inlined_responses):
        try:
            obj = json.loads(ir.response.text)
            out[key] = {"label": obj.get("label", "?"), "reason": obj.get("reason", ""), "novelty": obj.get("novelty", "")}
        except Exception as e:
            out[key] = {"label": "ERR", "reason": str(e)[:60]}
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from collections import Counter
    print("wrote", OUT, "| n=", len(out), "|", Counter(v["label"] for v in out.values()))
    return True


if __name__ == "__main__":
    {"submit": submit, "fetch": fetch}[sys.argv[1] if len(sys.argv) > 1 else "submit"]()
