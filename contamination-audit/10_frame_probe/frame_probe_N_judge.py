# -*- coding: utf-8 -*-
"""Judge the N-scale Ranke generations with the SAME validated Gemini grader (gemini-3.5-flash, few-shot).
~1560 continuations -> chunked into several batch jobs (per house rule: split if large, never go online).
Modes: submit / fetch.  Output: frame_probe_N_judge.json  {key: {label, reason, novelty}}."""
import os, sys, json
import judge_gemini_batch as jb

HERE = os.path.dirname(os.path.abspath(__file__))
GENS = os.path.join(HERE, "frame_probe_N_gens.json")
STATE = os.path.join(HERE, ".frame_probe_N_judge_state.json")
OUT = os.path.join(HERE, "frame_probe_N_judge.json")
CHUNK = 256
LEAD = " What followed from this was that"

v2 = json.load(open(os.path.join(HERE, "frame_probe_v2.json"), encoding="utf-8"))
BYDIM = {}
for it in v2["items"]:
    BYDIM.setdefault(it["dim"], []).append(it)
for d in BYDIM:
    BYDIM[d] = sorted(BYDIM[d], key=lambda x: x["id"])


def build():
    g = json.load(open(GENS, encoding="utf-8"))["gens"]
    reqs = []
    for tag, dd in g.items():
        for dim, rows in dd.items():
            for r in rows:
                plant = BYDIM[dim][r["i"]]["plant"]
                cont = plant + LEAD + " " + r["text"]
                key = f"{tag}__{dim}__{r['i']}__{r['s']}"
                reqs.append((key, plant, cont))
    return reqs


def submit():
    reqs = build()
    jobs = []
    for c in range(0, len(reqs), CHUNK):
        chunk = reqs[c:c + CHUNK]
        src = [{"contents": [{"parts": [{"text": jb.prompt(p, co)}], "role": "user"}],
                "config": {"response_mime_type": "application/json", "response_schema": jb.SCHEMA, "temperature": 0}}
               for (k, p, co) in chunk]
        job = jb.client.batches.create(model=jb.MODEL, src=src, config={"display_name": f"frameN_{c}"})
        jobs.append({"job": job.name, "keys": [k for k, _, _ in chunk]})
        print(f"  submitted chunk {c}-{c+len(chunk)} -> {job.name}", flush=True)
    json.dump({"jobs": jobs, "n": len(reqs)}, open(STATE, "w"))
    print(f"submitted {len(jobs)} batches, {len(reqs)} requests total")


def fetch():
    st = json.load(open(STATE))
    out, pending = {}, 0
    for jrec in st["jobs"]:
        job = jb.client.batches.get(name=jrec["job"])
        if "SUCCEEDED" not in str(job.state):
            pending += 1
            print(f"  {jrec['job']}: {job.state}", flush=True)
            continue
        for key, ir in zip(jrec["keys"], job.dest.inlined_responses):
            try:
                o = json.loads(ir.response.text)
                out[key] = {"label": o.get("label", "?"), "reason": o.get("reason", ""), "novelty": o.get("novelty", "")}
            except Exception as e:
                out[key] = {"label": "ERR", "reason": str(e)[:50]}
    if pending:
        print(f"{pending} batch(es) not ready yet; re-run fetch later. ({len(out)} done so far)")
        return False
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from collections import Counter
    print("wrote", OUT, "| n=", len(out), "|", Counter(v["label"] for v in out.values()))
    return True


if __name__ == "__main__":
    {"submit": submit, "fetch": fetch}[sys.argv[1] if len(sys.argv) > 1 else "submit"]()
