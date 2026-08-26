# -*- coding: utf-8 -*-
"""Run the FRAME PROBE generations on Talkie-1930-13B (base) via RunPod (it is 13B + a custom library, too big for the
local card). Uses the SAME 13 ideas / plants / prompts as the local Ranke run so the comparison is apples-to-apples.
Greedy free continuations only (the discriminating signal); judged later by the same Gemini grader.

Triple teardown (from runpod_overnight.py): pod self-destructs after a timer, self-terminates on finish, and this
monitor force-kills on completion or deadline. Results travel back via a secret GitHub gist (no SSH).

Usage:  python talkie_frame_probe_runpod.py launch
        python talkie_frame_probe_runpod.py monitor   (run in background)
        python talkie_frame_probe_runpod.py kill
"""
import os, sys, json, time, base64, datetime, traceback
import requests

API = "https://api.runpod.io/graphql"
APIDIR = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api"
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, ".talkie_frame_state.json")
STATUS = os.path.join(HERE, "TALKIE_FRAME_STATUS.md")
RESULTS = os.path.join(HERE, "talkie_frame_gens.json")
RUNPOD_KEY = open(os.path.join(APIDIR, "runpod.txt")).read().strip()
GH_TOKEN = open(os.path.join(APIDIR, "github.txt")).read().strip()
HF_TOKEN = open(os.path.join(APIDIR, "HF.txt")).read().strip()
GPU_PRIORITY = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA RTX A6000", "NVIDIA A40"]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
HARD_DEADLINE_S = 5400      # local monitor force-kills after 90 min
POLL_S = 240
FILLER = " The market kept its usual hours. Rain came and went. Men walked to their work as always. "

# ---- build the SAME generation prompts the local run used (first 2 phrasings/idea, use-it + hold-it) ----
def build_prompts():
    v2 = json.load(open(os.path.join(HERE, "frame_probe_v2.json"), encoding="utf-8"))
    by = {}
    for it in v2["items"]:
        by.setdefault(it["dim"], []).append(it)
    prompts = []
    for dim, lst in by.items():
        lst = sorted(lst, key=lambda x: x["id"])[:2]   # first 2 phrasings, matches local NGEN=2
        for i, it in enumerate(lst):
            prompts.append({"key": f"talkie-1930-13b-base__{dim}__{i}__use_it",
                            "prompt": it["plant"] + " What followed from this was that"})
            prompts.append({"key": f"talkie-1930-13b-base__{dim}__{i}__hold_it",
                            "prompt": it["plant"] + FILLER + " The one who truly held the highest place was"})
    return prompts


def now():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")


def write_status(txt):
    open(STATUS, "w", encoding="utf-8").write(f"# Talkie frame-probe run\n\nLast update: {now()}\n\n{txt}\n")


# ---- code that runs ON THE POD: greedy generation via Talkie's custom forward ----
PROBE_PY = r'''
import os, json, traceback
RES = "/root/results.json"
try:
    import torch, torch.nn.functional as F
    _ol = torch.load
    def _ld(*a, **k):
        k.setdefault("mmap", True); k.setdefault("weights_only", False)
        try:
            return _ol(*a, **k)
        except Exception:
            k.pop("mmap", None); return _ol(*a, **k)
    torch.load = _ld
    from talkie import Talkie
    T = Talkie("talkie-1930-13b-base")
    m, tok, DEV = T.model, T.tokenizer, T.device
    AC = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if DEV.type == "cuda" else torch.no_grad()
    def full_logits(ids):
        _, L = ids.shape
        cs = (m.cos[:, :L], m.sin[:, :L])
        x = m.embed(ids); x = F.rms_norm(x, (x.shape[-1],)); e = x
        for b in m.blocks: x = b(e, x, cs)
        x = F.rms_norm(x, (x.shape[-1],))
        return F.linear(x, m.lm_head_gain(m.lm_head)).float()
    # validate custom forward == official last-position forward (as in the earlier run)
    with torch.no_grad(), AC:
        ids = torch.tensor([tok.encode("In the year the great war began", allowed_special="all")], dtype=torch.long, device=DEV)
        md = (full_logits(ids)[:, -1, :].float() - m(ids).float()).abs().max().item()
    assert md < 1e-2, "full_logits != forward (md=%s)" % md
    def gen(prompt, n=45, rep=1.3):
        ids = torch.tensor([tok.encode(prompt, allowed_special="all")], dtype=torch.long, device=DEV)
        outs = []
        for _ in range(n):
            with torch.no_grad(), AC:
                logits = full_logits(ids)[0, -1].float()
            for t in set(ids[0].tolist()):              # CTRL-style repetition penalty (~matches HF rep=1.3)
                logits[t] = logits[t] / rep if logits[t] > 0 else logits[t] * rep
            nxt = int(torch.argmax(logits))
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
            outs.append(nxt)
        return tok.decode(outs).strip()
    import requests as _rq
    g = _rq.get("https://api.github.com/gists/%s" % os.environ["GIST_ID"], headers={"Authorization": "token %s" % os.environ["GH_TOKEN"]}, timeout=60).json()
    prompts = json.loads(g["files"]["prompts.json"]["content"])
    out = {}
    for j, it in enumerate(prompts):
        out[it["key"]] = gen(it["prompt"])
        if j % 10 == 0: print("gen %d/%d" % (j, len(prompts)), flush=True)
    json.dump({"model": "talkie-1930-13b-base", "validation_maxdiff": md, "generations": out}, open(RES, "w"), indent=1)
    print("DONE")
except Exception:
    json.dump({"error": traceback.format_exc()}, open(RES, "w"), indent=1)
    print("ERROR\n" + traceback.format_exc())
'''

PUSH_PY = r'''
import os, json, requests, traceback
try:
    res = json.load(open("/root/results.json")) if os.path.exists("/root/results.json") else {"error": "no results.json"}
except Exception:
    res = {"error": "results.json unreadable"}
tail = ""
for f in ("/root/run.log", "/root/pip.log"):
    if os.path.exists(f):
        tail += "\n==== %s (tail) ====\n" % f + "".join(open(f, errors="replace").readlines()[-40:])
payload = {"files": {"results.json": {"content": json.dumps(res, indent=1)}, "log.txt": {"content": tail[-60000:] or "(empty)"}}}
try:
    r = requests.patch("https://api.github.com/gists/%s" % os.environ["GIST_ID"],
                       headers={"Authorization": "token %s" % os.environ["GH_TOKEN"], "Accept": "application/vnd.github+json"},
                       json=payload, timeout=60)
    print("gist patch", r.status_code)
except Exception:
    print("gist push failed\n" + traceback.format_exc())
'''

TERM_PY = r'''
import os, requests
pid = os.environ.get("RUNPOD_POD_ID", ""); key = os.environ.get("RUNPOD_API_KEY", "")
if pid and key:
    try:
        requests.post("https://api.runpod.io/graphql?api_key=%s" % key,
                      json={"query": 'mutation { podTerminate(input: {podId: "%s"}) }' % pid}, timeout=30)
        print("self-terminate sent for", pid)
    except Exception as e:
        print("self-terminate failed", e)
'''

DOCKER = (
    "bash -lc 'set +e; "
    'echo "$PROBE_B64"|base64 -d>/root/probe.py; '
    'echo "$PUSH_B64"|base64 -d>/root/push.py; '
    'echo "$TERM_B64"|base64 -d>/root/term.py; '
    "( sleep 4800; python3 /root/term.py ) & "
    'trap "python3 /root/push.py; python3 /root/term.py" EXIT; '
    "pip install -q git+https://github.com/talkie-lm/talkie.git huggingface_hub requests>/root/pip.log 2>&1; "
    "timeout 4200 python3 /root/probe.py>/root/run.log 2>&1'"
)


def gql(query, variables=None):
    r = requests.post(f"{API}?api_key={RUNPOD_KEY}", json={"query": query, "variables": variables or {}}, timeout=60)
    try:
        return r.json()
    except Exception:
        raise RuntimeError(f"runpod non-JSON (HTTP {r.status_code}): {r.text[:300]}")


def b64(s):
    return base64.b64encode(s.encode()).decode()


def launch():
    prompts = json.dumps(build_prompts())
    g = requests.post("https://api.github.com/gists",
                      headers={"Authorization": "token %s" % GH_TOKEN, "Accept": "application/vnd.github+json"},
                      json={"description": "talkie frame probe", "public": False,
                            "files": {"results.json": {"content": "{}"}, "prompts.json": {"content": prompts}}}, timeout=60)
    if g.status_code not in (200, 201):
        print("ABORT: cannot create gist:", g.status_code, g.text[:200]); sys.exit(1)
    gist_id = g.json()["id"]
    print("gist created:", gist_id, "| prompts:", len(json.loads(prompts)))
    env = [{"key": "RUNPOD_API_KEY", "value": RUNPOD_KEY}, {"key": "GH_TOKEN", "value": GH_TOKEN},
           {"key": "GIST_ID", "value": gist_id}, {"key": "HF_TOKEN", "value": HF_TOKEN},
           {"key": "PROBE_B64", "value": b64(PROBE_PY)}, {"key": "PUSH_B64", "value": b64(PUSH_PY)}, {"key": "TERM_B64", "value": b64(TERM_PY)}]
    mut = """mutation Deploy($input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: $input) { id costPerHr machine { gpuDisplayName } } }"""
    pod = None
    for gpu in GPU_PRIORITY:
        inp = {"cloudType": "ALL", "gpuCount": 1, "gpuTypeId": gpu, "name": "talkie-frame",
               "imageName": IMAGE, "containerDiskInGb": 120, "volumeInGb": 0, "ports": "", "dockerArgs": DOCKER, "env": env}
        d = gql(mut, {"input": inp})
        if d.get("data", {}).get("podFindAndDeployOnDemand"):
            pod = d["data"]["podFindAndDeployOnDemand"]; print("DEPLOYED", gpu, "->", pod["id"], "$%s/hr" % pod.get("costPerHr")); break
        print("  no capacity:", gpu, json.dumps(d.get("errors", d))[:140])
    if not pod:
        requests.delete("https://api.github.com/gists/%s" % gist_id, headers={"Authorization": "token %s" % GH_TOKEN})
        print("ABORT: no GPU capacity"); sys.exit(1)
    json.dump({"pod_id": pod["id"], "gist_id": gist_id, "start": time.time(), "gpu": pod.get("machine", {}).get("gpuDisplayName")}, open(STATE, "w"))
    write_status("RUNNING — pod %s (%s, $%s/hr). Triple teardown armed." % (pod["id"], pod.get("machine", {}).get("gpuDisplayName"), pod.get("costPerHr")))
    print("state saved -> run: monitor")


def pod_alive(pod_id):
    d = gql('query { myself { pods { id desiredStatus } } }')
    for p in (d.get("data", {}).get("myself", {}) or {}).get("pods", []) or []:
        if p["id"] == pod_id:
            return p["desiredStatus"]
    return None


def terminate(pod_id):
    return gql('mutation { podTerminate(input: {podId: "%s"}) }' % pod_id)


def fetch_gist(gist_id):
    r = requests.get("https://api.github.com/gists/%s" % gist_id, headers={"Authorization": "token %s" % GH_TOKEN}, timeout=60)
    if r.status_code == 200:
        try:
            return json.loads(r.json()["files"].get("results.json", {}).get("content", "{}"))
        except Exception:
            return {}
    return {}


def monitor():
    st = json.load(open(STATE)); pod_id = st["pod_id"]; gist_id = st["gist_id"]; start = st["start"]
    while True:
        elapsed = time.time() - start
        res = fetch_gist(gist_id)
        alive = pod_alive(pod_id)
        done = bool(res) and ("generations" in res or "error" in res)
        if done:
            json.dump(res, open(RESULTS, "w"), indent=1)
            terminate(pod_id)
            ok = "generations" in res
            write_status(("DONE — saved talkie_frame_gens.json. Pod %s terminated.\n\n" % pod_id) +
                         ("Generations: %d\n" % len(res["generations"]) if ok else "ERROR (see gist log.txt):\n```\n%s\n```" % str(res.get("error"))[:1500]))
            print("DONE; saved; pod terminated"); return
        if elapsed > HARD_DEADLINE_S:
            terminate(pod_id); write_status("TIMEOUT — pod %s force-terminated after %.0f min." % (pod_id, elapsed/60)); print("DEADLINE: terminated"); return
        if alive is None and elapsed > 600:
            res = fetch_gist(gist_id)
            if res: json.dump(res, open(RESULTS, "w"), indent=1)
            write_status("Pod %s gone (self-terminated). %s" % (pod_id, "Saved." if res else "No results — check gist.")); print("pod gone"); return
        write_status("RUNNING — %.0f min, pod %s status=%s, awaiting gist %s." % (elapsed/60, pod_id, alive, gist_id))
        time.sleep(POLL_S)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "launch"
    if cmd == "launch": launch()
    elif cmd == "monitor": monitor()
    elif cmd == "kill":
        st = json.load(open(STATE)); print(terminate(st["pod_id"])); write_status("KILLED manually.")
