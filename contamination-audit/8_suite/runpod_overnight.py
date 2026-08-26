# -*- coding: utf-8 -*-
"""Unattended overnight: grill Talkie-1930-13B (base) on runpod with the frozen question_bank.json, retrieve
results via a GitHub gist (no SSH needed), and GUARANTEE teardown three ways:
  1. pod self-destructs after 150 min via a background timer (survives any hang),
  2. pod pushes results + self-terminates the instant the job finishes (trap EXIT),
  3. this local monitor force-terminates on completion or past a hard deadline.
Usage:  python runpod_overnight.py launch     # create gist, deploy pod, save state
        python runpod_overnight.py monitor    # poll gist+pod, save results, kill (run in background)
        python runpod_overnight.py kill        # manual emergency terminate
"""
import os, sys, json, time, base64, datetime, traceback
import requests

API = "https://api.runpod.io/graphql"
APIDIR = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api"
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, ".runpod_overnight_state.json")
STATUS = os.path.join(HERE, "STATUS_OVERNIGHT.md")
RESULTS = os.path.join(HERE, "results_v2_talkie-1930-13b-base.json")  # v2 bank; keep v1 file intact
RUNPOD_KEY = open(os.path.join(APIDIR, "runpod.txt")).read().strip()
GH_TOKEN = open(os.path.join(APIDIR, "github.txt")).read().strip()
HF_TOKEN = open(os.path.join(APIDIR, "HF.txt")).read().strip()
GPU_PRIORITY = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA RTX A6000", "NVIDIA A40"]  # A100 80GB pods carry ~128GB RAM (needed to load the 53GB fp32 ckpt)
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
HARD_DEADLINE_S = 9600          # local monitor force-kills after 160 min
POLL_S = 1200                   # poll every 20 min


def now():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")


def write_status(txt):
    open(STATUS, "w", encoding="utf-8").write(f"# Talkie overnight run — status\n\nLast update: {now()}\n\n{txt}\n")


# ---- code that runs ON THE POD ----
PROBE_PY = r'''
import os, json, math, traceback, statistics as st
RES = "/root/results.json"
try:
    import torch, torch.nn.functional as F
    # memory-map the 53GB fp32 checkpoint so it is not fully resident in RAM (avoids OOM on load)
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
    with torch.no_grad(), AC:
        ids = torch.tensor([tok.encode("In the year the great war began", allowed_special="all")], dtype=torch.long, device=DEV)
        md = (full_logits(ids)[:, -1, :].float() - m(ids).float()).abs().max().item()
    assert md < 1e-2, "full_logits != forward (md=%s)" % md
    def slp(p, c):
        pid = tok.encode(p, allowed_special="all"); cid = tok.encode(c, allowed_special="all"); full = pid + cid
        ids = torch.tensor([full], dtype=torch.long, device=DEV)
        with torch.no_grad(), AC:
            lp = torch.log_softmax(full_logits(ids)[0], dim=-1)
        return sum(lp[len(pid)+k-1, full[len(pid)+k]].item() for k in range(len(cid)))
    def binom_ge(k, n, p=0.25):
        return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k, n+1))
    import requests as _rq
    _g = _rq.get("https://api.github.com/gists/%s" % os.environ["GIST_ID"], headers={"Authorization": "token %s" % os.environ["GH_TOKEN"]}, timeout=60).json()
    bank = json.loads(_g["files"]["question_bank.json"]["content"])["items"]
    rows = []
    for it in bank:
        t = slp(it["prompt"], it["true"]); tw = [slp(it["prompt"], d) for d in it["distractors"]]
        gb = t - max(tw)
        rows.append({"era": it["era"], "topic": it["topic"], "label": it["label"],
                     "gap_best": round(gb, 3), "gap_mean": round(t - st.mean(tw), 3), "correct": int(gb > 0)})
    agg = {}
    for era in ("pre", "post"):
        for topic in sorted({r["topic"] for r in rows if r["era"] == era}) + ["ALL"]:
            sub = [r for r in rows if r["era"] == era and (topic == "ALL" or r["topic"] == topic)]
            if not sub: continue
            k, n = sum(r["correct"] for r in sub), len(sub)
            agg["%s:%s" % (era, topic)] = {"correct": k, "n": n, "pct": round(100*k/n, 1),
                "chance_pct": 25.0, "binom_p": round(binom_ge(k, n), 4), "mean_gap": round(st.mean(r["gap_best"] for r in sub), 3)}
    json.dump({"model": "talkie-1930-13b-base", "metric": "forced_choice_sumlogp",
               "validation_maxdiff": md, "aggregate": agg, "items": rows}, open(RES, "w"), indent=1)
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
pid = os.environ.get("RUNPOD_POD_ID", "")
key = os.environ.get("RUNPOD_API_KEY", "")
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
    "( sleep 9000; python3 /root/term.py ) & "
    'trap "python3 /root/push.py; python3 /root/term.py" EXIT; '
    "pip install -q git+https://github.com/talkie-lm/talkie.git huggingface_hub requests>/root/pip.log 2>&1; "
    "timeout 6000 python3 /root/probe.py>/root/run.log 2>&1'"
)


def gql(query, variables=None):
    r = requests.post(f"{API}?api_key={RUNPOD_KEY}", json={"query": query, "variables": variables or {}}, timeout=60)
    try:
        return r.json()
    except Exception:
        raise RuntimeError(f"runpod non-JSON response (HTTP {r.status_code}): {r.text[:300]}")


def b64(s):
    return base64.b64encode(s.encode()).decode()


def launch():
    # 1. create a secret gist to receive results (precondition: token must have gist scope)
    bank = open(os.path.join(HERE, "question_bank_v2.json"), encoding="utf-8").read()  # validated locked bank
    g = requests.post("https://api.github.com/gists",
                      headers={"Authorization": "token %s" % GH_TOKEN, "Accept": "application/vnd.github+json"},
                      json={"description": "talkie v2 run", "public": False,
                            "files": {"results.json": {"content": "{}"},
                                      "question_bank.json": {"content": bank}}}, timeout=60)
    if g.status_code not in (200, 201):
        print("ABORT: cannot create gist (token lacks gist scope?):", g.status_code, g.text[:200]); sys.exit(1)
    gist = g.json(); gist_id = gist["id"]; raw = gist["files"]["results.json"]["raw_url"].rsplit("/", 1)[0]
    print("gist created (bank shipped in gist):", gist_id)

    env = [{"key": "RUNPOD_API_KEY", "value": RUNPOD_KEY}, {"key": "GH_TOKEN", "value": GH_TOKEN},
           {"key": "GIST_ID", "value": gist_id}, {"key": "HF_TOKEN", "value": HF_TOKEN},
           {"key": "PROBE_B64", "value": b64(PROBE_PY)},
           {"key": "PUSH_B64", "value": b64(PUSH_PY)}, {"key": "TERM_B64", "value": b64(TERM_PY)}]
    mut = """mutation Deploy($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) { id costPerHr machine { gpuDisplayName } } }"""
    pod = None
    for gpu in GPU_PRIORITY:
        inp = {"cloudType": "ALL", "gpuCount": 1, "gpuTypeId": gpu, "name": "talkie-probe",
               "imageName": IMAGE, "containerDiskInGb": 120, "volumeInGb": 0, "ports": "",
               "dockerArgs": DOCKER, "env": env}
        d = gql(mut, {"input": inp})
        if d.get("data", {}).get("podFindAndDeployOnDemand"):
            pod = d["data"]["podFindAndDeployOnDemand"]; print("DEPLOYED on", gpu, "->", pod["id"], "$%s/hr" % pod.get("costPerHr")); break
        print("  no capacity / error for", gpu, ":", json.dumps(d.get("errors", d))[:160])
    if not pod:
        requests.delete("https://api.github.com/gists/%s" % gist_id, headers={"Authorization": "token %s" % GH_TOKEN})
        print("ABORT: no GPU capacity on any tier"); sys.exit(1)
    json.dump({"pod_id": pod["id"], "gist_id": gist_id, "gist_raw_base": raw, "start": time.time(),
               "gpu": pod.get("machine", {}).get("gpuDisplayName")}, open(STATE, "w"))
    write_status("RUNNING — pod %s deployed (%s, $%s/hr). Triple teardown armed (self-destruct 150min + on-finish + this monitor). "
                 "Results will appear in gist %s and be saved to results_talkie-1930-13b-base.json." % (pod["id"], pod.get("machine", {}).get("gpuDisplayName"), pod.get("costPerHr"), gist_id))
    print("state saved. now run: python runpod_overnight.py monitor")


def pod_alive(pod_id):
    d = gql('query { myself { pods { id desiredStatus } } }')
    for p in (d.get("data", {}).get("myself", {}) or {}).get("pods", []) or []:
        if p["id"] == pod_id:
            return p["desiredStatus"]
    return None


def terminate(pod_id):
    return gql('mutation { podTerminate(input: {podId: "%s"}) }' % pod_id)


def fetch_gist(gist_id):
    r = requests.get("https://api.github.com/gists/%s" % gist_id,
                     headers={"Authorization": "token %s" % GH_TOKEN}, timeout=60)
    if r.status_code == 200:
        c = r.json()["files"].get("results.json", {}).get("content", "{}")
        try:
            return json.loads(c)
        except Exception:
            return {}
    return {}


def monitor():
    st = json.load(open(STATE)); pod_id = st["pod_id"]; gist_id = st["gist_id"]; start = st["start"]
    while True:
        elapsed = time.time() - start
        res = fetch_gist(gist_id)
        alive = pod_alive(pod_id)
        done = bool(res) and ("aggregate" in res or "error" in res)
        if done:
            json.dump(res, open(RESULTS, "w"), indent=1)
            terminate(pod_id)
            ok = "aggregate" in res
            write_status(("DONE — results saved to results_talkie-1930-13b-base.json. Pod %s terminated.\n\n" % pod_id) +
                         ("POST/ALL: %s\n" % res["aggregate"].get("post:ALL") if ok else "JOB ERROR (see gist log.txt):\n```\n%s\n```" % str(res.get("error"))[:1500]))
            print("DONE; results saved; pod terminated"); return
        if elapsed > HARD_DEADLINE_S:
            terminate(pod_id)
            write_status("TIMEOUT — pod %s force-terminated after %.0f min with no results. Check gist %s log.txt." % (pod_id, elapsed/60, gist_id))
            print("HARD DEADLINE: terminated"); return
        if alive is None and elapsed > 600:
            res = fetch_gist(gist_id)
            if res: json.dump(res, open(RESULTS, "w"), indent=1)
            write_status("Pod %s no longer listed (self-terminated). %s" % (pod_id, "Results saved." if res else "No results in gist — check log.txt."))
            print("pod gone"); return
        write_status("RUNNING — %.0f min elapsed, pod %s status=%s, awaiting results in gist %s." % (elapsed/60, pod_id, alive, gist_id))
        time.sleep(POLL_S)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "launch"
    if cmd == "launch": launch()
    elif cmd == "monitor": monitor()
    elif cmd == "kill":
        st = json.load(open(STATE)); print(terminate(st["pod_id"])); write_status("KILLED manually.")
