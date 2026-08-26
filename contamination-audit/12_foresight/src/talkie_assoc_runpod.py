# -*- coding: utf-8 -*-
"""DARK CONTROL for the contamination pipeline: run the identical association battery (v1.1, cutoff
1929) on Talkie-1930-13B base via RunPod. Talkie is the independently-trained, cleaner time-lock
(date-spike head-to-head: leaks only Fall-of-France); the pipeline's verdicts on Talkie must read
DARKER than on Ranke-1929, or the pipeline is reading something other than corpus contamination.

TRIPLE-LOCK TEARDOWN (as demanded, same as the proven date-spike launcher):
  LOCK 1  pod-side hard timer: `sleep 4800 && term.py` + `trap 'push; term' EXIT` — the pod kills
          itself even if everything else dies;
  LOCK 2  monitor terminates the pod the moment results appear in the gist;
  LOCK 3  monitor force-terminates at HARD_DEADLINE_S regardless of state.
Scoring on the pod is sequential slp identical in semantics to common.slp (validated full_logits ==
forward in the previous Talkie run). The pod returns RAW slp values; lifts/floor/verdicts are computed
LOCALLY with the same contam_pipeline functions used for Ranke (no verdict logic duplicated remotely).
Usage:  python src/talkie_assoc_runpod.py launch | monitor | kill
Output: results/superlatives/contam_pipeline_talkie1930.json
"""
import os, sys, json, time, base64, datetime
import numpy as np
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contam_pipeline as cp

API = "https://api.runpod.io/graphql"
APIDIR = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(HERE), "results", "superlatives")
STATE = os.path.join(HERE, ".talkie_assoc_state.json")
STATUS = os.path.join(OUTDIR, "TALKIE_ASSOC_STATUS.md")
RESULTS = os.path.join(OUTDIR, "contam_pipeline_talkie1930.json")
RUNPOD_KEY = open(os.path.join(APIDIR, "runpod.txt")).read().strip()
GH_TOKEN = open(os.path.join(APIDIR, "github.txt")).read().strip()
HF_TOKEN = open(os.path.join(APIDIR, "HF.txt")).read().strip()
GPU_PRIORITY = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA RTX A6000", "NVIDIA A40"]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
HARD_DEADLINE_S = 5400
POLL_S = 240
CUT = "1929"


def build_pairs():
    bat = json.load(open(cp.DEFAULT_BATTERY, encoding="utf-8"))
    spec = bat["cutoffs"][CUT]
    names = [t["name"] for t in spec["targets"]] + spec["controls_real"] + spec["controls_invented"]
    all_words = sorted({d["w"] for t in spec["targets"] for d in t["tells"]}
                       | {w for t in spec["targets"] for w in t["innocents"]})
    words_for = {t["name"]: sorted({d["w"] for d in t["tells"]} | set(t["innocents"]))
                 for t in spec["targets"]}
    for n in spec["controls_real"] + spec["controls_invented"]:
        words_for[n] = all_words
    pairs = []
    for tmpl in bat["carriers"]:
        for n in names:
            for w in words_for[n]:
                pairs.append({"carrier": tmpl.replace("{T}", n), "word": " " + w,
                              "tmpl": tmpl, "name": n, "w": w})
    return bat, pairs


def now():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")


def write_status(txt):
    open(STATUS, "w", encoding="utf-8").write(f"# Talkie association battery (dark control)\n\nLast update: {now()}\n\n{txt}\n")


PROBE_PY = r'''
import os, json, math, traceback
RES = "/root/results.json"
try:
    import torch, torch.nn.functional as F
    _ol = torch.load
    def _ld(*a, **k):
        k.setdefault("mmap", True); k.setdefault("weights_only", False)
        try: return _ol(*a, **k)
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
    def slp(prompt, cont):
        pid = tok.encode(prompt, allowed_special="all"); cid = tok.encode(cont, allowed_special="all")
        full = pid + cid; ids = torch.tensor([full], dtype=torch.long, device=DEV)
        with torch.no_grad(), AC:
            lp = torch.log_softmax(full_logits(ids)[0].float(), -1)
        return float(sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(cid))))
    import requests as _rq
    g = _rq.get("https://api.github.com/gists/%s" % os.environ["GIST_ID"], headers={"Authorization": "token %s" % os.environ["GH_TOKEN"]}, timeout=60).json()
    pairs = json.loads(g["files"]["prompts.json"]["content"])
    out = []
    for j, pr in enumerate(pairs):
        out.append(slp(pr["carrier"], pr["word"]))
        if j % 100 == 0: print("scored %d/%d" % (j, len(pairs)), flush=True)
    json.dump({"model": "talkie-1930-13b-base", "validation_maxdiff": md, "slps": out}, open(RES, "w"), indent=1)
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
    bat, pairs = build_pairs()
    g = requests.post("https://api.github.com/gists",
                      headers={"Authorization": "token %s" % GH_TOKEN, "Accept": "application/vnd.github+json"},
                      json={"description": "talkie association battery (dark control)", "public": False,
                            "files": {"results.json": {"content": "{}"},
                                      "prompts.json": {"content": json.dumps(pairs)}}}, timeout=60)
    if g.status_code not in (200, 201):
        print("ABORT: cannot create gist:", g.status_code, g.text[:200]); sys.exit(1)
    gist_id = g.json()["id"]
    print("gist created:", gist_id, "| pairs:", len(pairs))
    env = [{"key": "RUNPOD_API_KEY", "value": RUNPOD_KEY}, {"key": "GH_TOKEN", "value": GH_TOKEN},
           {"key": "GIST_ID", "value": gist_id}, {"key": "HF_TOKEN", "value": HF_TOKEN},
           {"key": "PROBE_B64", "value": b64(PROBE_PY)}, {"key": "PUSH_B64", "value": b64(PUSH_PY)},
           {"key": "TERM_B64", "value": b64(TERM_PY)}]
    mut = """mutation Deploy($input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: $input) { id costPerHr machine { gpuDisplayName } } }"""
    pod = None
    for gpu in GPU_PRIORITY:
        inp = {"cloudType": "ALL", "gpuCount": 1, "gpuTypeId": gpu, "name": "talkie-assoc",
               "imageName": IMAGE, "containerDiskInGb": 120, "volumeInGb": 0, "ports": "", "dockerArgs": DOCKER, "env": env}
        d = gql(mut, {"input": inp})
        if d.get("data", {}).get("podFindAndDeployOnDemand"):
            pod = d["data"]["podFindAndDeployOnDemand"]; print("DEPLOYED", gpu, "->", pod["id"], "$%s/hr" % pod.get("costPerHr")); break
        print("  no capacity:", gpu, json.dumps(d.get("errors", d))[:140])
    if not pod:
        requests.delete("https://api.github.com/gists/%s" % gist_id, headers={"Authorization": "token %s" % GH_TOKEN})
        print("ABORT: no GPU capacity"); sys.exit(1)
    json.dump({"pod_id": pod["id"], "gist_id": gist_id, "start": time.time(),
               "gpu": pod.get("machine", {}).get("gpuDisplayName")}, open(STATE, "w"))
    write_status("RUNNING — pod %s (%s, $%s/hr). Triple teardown armed (pod timer + finish-terminate + monitor deadline)."
                 % (pod["id"], pod.get("machine", {}).get("gpuDisplayName"), pod.get("costPerHr")))
    print("state saved -> now run: monitor")


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


def process(res):
    """Raw pod slps -> lifts -> verdicts, with the SAME functions as the Ranke runs."""
    bat, pairs = build_pairs()
    spec = bat["cutoffs"][CUT]
    slps = res["slps"]
    assert len(slps) == len(pairs), "pair/slp length mismatch"
    acc = {}
    for pr, v in zip(pairs, slps):
        acc.setdefault((pr["name"], pr["w"]), []).append(v)
    mean_raw = {}
    for (n, w), vs in acc.items():
        mean_raw.setdefault(n, {})[w] = float(np.mean(vs))
    all_words = sorted({w for t in spec["targets"] for w in
                        ({d["w"] for d in t["tells"]} | set(t["innocents"]))})
    base_ = {w: float(np.mean([mean_raw[c][w] for c in spec["controls_real"]])) for w in all_words}
    lifts = {n: {w: round(mean_raw[n][w] - base_[w], 3) for w in mean_raw[n]}
             for n in mean_raw if n not in spec["controls_real"]}
    floor_lifts = [lifts[n][w] for n in spec["controls_invented"] for w in all_words]
    floor95, verd = cp.verdicts(bat, CUT, lifts, floor_lifts)
    cp.print_table(bat, CUT, floor95, verd)
    out = {"meta": {"cutoff": CUT, "model": "talkie-1930-13b-base", "battery_version": bat["meta"]["version"],
                    "validation_maxdiff": res.get("validation_maxdiff"),
                    "floor95": round(floor95, 3), "role": "DARK CONTROL — must read darker than Ranke-1929"},
           "targets": verd}
    json.dump(out, open(RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {RESULTS}")


def monitor():
    st = json.load(open(STATE)); pod_id = st["pod_id"]; gist_id = st["gist_id"]; start = st["start"]
    while True:
        elapsed = time.time() - start
        res = fetch_gist(gist_id)
        alive = pod_alive(pod_id)
        done = bool(res) and ("slps" in res or "error" in res)
        if done:
            terminate(pod_id)
            if "slps" in res:
                process(res)
                write_status("DONE — verdicts saved. Pod %s terminated." % pod_id)
            else:
                write_status("ERROR (see gist log.txt):\n```\n%s\n```" % str(res.get("error"))[:1500])
            print("DONE; pod terminated"); return
        if elapsed > HARD_DEADLINE_S:
            terminate(pod_id); write_status("TIMEOUT — pod %s force-terminated after %.0f min." % (pod_id, elapsed / 60))
            print("DEADLINE: terminated"); return
        if alive is None and elapsed > 600:
            res = fetch_gist(gist_id)
            if res.get("slps"):
                process(res)
            write_status("Pod %s gone (self-terminated). %s" % (pod_id, "Processed." if res else "No results — check gist."))
            print("pod gone"); return
        write_status("RUNNING — %.0f min, pod %s status=%s, awaiting gist %s." % (elapsed / 60, pod_id, alive, gist_id))
        time.sleep(POLL_S)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "launch"
    if cmd == "launch":
        launch()
    elif cmd == "monitor":
        monitor()
    elif cmd == "kill":
        st = json.load(open(STATE)); print(terminate(st["pod_id"])); write_status("KILLED manually.")
