#!/usr/bin/env python3
"""RunPod A100-80GB manager for plan v2.4 (talkie-ranke-mutual-0928).

  create     : create one A100 80GB pod (SXM, then PCIe), wait for SSH, save state
  ssh CMD    : run CMD on the pod
  put L R    : copy local path to pod
  get R L    : copy pod path to local
  list       : list ALL pods on the account (teardown check)
  terminate  : terminate the pod in state, then verify the account pod list
"""
import json, subprocess, sys, time
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
STATE = HERE.parents[1] / "runs" / "pod_state.json"
KEY = Path("D:/UZHechist Dropbox/Joachim Voth/claudecode/api/runpod.txt").read_text().strip()
GQL = f"https://api.runpod.io/graphql?api_key={KEY}"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
PUB = SSH_KEY.with_suffix(".pub").read_text().strip()
OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6", "-i", str(SSH_KEY)]
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"


def gql(q, v=None):
    r = requests.post(GQL, json={"query": q, "variables": v or {}}, timeout=60)
    r.raise_for_status()
    d = r.json()
    if "errors" in d:
        raise RuntimeError(d["errors"])
    return d["data"]


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def create():
    if state().get("pod_id") and not state().get("terminated"):
        sys.exit(f"pod already in state: {state()}")
    last = None
    for gpu in ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"]:
        for cloud in ["SECURE", "COMMUNITY"]:
            m = f'''mutation {{ podFindAndDeployOnDemand(input: {{
              cloudType: {cloud}, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 200,
              minVcpuCount: 8, minMemoryInGb: 60, gpuTypeId: "{gpu}", name: "talkie-ranke-0928",
              imageName: "{IMAGE}", dockerArgs: "", ports: "22/tcp", volumeMountPath: "/workspace",
              startSsh: true, env: [{{ key: "PUBLIC_KEY", value: "{PUB}" }}] }}) {{ id costPerHr machineId }} }}'''
            try:
                d = gql(m)["podFindAndDeployOnDemand"]
                st = {"pod_id": d["id"], "gpu": gpu, "cloud": cloud, "cost_per_hr": d.get("costPerHr"),
                      "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                STATE.parent.mkdir(parents=True, exist_ok=True)
                STATE.write_text(json.dumps(st, indent=2))
                print("created", st, flush=True)
                break
            except Exception as e:
                last = e
                print("unavailable", gpu, cloud, str(e)[:200], flush=True)
        else:
            continue
        break
    else:
        raise SystemExit(f"no A100 80GB available: {last}")
    st = state()
    q = 'query P($id:String!){ pod(input:{podId:$id}){ id desiredStatus runtime{ uptimeInSeconds ports{ ip isIpPublic privatePort publicPort } } } }'
    for _ in range(120):
        time.sleep(10)
        try:
            p = gql(q, {"id": st["pod_id"]})["pod"]
        except Exception as e:
            print("api retry", e); continue
        for port in ((p.get("runtime") or {}).get("ports") or []):
            if port["privatePort"] == 22 and port["isIpPublic"]:
                st.update(ip=port["ip"], port=int(port["publicPort"]))
                STATE.write_text(json.dumps(st, indent=2))
                print("ssh", st["ip"], st["port"], flush=True)
                for _ in range(60):
                    r = subprocess.run(["ssh", f"root@{st['ip']}", "-p", str(st["port"])] + OPTS +
                                       ["-o", "ConnectTimeout=8", "echo ok"], capture_output=True, text=True)
                    if r.returncode == 0:
                        print("SSH READY", flush=True); return
                    time.sleep(5)
                raise SystemExit("ssh never answered")
        print("waiting", p.get("desiredStatus"), flush=True)
    raise SystemExit("pod not ready in 20 min")


def ssh(cmd, capture=False):
    st = state()
    c = ["ssh", f"root@{st['ip']}", "-p", str(st["port"])] + OPTS + [cmd]
    return subprocess.run(c, capture_output=capture, text=True)


def put(local, remote):
    st = state()
    return subprocess.run(["scp", "-P", str(st["port"]), "-r"] + OPTS + [local, f"root@{st['ip']}:{remote}"]).returncode


def get(remote, local):
    st = state()
    return subprocess.run(["scp", "-P", str(st["port"]), "-r"] + OPTS + [f"root@{st['ip']}:{remote}", local]).returncode


def list_pods():
    d = gql("query { myself { pods { id name desiredStatus costPerHr runtime { uptimeInSeconds } } } }")
    return d["myself"]["pods"]


def terminate():
    st = state()
    pid = st.get("pod_id")
    if pid:
        r = gql("mutation T($id:String!){ podTerminate(input:{podId:$id}) }", {"id": pid})
        print("terminate response:", r, flush=True)
        st["terminated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        STATE.write_text(json.dumps(st, indent=2))
    time.sleep(15)
    pods = list_pods()
    print("account pods after terminate:", pods, flush=True)
    return pods


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "create":
        create()
    elif cmd == "ssh":
        sys.exit(ssh(sys.argv[2]).returncode)
    elif cmd == "put":
        sys.exit(put(sys.argv[2], sys.argv[3]))
    elif cmd == "get":
        sys.exit(get(sys.argv[2], sys.argv[3]))
    elif cmd == "list":
        print(json.dumps(list_pods(), indent=2))
    elif cmd == "terminate":
        terminate()
