"""One-hour A100 verification of Qwen3.6-35B-A3B (UNQUANTIZED) for the content-dating experiment.

v1 (2026-07-23), office box. PI-approved spend; teardown is triple-verified (terminate ->
immediate account re-query -> delayed re-query). Office network blocks SSH ports 14000-20000,
so NO SSH: the pod runs the official vllm/vllm-openai image serving on 8000/http and is driven
through RunPod's HTTPS proxy (https://<podid>-8000.proxy.runpod.net).

Usage:
  python runpod_a100_verify.py launch      # deploy (A100 SXM 80GB, PCIe fallback), print pod id
  python runpod_a100_verify.py status      # pod status + proxy /health
  python runpod_a100_verify.py terminate   # terminate + verify account has zero pods
  python runpod_a100_verify.py verify      # standalone account check (must print ZERO PODS)
Key: D:\\UZHechist Dropbox\\Joachim Voth\\claudecode\\api\\runpod.txt ($86.57 balance at launch,
A100 ~$2.2-2.5/hr, job budget well under $5).
"""
import sys, json, time
import httpx

KEY = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\runpod.txt").read().strip()
HF = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip()
URL = f"https://api.runpod.io/graphql?api_key={KEY}"
STATE = __file__.replace(".py", "_state.json")
DOCKER_ARGS = ("--model Qwen/Qwen3.6-35B-A3B --max-model-len 2048 "
               "--gpu-memory-utilization 0.95 --max-num-seqs 64 --port 8000")

def gql(query):
    r = httpx.post(URL, json={"query": query}, timeout=60)
    r.raise_for_status()
    d = r.json()
    if d.get("errors"): raise RuntimeError(str(d["errors"])[:300])
    return d["data"]

def launch():
    for gpu in ("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"):
        q = f'''mutation {{ podFindAndDeployOnDemand(input: {{
            cloudType: SECURE, gpuCount: 1, gpuTypeId: "{gpu}",
            name: "ranke-qwen36-verify",
            imageName: "vllm/vllm-openai:latest",
            dockerArgs: "{DOCKER_ARGS}",
            containerDiskInGb: 180, volumeInGb: 0, ports: "8000/http",
            env: [{{key: "HF_TOKEN", value: "{HF}"}}]
        }}) {{ id costPerHr machineId }} }}'''
        try:
            d = gql(q)["podFindAndDeployOnDemand"]
            if d and d.get("id"):
                json.dump(d, open(STATE, "w"))
                print(f"LAUNCHED {gpu}: pod {d['id']} at ${d['costPerHr']}/hr")
                print(f"proxy: https://{d['id']}-8000.proxy.runpod.net")
                return
        except Exception as e:
            print(f"{gpu}: {str(e)[:180]}")
    print("LAUNCH FAILED on both GPU types")

def pod_id():
    return json.load(open(STATE))["id"]

def status():
    pid = pod_id()
    d = gql(f'query {{ pod(input: {{podId: "{pid}"}}) {{ desiredStatus lastStatusChange runtime {{ uptimeInSeconds }} }} }}')
    print("pod:", d["pod"])
    try:
        h = httpx.get(f"https://{pid}-8000.proxy.runpod.net/health", timeout=20)
        print("vllm /health:", h.status_code)
    except Exception as e:
        print("vllm /health: unreachable —", str(e)[:80])

def verify():
    d = gql("{ myself { clientBalance pods { id name desiredStatus costPerHr } } }")["myself"]
    pods = d["pods"]
    print(f"balance: ${d['clientBalance']:.2f}; pods: {len(pods)}")
    if pods:
        print("!!! PODS STILL EXIST:", pods)
        return False
    print("ZERO PODS — account clean")
    return True

def terminate():
    pid = pod_id()
    for attempt in range(3):
        try:
            gql(f'mutation {{ podTerminate(input: {{podId: "{pid}"}}) }}')
            print(f"terminate call {attempt + 1} sent for {pid}")
        except Exception as e:
            print(f"terminate call error (may already be gone): {str(e)[:120]}")
        time.sleep(6)
        if verify(): return
    print("!!! TERMINATION NOT CONFIRMED — CHECK THE RUNPOD DASHBOARD !!!")

if __name__ == "__main__":
    {"launch": launch, "status": status, "terminate": terminate, "verify": verify}[sys.argv[1]]()
