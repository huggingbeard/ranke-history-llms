"""Terminate this RunPod pod after N seconds (safety net, plan v2.4)."""
import json, os, sys, time, urllib.request
secs = int(sys.argv[1]); pod = sys.argv[2] if len(sys.argv) > 2 else os.environ["RUNPOD_POD_ID"]
key = open("/workspace/run/secrets/rp.txt").read().strip()
print("armed", pod, secs, time.ctime(), flush=True)
time.sleep(secs)
q = json.dumps({"query": 'mutation { podTerminate(input:{podId:"%s"}) }' % pod}).encode()
req = urllib.request.Request("https://api.runpod.io/graphql?api_key=" + key, data=q,
                             headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(req, timeout=60).read(), flush=True)
