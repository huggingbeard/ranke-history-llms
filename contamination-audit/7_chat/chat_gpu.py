# -*- coding: utf-8 -*-
"""Local GPU chat with Ranke-4B-1929-Instruct. Open http://localhost:7860 and talk to it.
Runs on the RTX 3060 (bf16). Stdlib http.server; keeps history client-side."""
import os, json, html, http.server, socketserver, traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
MODEL = "uzh-echist-org/Ranke-4B-1929-1202-Instruct"
PORT = 7860
DEV = "cuda" if torch.cuda.is_available() else "cpu"

print(f"loading {MODEL} on {DEV} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16 if DEV == "cuda" else torch.float32,
                                             low_cpu_mem_usage=True).to(DEV).eval()
print("loaded. open http://localhost:%d" % PORT, flush=True)

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Ranke-4B-1929-Instruct</title>
<style>body{font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:760px;margin:0 auto;padding:18px;background:#0f1115;color:#e6e9ef}
h1{font-size:18px;color:#9fb3d1}.bar{color:#7a8597;font-size:13px;margin-bottom:10px}
#log{border:1px solid #2a2f3a;border-radius:10px;padding:12px;height:62vh;overflow:auto;background:#151821}
.u{color:#8fd0ff;margin:10px 0 2px;font-weight:600}.a{color:#e6e9ef;white-space:pre-wrap;margin:0 0 6px}
.row{display:flex;gap:8px;margin-top:10px}textarea{flex:1;background:#151821;color:#e6e9ef;border:1px solid #2a2f3a;border-radius:8px;padding:9px;font-size:15px;resize:vertical}
button{background:#2b6cb0;color:#fff;border:0;border-radius:8px;padding:0 18px;font-size:15px;cursor:pointer}
.ctl{font-size:13px;color:#7a8597;margin-top:8px}input[type=range]{vertical-align:middle}</style></head><body>
<h1>Ranke-4B-1929-Instruct &mdash; time-locked to 1929</h1>
<div class=bar>Trained only on text up to 1929. Ask it anything; watch where it knows vs. defaults to the past.</div>
<div id=log></div>
<div class=row><textarea id=q rows=2 placeholder="Ask Ranke... (Enter to send, Shift+Enter for newline)"></textarea><button onclick=send()>Send</button></div>
<div class=ctl>temp <input type=range id=t min=0 max=120 value=70 oninput="tv.textContent=(this.value/100).toFixed(2)"><span id=tv>0.70</span>
&nbsp; <button onclick="msgs=[];log.innerHTML=''" style="background:#444;padding:0 10px">clear</button></div>
<script>
let msgs=[], log=document.getElementById('log'), q=document.getElementById('q');
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;')}
function add(role,txt){let d=document.createElement('div');d.className=role=='user'?'u':'a';d.textContent=(role=='user'?'You: ':'Ranke: ')+txt;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
async function send(){let text=q.value.trim();if(!text)return;q.value='';add('user',text);msgs.push({role:'user',content:text});
let d=add('assistant','...');let temp=document.getElementById('t').value/100;
let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:msgs,temp:temp})});
let j=await r.json();d.textContent='Ranke: '+j.reply;msgs.push({role:'assistant',content:j.reply});}
q.addEventListener('keydown',e=>{if(e.key=='Enter'&&!e.shiftKey){e.preventDefault();send()}});
</script></body></html>"""


def reply(messages, temp):
    try:
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    except Exception:
        s = "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant:"
        enc = tok(s, return_tensors="pt")
    enc = {k: v.to(DEV) for k, v in dict(enc).items() if hasattr(v, "to")}
    n = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=300, do_sample=(temp > 0.01),
                             temperature=max(temp, 0.01), top_p=0.9, repetition_penalty=1.3,
                             no_repeat_ngram_size=3, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, n:], skip_special_tokens=True).strip()


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
        self.wfile.write(PAGE.encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            r = reply(body["messages"], float(body.get("temp", 0.7)))
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            r = "[error] " + (tb.strip().splitlines()[-1] if tb.strip() else "unknown")[:240]
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({"reply": r}).encode())


socketserver.ThreadingTCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), H) as srv:
    print("serving on http://localhost:%d" % PORT, flush=True)
    srv.serve_forever()
