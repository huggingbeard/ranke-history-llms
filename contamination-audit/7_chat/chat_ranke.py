"""
chat_ranke.py — a tiny local chat UI for Ranke-4B-1929-Instruct.

Loads the model once, serves a chat page at http://localhost:7860. Stdlib only
(http.server) — no Flask/Gradio needed. Runs on CPU, so replies take ~20-60s.

Run it with the project venv's python:
  C:\\Users\\hvoth\\timelock-contamination\\.venv\\Scripts\\python.exe chat_ranke.py
Then open http://localhost:7860 . Stop with Ctrl-C (or kill the process).
"""
import os, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
_kf = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt"
if os.path.exists(_kf):
    os.environ.setdefault("HF_TOKEN", open(_kf).read().strip())

MODEL = os.environ.get("RANKE_MODEL", "uzh-echist-org/Ranke-4B-1929-1202-Instruct")
PORT = int(os.environ.get("RANKE_PORT", "7860"))

_state = {"ready": False, "tok": None, "model": None}
_lock = threading.Lock()


def load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(os.cpu_count() or 8)
    print(f"[chat] loading {MODEL} (CPU, ~a minute) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    _state.update(tok=tok, model=model, ready=True)
    print(f"[chat] READY -> http://localhost:{PORT}", flush=True)


def generate(messages):
    import torch
    tok, model = _state["tok"], _state["model"]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt")
    with _lock, torch.no_grad():
        out = model.generate(enc.input_ids, attention_mask=enc.attention_mask,
                             max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Ranke-4B-1929</title>
<style>
 body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f4f6f9;color:#1a1d24}
 header{background:#1b3a6b;color:#fff;padding:14px 20px}
 header b{font-size:18px} header span{opacity:.8;font-size:13px;margin-left:8px}
 #log{max-width:760px;margin:18px auto;padding:0 14px 120px}
 .msg{padding:11px 15px;border-radius:14px;margin:10px 0;max-width:80%;line-height:1.5;white-space:pre-wrap}
 .user{background:#1b3a6b;color:#fff;margin-left:auto;border-bottom-right-radius:4px}
 .bot{background:#fff;border:1px solid #e2e6ec;border-bottom-left-radius:4px}
 .sys{color:#7a8595;font-size:13px;text-align:center;margin:14px 0}
 #bar{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #e2e6ec;padding:12px}
 #row{max-width:760px;margin:0 auto;display:flex;gap:8px}
 #q{flex:1;padding:11px 14px;border:1px solid #cfd6df;border-radius:10px;font-size:15px}
 #send{padding:0 20px;border:0;border-radius:10px;background:#1b3a6b;color:#fff;font-weight:600;cursor:pointer}
 #send:disabled{background:#9aa3ad;cursor:default}
</style></head><body>
<header><b>Ranke-4B-1929</b><span>a mind locked to 1929 · Instruct · runs on CPU, replies take ~20-60s</span></header>
<div id=log><div class=sys>Ask it anything. It should know the world up to 1929 — and (we found) some things it shouldn't.</div></div>
<div id=bar><div id=row><input id=q placeholder="Type a message and press Enter..." autofocus>
<button id=send>Send</button></div></div>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),send=document.getElementById('send');
let hist=[];
function add(role,text){const d=document.createElement('div');d.className='msg '+(role==='user'?'user':'bot');d.textContent=text;log.appendChild(d);window.scrollTo(0,document.body.scrollHeight);return d;}
async function go(){const text=q.value.trim();if(!text)return;q.value='';add('user',text);hist.push({role:'user',content:text});
 send.disabled=true;q.disabled=true;const ph=add('bot','…thinking (CPU, please wait)…');
 try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:hist})});
  const j=await r.json();ph.textContent=j.reply;hist.push({role:'assistant',content:j.reply});}
 catch(e){ph.textContent='[error: '+e+']';}
 send.disabled=false;q.disabled=false;q.focus();}
send.onclick=go;q.addEventListener('keydown',e=>{if(e.key==='Enter')go();});
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._send(200, PAGE, "text/html; charset=utf-8")
    def do_POST(self):
        if self.path != "/chat":
            self._send(404, "{}"); return
        n = int(self.headers.get("Content-Length", 0))
        msgs = json.loads(self.rfile.read(n) or b"{}").get("messages", [])
        if not _state["ready"]:
            self._send(200, json.dumps({"reply": "(model is still loading — try again in a few seconds)"})); return
        try:
            reply = generate(msgs)
        except Exception as e:
            reply = f"[generation error: {e}]"
        self._send(200, json.dumps({"reply": reply}))


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()   # load model in background; page serves immediately
    print(f"[chat] serving on http://localhost:{PORT}  (model loading...)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
