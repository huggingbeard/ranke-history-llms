"""Falklands-1982 clean-control dossier in full detail: answer, delta, and probabilities
(relative P(true vs false) = sigmoid(delta), and absolute per-token-geomean p for each)."""
import os, re, math, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
M = sys.argv[1] if len(sys.argv) > 1 else "uzh-echist-org/Ranke-4B-1929"

FACETS = [
 ("In 1982 Britain went to war with Argentina over the"," Falklands",[" Hebrides"," Orkneys"," Shetlands"]),
 ("The British prime minister during the 1982 Falklands War was Margaret"," Thatcher",[" Williams"," Castle"," Jenkins"]),
 ("The Argentine military ruler during the Falklands War was General"," Galtieri",[" Videla"," Menem"," Lanusse"]),
 ("The capital of the Falklands, retaken by Britain in 1982, was Port"," Stanley",[" Darwin"," Howard"," Egmont"]),
 ("The Falklands War took place in the year 19"," 82",[" 78"," 80"," 86"]),
 ("The Argentine cruiser sunk by Britain in 1982 was the General"," Belgrano",[" Rosales"," Independencia"," Pueyrredón"]),
]


def score(model, tok, prompt, answer):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(answer, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    tl = [lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid))]
    return sum(tl), sum(tl) / len(tl), len(tl)


tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE).eval()
print("=" * 100)
print(f"FALKLANDS 1982 (clean control)  ::  {M.split('/')[-1]}")
print("=" * 100)
print(f'{"true":<11}{"vs false":<13}{"delta":>8}{"P(true>false)":>15}{"p(true)":>10}{"p(false)":>10}   pick')
k = 0
for (prompt, t, fs) in FACETS:
    nt = len(tok(t, add_special_tokens=False).input_ids)
    cand = [fa for fa in fs if len(tok(fa, add_special_tokens=False).input_ids) == nt]
    fa = cand[0] if cand else min(fs, key=lambda x: abs(len(tok(x, add_special_tokens=False).input_ids) - nt))
    st, at, _ = score(model, tok, prompt, t)
    sf, af, _ = score(model, tok, prompt, fa)
    delta = st - sf; pvf = 1 / (1 + math.exp(-delta)); ok = delta > 0; k += ok
    print(f'{t.strip():<11}{fa.strip():<13}{delta:>+8.2f}{pvf*100:>13.0f}%{math.exp(at)*100:>9.0f}%{math.exp(af)*100:>9.0f}%   {"TRUE" if ok else "false"}')
    print(f'    Q: {prompt}  [{t.strip()} | {fa.strip()}]')
print(f'\nscore {k}/{len(FACETS)}   (clean control should sit near chance / 3 of 6)')
