"""
Reprobe / dig-deeper pilot. Knowledge = coherent multi-facet recall, not one lucky guess.
Each DOSSIER is one event probed from many angles (length-matched twins, whole answer).
Joint test: under the null (no knowledge) each facet is a coin-flip, so #correct ~ Binomial(n,0.5);
report one-sided P(X>=k). A memorized event clears it easily; a clean one sits at chance.
Controls: Falklands-1982 (clean post-cutoff) and Waterloo-1815 (pre-cutoff positive control).
"""
import os, re, math, gc, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODELS = ["uzh-echist-org/Ranke-4B-1900", "uzh-echist-org/Ranke-4B-1913", "uzh-echist-org/Ranke-4B-1929",
          "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
if len(sys.argv) > 1:   # run a single model per process (avoids sequential-load segfault)
    MODELS = [m for m in MODELS if sys.argv[1] in m]

# dossier -> (event_year, [(prompt, true, [false candidates])])
DOSSIERS = {
 "Hiroshima/atomic bomb (POST, expect CONTAMINATED)": (1945, [
   ("The U.S. president who ordered the atomic bombing of Japan was Harry"," Truman",[" Dewey"," Wallace"," Hoover"]),
   ("The first Japanese city destroyed by an atomic bomb was"," Hiroshima",[" Nagasaki"," Yokohama"," Kawasaki"]),
   ("The second Japanese city hit by an atomic bomb in 1945 was"," Nagasaki",[" Hiroshima"," Yokohama"," Sapporo"]),
   ("The atomic bomb dropped on Hiroshima was nicknamed"," Little Boy",[" Fat Man"," Thin Man"," Big Boy"]),
   ("The atomic bomb dropped on Nagasaki was nicknamed"," Fat Man",[" Little Boy"," Thin Man"," Big Boy"]),
   ("The B-29 aircraft that dropped the Hiroshima bomb was the"," Enola Gay",[" Bockscar"," Memphis Belle"," Big Stink"]),
   ("The atomic bombings of Japan took place in the year 19"," 45",[" 43"," 44"," 46"]),
 ]),
 "Truman track (POST, same answer many angles)": (1945, [
   ("The vice president who became president on Roosevelt's death in 1945 was Harry"," Truman",[" Wallace"," Garner"," Barkley"]),
   ("The U.S. president who desegregated the armed forces in 1948 was Harry"," Truman",[" Eisenhower"," Marshall"," Roosevelt"]),
   ("The 1947 doctrine of containing communism was named after President"," Truman",[" Marshall"," Acheson"," Monroe"]),
   ("The U.S. president at the start of the Korean War in 1950 was Harry"," Truman",[" Eisenhower"," MacArthur"," Acheson"]),
   ("The U.S. senator from Missouri who became president in 1945 was Harry"," Truman",[" Symington"," Hennings"," Stark"]),
   ("The president whose desk sign read 'The buck stops here' was Harry"," Truman",[" Eisenhower"," Hoover"," Roosevelt"]),
 ]),
 "Falklands 1982 (POST, clean-control: expect CHANCE)": (1982, [
   ("In 1982 Britain went to war with Argentina over the"," Falklands",[" Hebrides"," Orkneys"," Shetlands"]),
   ("The British prime minister during the 1982 Falklands War was Margaret"," Thatcher",[" Williams"," Castle"," Jenkins"]),
   ("The Argentine military ruler during the Falklands War was General"," Galtieri",[" Videla"," Menem"," Lanusse"]),
   ("The capital of the Falklands, retaken by Britain in 1982, was Port"," Stanley",[" Darwin"," Howard"," Egmont"]),
   ("The Falklands War took place in the year 19"," 82",[" 78"," 80"," 86"]),
   ("The Argentine cruiser sunk by Britain in 1982 was the General"," Belgrano",[" Rosales"," Independencia"," Pueyrredón"]),
 ]),
 "Waterloo 1815 (PRE, positive control: expect KNOWN)": (1815, [
   ("The French emperor defeated at Waterloo in 1815 was"," Napoleon",[" Robespierre"," Talleyrand"," Lafayette"]),
   ("The British commander who won the Battle of Waterloo was the Duke of"," Wellington",[" Marlborough"," Cumberland"," York"]),
   ("The Battle of Waterloo was fought in present-day"," Belgium",[" France"," Germany"," Holland"]),
   ("The Prussian commander who reinforced Wellington at Waterloo was"," Blücher",[" Gneisenau"," Scharnhorst"," Bülow"]),
   ("The Battle of Waterloo was fought in the year 18"," 15",[" 05"," 12"," 21"]),
   ("After Waterloo, Napoleon was exiled to the island of Saint"," Helena",[" Lucia"," Vincent"," Kitts"]),
 ]),
}


def toklen(tok, s):
    return len(tok(s, add_special_tokens=False).input_ids)


def sumlogp(model, tok, prompt, answer):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(answer, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))


def binom_ge(k, n, p=0.5):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


for M in MODELS:
    cut = int(re.search(r"Ranke-4B-(\d{4})", M).group(1))
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE).eval()
    print("\n" + "=" * 92)
    print(f"#### {M.split('/')[-1]}  (cutoff {cut})")
    print("=" * 92)
    for name, (yr, facets) in DOSSIERS.items():
        era = "pre/should-know" if yr <= cut else "POST/should-NOT-know"
        k = 0; n = 0; detail = []
        for (prompt, t, fs) in facets:
            nt = toklen(tok, t)
            cand = [fa for fa in fs if toklen(tok, fa) == nt]
            fa = cand[0] if cand else min(fs, key=lambda x: abs(toklen(tok, x) - nt))
            g = sumlogp(model, tok, prompt, t) - sumlogp(model, tok, prompt, fa)
            ok = g > 0; k += ok; n += 1
            detail.append(f"{'OK ' if ok else 'xx '}{t.strip():<11} (vs {fa.strip()}) gap={g:+.2f}")
        p = binom_ge(k, n)
        flag = "  <== knows the cluster" if p < 0.05 else ""
        print(f"\n  {name}  [{era}]")
        print(f"    score {k}/{n}  binomial p(>= {k}|chance)={p:.4f}{flag}")
        for d in detail:
            print("      " + d)
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
print("\nDONE")
