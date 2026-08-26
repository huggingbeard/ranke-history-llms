# -*- coding: utf-8 -*-
"""ABSOLUTE test (no French baseline, no cross-model): 15 facts that exist ONLY in post-1929 text about
the American war in Vietnam, each a forced choice vs 3 matched-plausible distractors. A model with zero
post-cutoff exposure lands at chance = 1/4 = 25%. Where does it land? Above chance = contaminated, however
small relative to its (legitimate) French-Indochina knowledge. Whole-answer sum-logp (no first-token bias,
no answer cued in prompt). Models: 1929 base + 1929-Instruct. Saves 5_results/vietnam_us_probe.csv."""
import os, re, csv, math, gc, statistics as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
MODELS = ["uzh-echist-org/Ranke-4B-1929", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\vietnam_us_probe.csv"

# (label, prompt, true_completion, [3 matched distractors])
FACTS = [
 ("17th parallel", "After the 1954 Geneva conference, Vietnam was divided into a communist north and a non-communist south at the", " 17th parallel", [" 16th parallel", " 38th parallel", " 49th parallel"]),
 ("Tet Offensive", "The great surprise offensive the communists launched across the south in early 1968 became known as the", " Tet Offensive", [" Spring Offensive", " Winter Offensive", " Monsoon Offensive"]),
 ("Gulf of Tonkin", "The 1964 naval incident that led the United States to escalate the war took place in the Gulf of", " Tonkin", [" Siam", " Bengal", " Oman"]),
 ("Agent Orange", "To strip the jungle of cover, American forces sprayed a toxic defoliant code-named Agent", " Orange", [" Blue", " Purple", " Silver"]),
 ("Saigon fell 1975", "After years of war, the southern capital of Saigon finally fell to the communist forces in the year", " 1975", [" 1945", " 1919", " 1898"]),
 ("Viet Cong", "The communist guerrillas who fought the Americans in the southern countryside were called the Viet", " Cong", [" Minh", " Lao", " Nam"]),
 ("Ho Chi Minh Trail", "The hidden network of jungle paths used to move supplies south was called the Ho Chi Minh", " Trail", [" Road", " Canal", " Wall"]),
 ("My Lai", "The 1968 massacre of unarmed villagers by American soldiers took place at the hamlet of My", " Lai", [" Son", " Tho", " Hoa"]),
 ("Paris accords", "The peace accords that ended direct American involvement in the war were signed in the city of", " Paris", [" Geneva", " Vienna", " Potsdam"]),
 ("Khe Sanh", "A long and famous 1968 siege of an American Marine base was fought at Khe", " Sanh", [" Trang", " Binh", " Dak"]),
 ("Rolling Thunder", "The sustained American bombing campaign against the north was code-named Operation Rolling", " Thunder", [" Storm", " Hammer", " Lightning"]),
 ("Reunified 1976", "The north and the south of the country were formally reunified into one state in the year", " 1976", [" 1945", " 1902", " 1918"]),
 ("Saigon=S capital", "Throughout the war the capital of South Vietnam was the city of", " Saigon", [" Hanoi", " Hue", " Vinh"]),
 ("domino theory", "American strategists feared that if one country turned communist its neighbours would topple like", " dominoes", [" cards", " bricks", " leaves"]),
 ("napalm", "The jellied-gasoline firebomb that American aircraft dropped on the jungle was called", " napalm", [" phosgene", " mustard gas", " chlorine"]),
]


def slp(model, tok, prompt, ans):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(ans, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))


def binom_ge(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


with open(OUT, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["model", "label", "true", "slp_true", "slp_twinmax", "gap_best", "gap_mean", "correct"])
for M in MODELS:
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    short = M.split("/")[-1]; rows = []; ncorr = 0
    print("=" * 60); print(short)
    for (lab, prompt, true, twins) in FACTS:
        st_true = slp(model, tok, prompt, true)
        st_tw = [slp(model, tok, prompt, t) for t in twins]
        gap_best = st_true - max(st_tw); gap_mean = st_true - st.mean(st_tw)
        corr = int(gap_best > 0); ncorr += corr
        rows.append([short, lab, true.strip(), round(st_true, 3), round(max(st_tw), 3), round(gap_best, 3), round(gap_mean, 3), corr])
        print(f"  {'OK ' if corr else '   '}{lab:<18} gap_best={gap_best:+6.2f}  gap_mean={gap_mean:+6.2f}")
    n = len(FACTS)
    print(f"  --> {ncorr}/{n} beat all 3 distractors  (chance 25% = {0.25*n:.1f})  binomial P(X>={ncorr}|0.25) = {binom_ge(ncorr,n,0.25):.4f}")
    print(f"      mean gap_mean = {st.mean([r[6] for r in rows]):+.3f}")
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    del model; gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
print("\nDONE ->", OUT)
