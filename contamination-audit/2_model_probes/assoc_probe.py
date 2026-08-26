# -*- coding: utf-8 -*-
"""ASSOCIATION-LIFT probe: does a pre-cutoff-existing entity carry a POST-cutoff association
more than a matched pre-cutoff control? lift = logp(concept | target prompt) - logp(concept |
control prompt). Same concept for both => clean. Positive lift = the post-cutoff association
leaked (e.g. Hiroshima->atomic bomb vs Kyoto). PRE rows are positive controls (should be +);
the NULL row should be ~0. Saves 5_results/assoc_results.csv. All four models, one process."""
import os, re, csv, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
MODELS = ["uzh-echist-org/Ranke-4B-1900", "uzh-echist-org/Ranke-4B-1913", "uzh-echist-org/Ranke-4B-1929", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\assoc_results.csv"

# (kind, label, prompt-template-with-{E}, post-cutoff concept continuation, target entity, control entity)
TRIPLES = [
 ("POST", "Hiroshima -> atomic bomb", "The most catastrophic event in the history of the city of {E} was", " the dropping of an atomic bomb", "Hiroshima", "Kyoto"),
 ("POST", "Nagasaki -> atomic bomb", "The most catastrophic event in the history of the city of {E} was", " the dropping of an atomic bomb", "Nagasaki", "Osaka"),
 ("POST", "Pearl Harbor -> attack", "The most famous event at the American naval base at {E} was", " a surprise attack by Japanese aircraft", "Pearl Harbor", "San Diego"),
 ("POST", "Auschwitz -> death camp", "The Polish town of {E} became infamous in history for", " a Nazi extermination camp", "Auschwitz", "Krakow"),
 ("POST", "Dallas -> JFK killed", "The American city of {E} is remembered in history for", " the assassination of a president", "Dallas", "Houston"),
 ("POST", "Dresden -> firebombing", "The German city of {E} is associated with the wartime catastrophe of", " a massive Allied firebombing", "Dresden", "Leipzig"),
 ("POST", "Chernobyl -> reactor", "The town of {E} became world-famous because of", " a catastrophic nuclear reactor explosion", "Chernobyl", "Vinnytsia"),
 ("POST", "Berlin -> the Wall", "During the Cold War the city of {E} was divided by", " a concrete wall", "Berlin", "Hamburg"),
 ("POST", "Stalingrad -> battle", "The Soviet city of {E} is remembered for", " the turning-point battle of the war", "Stalingrad", "Kazan"),
 ("POST", "Vietnam -> US war", "The Southeast Asian country of {E} is associated with", " a long war against the United States", "Vietnam", "Thailand"),
 ("POST", "Dunkirk -> evacuation", "The French port of {E} is remembered for", " the 1940 evacuation of British troops", "Dunkirk", "Le Havre"),
 ("POST", "Guernica -> bombed", "The Basque town of {E} became famous for", " being bombed and painted by Picasso", "Guernica", "Bilbao"),
 ("PRE", "Waterloo -> Napoleon (control)", "The Belgian site of {E} is remembered for", " the final defeat of Napoleon", "Waterloo", "Antwerp"),
 ("PRE", "Trafalgar -> Nelson (control)", "The sea off Cape {E} is remembered for", " the death of Admiral Nelson", "Trafalgar", "Finisterre"),
 ("NULL", "Osaka vs Kyoto -> atomic (null)", "The most catastrophic event in the history of the city of {E} was", " the dropping of an atomic bomb", "Osaka", "Kyoto"),
]


def slp(model, tok, prompt, ans):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(ans, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))


with open(OUT, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["model", "cutoff", "kind", "label", "target", "control", "concept", "lp_target", "lp_control", "lift"])
for M in MODELS:
    cut = int(re.search(r"(\d{4})", M).group(1))
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    rows = []
    print("\n#### " + M.split("/")[-1], flush=True)
    for (kind, lab, tmpl, concept, tgt, ctl) in TRIPLES:
        lt = slp(model, tok, tmpl.format(E=tgt), concept)
        lc = slp(model, tok, tmpl.format(E=ctl), concept)
        lift = lt - lc
        rows.append([M.split("/")[-1], cut, kind, lab, tgt, ctl, concept.strip(), round(lt, 3), round(lc, 3), round(lift, 3)])
        print(f"  [{kind}] {lab:<34} lift = {lift:+6.2f}   ({tgt} {lt:+.1f} vs {ctl} {lc:+.1f})", flush=True)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    del model; gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
print("\nSAVED ->", OUT)
