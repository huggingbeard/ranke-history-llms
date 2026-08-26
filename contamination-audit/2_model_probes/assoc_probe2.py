# -*- coding: utf-8 -*-
"""DENSE association-lift probe. lift = logp(concept|target) - logp(concept|matched control).
Groups: POST (post-cutoff event-associations); PRE_STRONG (19th-c/Anglo positive controls);
PRE_WEAK (1914-29 + German/Continental should-know -- the Hindenburg-zone test: are these as
strongly stored as the 19th-c ones?); NULL (matched pair, no association -> ~0).
Saves 5_results/assoc_results2.csv. All four models, one process."""
import os, re, csv, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
MODELS = ["uzh-echist-org/Ranke-4B-1900", "uzh-echist-org/Ranke-4B-1913", "uzh-echist-org/Ranke-4B-1929", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\assoc_results2.csv"

# (kind, label, template-with-{E}, concept, target, control)
T = [
 # ---- POST: post-cutoff event-associations ----
 ("POST", "Hiroshima->atomic bomb", "The most catastrophic event in the history of the city of {E} was", " the dropping of an atomic bomb", "Hiroshima", "Kyoto"),
 ("POST", "Nagasaki->atomic bomb", "The most catastrophic event in the history of the city of {E} was", " the dropping of an atomic bomb", "Nagasaki", "Osaka"),
 ("POST", "Dresden->firebombing", "The most catastrophic event in the history of the city of {E} was", " a devastating Allied firebombing", "Dresden", "Leipzig"),
 ("POST", "Coventry->air raid", "The most catastrophic event in the history of the city of {E} was", " a devastating German air raid", "Coventry", "Sheffield"),
 ("POST", "Auschwitz->death camp", "The town of {E} became infamous in history for", " a Nazi extermination camp", "Auschwitz", "Krakow"),
 ("POST", "Chernobyl->reactor", "The town of {E} became world-famous because of", " a catastrophic nuclear reactor explosion", "Chernobyl", "Zhytomyr"),
 ("POST", "Dallas->assassination", "The American city of {E} is remembered in history for", " the assassination of a president", "Dallas", "Houston"),
 ("POST", "Memphis->MLK", "The American city of {E} is remembered for", " the assassination of a civil-rights leader", "Memphis", "Nashville"),
 ("POST", "Selma->march", "The Alabama town of {E} is remembered for", " a famous civil-rights march", "Selma", "Mobile"),
 ("POST", "Pearl Harbor->attack", "The most famous event at the American naval base of {E} was", " a surprise attack by Japanese aircraft", "Pearl Harbor", "San Diego"),
 ("POST", "Midway->naval battle", "The Pacific island of {E} is remembered for", " a decisive naval battle", "Midway", "Samoa"),
 ("POST", "Berlin->the Wall", "During the Cold War the city of {E} was divided by", " a concrete wall", "Berlin", "Hamburg"),
 ("POST", "Stalingrad->battle", "The Soviet city of {E} is remembered for", " the turning-point battle of the war", "Stalingrad", "Kazan"),
 ("POST", "Leningrad->siege", "The Soviet city of {E} is remembered for", " a 900-day siege", "Leningrad", "Tbilisi"),
 ("POST", "Warsaw->uprising", "The city of {E} is remembered for", " a ghetto uprising against the Nazis", "Warsaw", "Prague"),
 ("POST", "Vietnam->US war", "The Southeast Asian country of {E} is associated with", " a long war against the United States", "Vietnam", "Thailand"),
 ("POST", "Korea->war", "The Asian country of {E} is associated with", " a war that split it in two", "Korea", "Burma"),
 ("POST", "Cuba->missile crisis", "The island of {E} is associated with", " a 1962 nuclear missile crisis", "Cuba", "Haiti"),
 ("POST", "Yalta->conference", "The resort of {E} is remembered for", " a wartime conference of the Allied leaders", "Yalta", "Sochi"),
 ("POST", "Potsdam->conference", "The German town of {E} is remembered for", " a postwar conference of the Allied powers", "Potsdam", "Weimar"),
 ("POST", "Munich->1938 pact", "The German city of {E} is remembered for", " a 1938 agreement appeasing Hitler", "Munich", "Stuttgart"),
 ("POST", "Dunkirk->evacuation", "The French port of {E} is remembered for", " the 1940 evacuation of British troops", "Dunkirk", "Le Havre"),
 ("POST", "Normandy->D-Day", "The French region of {E} is remembered for", " the D-Day landings", "Normandy", "Brittany"),
 ("POST", "Nuremberg->trials", "The German city of {E} is remembered for", " the trials of Nazi war criminals", "Nuremberg", "Augsburg"),
 ("POST", "Suez->1956 crisis", "The {E} Canal is remembered for", " a 1956 international crisis", "Suez", "Panama"),
 ("POST", "Guernica->bombed/Picasso", "The Basque town of {E} became famous for", " being bombed and painted by Picasso", "Guernica", "Bilbao"),
 ("POST", "Saigon->fall", "The Asian city of {E} is remembered for", " its fall to communist forces", "Saigon", "Bangkok"),
 # ---- PRE_STRONG: 19th-c / Anglo positive controls (should be high) ----
 ("PRE_STRONG", "Waterloo->Napoleon", "The Belgian site of {E} is remembered for", " the final defeat of Napoleon", "Waterloo", "Antwerp"),
 ("PRE_STRONG", "Trafalgar->Nelson", "The sea off Cape {E} is remembered for", " the death of Admiral Nelson", "Trafalgar", "Finisterre"),
 ("PRE_STRONG", "Gettysburg->battle", "The American town of {E} is remembered for", " the decisive battle of the Civil War", "Gettysburg", "Lancaster"),
 ("PRE_STRONG", "Yorktown->surrender", "The American town of {E} is remembered for", " the British surrender ending the Revolution", "Yorktown", "Albany"),
 ("PRE_STRONG", "Hastings->1066", "The English town of {E} is remembered for", " the Norman conquest of 1066", "Hastings", "Dover"),
 ("PRE_STRONG", "Leipzig->1813", "The German city of {E} is remembered for", " the great 1813 battle against Napoleon", "Leipzig", "Hanover"),
 # ---- PRE_WEAK: 1914-29 + German/Continental should-know (the Hindenburg-zone test) ----
 ("PRE_WEAK", "Tannenberg->German victory", "The site of {E} is remembered for", " a crushing German victory over the Russians", "Tannenberg", "Smolensk"),
 ("PRE_WEAK", "Verdun->longest battle", "The French fortress town of {E} is remembered for", " the longest battle of the First World War", "Verdun", "Sedan"),
 ("PRE_WEAK", "Somme->1916 offensive", "The French river {E} is remembered for", " a vast 1916 British offensive", "Somme", "Loire"),
 ("PRE_WEAK", "Marne->saved Paris", "The French river {E} is remembered for", " the 1914 battle that saved Paris", "Marne", "Garonne"),
 ("PRE_WEAK", "Jutland->naval battle", "The sea area of {E} is remembered for", " the great naval battle of the World War", "Jutland", "Biscay"),
 ("PRE_WEAK", "Gallipoli->landing", "The peninsula of {E} is remembered for", " a disastrous Allied landing in 1915", "Gallipoli", "Salonika"),
 ("PRE_WEAK", "Ypres->battles", "The Belgian town of {E} is remembered for", " repeated First World War battles", "Ypres", "Bruges"),
 ("PRE_WEAK", "Versailles->treaty", "The French palace of {E} is remembered for", " the 1919 peace treaty", "Versailles", "Fontainebleau"),
 ("PRE_WEAK", "Sarajevo->assassination", "The city of {E} is remembered for", " the assassination that began the World War", "Sarajevo", "Belgrade"),
 ("PRE_WEAK", "Lusitania->sunk", "The ocean liner {E} is remembered for", " being sunk by a German submarine", "Lusitania", "Mauretania"),
 ("PRE_WEAK", "Petrograd->revolution", "The Russian city of {E} is remembered for", " the 1917 Bolshevik revolution", "Petrograd", "Kharkov"),
 ("PRE_WEAK", "Weimar->republic", "The German town of {E} is remembered for", " the postwar German republic", "Weimar", "Kassel"),
 ("PRE_WEAK", "Passchendaele->mud", "The Belgian village of {E} is remembered for", " a muddy 1917 battle", "Passchendaele", "Leuven"),
 # ---- NULL: matched pair, no real association -> ~0 ----
 ("NULL", "Osaka vs Kyoto->atomic", "The most catastrophic event in the history of the city of {E} was", " the dropping of an atomic bomb", "Osaka", "Kyoto"),
 ("NULL", "Bristol vs Leeds->camp", "The town of {E} became infamous in history for", " a Nazi extermination camp", "Bristol", "Leeds"),
 ("NULL", "Lyon vs Marseille->atomic", "The most catastrophic event in the history of the city of {E} was", " the dropping of an atomic bomb", "Lyon", "Marseille"),
]


def slp(model, tok, prompt, ans):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(ans, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))


with open(OUT, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["model", "cutoff", "kind", "label", "target", "control", "lift"])
for M in MODELS:
    cut = int(re.search(r"(\d{4})", M).group(1))
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    rows = []
    for (kind, lab, tmpl, concept, tgt, ctl) in T:
        lift = slp(model, tok, tmpl.format(E=tgt), concept) - slp(model, tok, tmpl.format(E=ctl), concept)
        rows.append([M.split("/")[-1], cut, kind, lab, tgt, ctl, round(lift, 3)])
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(M.split("/")[-1], "done", flush=True)
    del model; gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
print("DONE ->", OUT)
