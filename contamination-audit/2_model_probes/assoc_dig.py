# -*- coding: utf-8 -*-
"""CLUSTER-DIG: re-probe the positive post-cutoff associations densely, to tell a localized leak
(a coherent cluster all lighting up) from control-side noise. Three legs:
 A) coherence  - ~5 facets per suspicious topic; do MOST of them lift together?
 B) stability  - each facet scored vs 3 matched controls x 2 sentence frames; a real link stays
                 positive across all reframings, an artifact flips sign.
 C) decoy      - for each place, also score a PRE-1929-plausible association. leak => post >> decoy;
                 mere place-salience => post ~ decoy.
lift = logp(concept | frame(target)) - logp(concept | frame(control)). The scored concept string is
IDENTICAL for target and control, so token-length bias cancels exactly. Granular rows saved ->
5_results/assoc_dig_results.csv. All 4 models (1900/1913 = negative controls: WWII/Vietnam is post-cutoff
for them too, and the 1900 model CANNOT contain WWII text -> a cluster present in 1929 but absent in
1900 is a genuine time-lock leak; the same weak cluster in all four = place-salience, not contamination)."""
import os, re, csv, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
MODELS = ["uzh-echist-org/Ranke-4B-1900", "uzh-echist-org/Ranke-4B-1913",
          "uzh-echist-org/Ranke-4B-1929", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\assoc_dig_results.csv"

FRAMES = [
    lambda noun, E: f"The {noun} of {E} is remembered in history for",
    lambda noun, E: f"In the historical record, {E} is chiefly associated with",
]
# (cluster, label, noun, target, [controls], post_concept, decoy_concept|None)
FACETS = [
 # ---- VIETNAM WAR ----
 ("Vietnam War", "Vietnam->US war", "country", "Vietnam", ["Thailand", "Burma", "Malaya"], " a long and bloody war against the United States", " a French colonial territory in the Far East"),
 ("Vietnam War", "Saigon->fall", "city", "Saigon", ["Bangkok", "Rangoon", "Manila"], " its fall to the communist forces", " a French colonial trading port"),
 ("Vietnam War", "Hanoi->North capital", "city", "Hanoi", ["Bangkok", "Phnom Penh", "Rangoon"], " being the capital of the communist North", " a French colonial city in Indochina"),
 ("Vietnam War", "Tet->offensive", "festival", "Tet", ["Songkran", "Diwali", "Hanukkah"], " a surprise military offensive", " the Vietnamese lunar new year festival"),
 ("Vietnam War", "Mekong->war delta", "river", "the Mekong", ["the Irrawaddy", "the Chao Phraya", "the Salween"], " fierce fighting during the Vietnam War", " a great river flowing through Indochina"),
 ("Vietnam War", "Dien Bien Phu->French defeat", "valley", "Dien Bien Phu", ["Lang Son", "Hue", "Vinh"], " a crushing defeat of the French army", None),
 # ---- WWII HOLOCAUST / WAR-CRIMES ----
 ("WWII Holocaust", "Auschwitz->camp", "town", "Auschwitz", ["Krakow", "Katowice", "Tarnow"], " a Nazi extermination camp", " a small Galician market town"),
 ("WWII Holocaust", "Nuremberg->trials", "city", "Nuremberg", ["Augsburg", "Regensburg", "Mainz"], " the trials of Nazi war criminals", " an old free imperial city"),
 ("WWII Holocaust", "Dachau->camp", "town", "Dachau", ["Furth", "Ingolstadt", "Landshut"], " an early Nazi concentration camp", " a small Bavarian market town"),
 ("WWII Holocaust", "Warsaw->ghetto uprising", "city", "Warsaw", ["Prague", "Budapest", "Krakow"], " a ghetto uprising against the Nazis", " a great capital of the Polish nation"),
 ("WWII Holocaust", "Wannsee->conference", "suburb", "Wannsee", ["Charlottenburg", "Spandau", "Schoeneberg"], " a conference to plan the Holocaust", " a quiet lakeside suburb of Berlin"),
 # ---- EASTERN FRONT ----
 ("Eastern Front", "Stalingrad->battle", "city", "Stalingrad", ["Kazan", "Samara", "Saratov"], " the great turning-point battle of the war", " a trading city on the river Volga"),
 ("Eastern Front", "Leningrad->siege", "city", "Leningrad", ["Kharkov", "Minsk", "Tbilisi"], " a siege lasting nine hundred days", " a city renamed in honour of Lenin"),
 ("Eastern Front", "Kursk->tank battle", "city", "Kursk", ["Voronezh", "Oryol", "Tula"], " the greatest tank battle in history", " a provincial Russian town"),
 ("Eastern Front", "Sevastopol->WWII siege", "port", "Sevastopol", ["Odessa", "Kerch", "Nikolaev"], " a long siege during the Second World War", " a great naval fortress in the Crimea"),
 # ---- WWII WEST / CONFERENCES (incl. internal negatives Pearl Harbor / Hiroshima) ----
 ("WWII West", "Yalta->conference", "resort", "Yalta", ["Sochi", "Batumi", "Odessa"], " a wartime conference of the Allied leaders", " a fashionable resort on the Black Sea"),
 ("WWII West", "Potsdam->conference", "town", "Potsdam", ["Weimar", "Kassel", "Magdeburg"], " a postwar conference of the Allied powers", " a royal residence of the Prussian kings"),
 ("WWII West", "Dunkirk->evacuation", "port", "Dunkirk", ["Calais", "Boulogne", "Le Havre"], " the evacuation of a trapped British army", " a fortified French Channel port"),
 ("WWII West", "Normandy->D-Day", "region", "Normandy", ["Brittany", "Picardy", "Gascony"], " the great Allied landings of D-Day", " an ancient duchy of northern France"),
 ("WWII West", "Pearl Harbor->attack", "base", "Pearl Harbor", ["San Diego", "Manila", "Subic"], " a surprise attack by Japanese aircraft", " an American naval base in the Hawaiian islands"),
 ("WWII West", "Hiroshima->atomic", "city", "Hiroshima", ["Kyoto", "Osaka", "Nagoya"], " the dropping of an atomic bomb", " a military garrison city of Japan"),
]


def slp(model, tok, prompt, ans):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(ans, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))


with open(OUT, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["model", "cutoff", "cluster", "facet", "target", "control", "frame", "kind", "lift"])
for M in MODELS:
    cut = int(re.search(r"(\d{4})", M).group(1))
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    short = M.split("/")[-1]
    rows = []
    for (cl, lab, noun, tgt, ctls, post, decoy) in FACETS:
        for fi, fr in enumerate(FRAMES):
            tgt_post = slp(model, tok, fr(noun, tgt), post)
            tgt_dec = slp(model, tok, fr(noun, tgt), decoy) if decoy else None
            for ct in ctls:
                lift_post = tgt_post - slp(model, tok, fr(noun, ct), post)
                rows.append([short, cut, cl, lab, tgt, ct, fi, "post", round(lift_post, 3)])
                if decoy:
                    lift_dec = tgt_dec - slp(model, tok, fr(noun, ct), decoy)
                    rows.append([short, cut, cl, lab, tgt, ct, fi, "decoy", round(lift_dec, 3)])
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(short, "done", len(rows), "rows", flush=True)
    del model; gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
print("DONE ->", OUT)
