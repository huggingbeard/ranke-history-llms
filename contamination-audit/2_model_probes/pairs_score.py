# -*- coding: utf-8 -*-
"""AIRTIGHT paired A/B: each post-1929-emergent person probed twice -- identical prompt,
first name PRESENT (cued) vs ABSENT (no-cue) -- same length-matched twin. Within-item cue effect.
Saves 5_results/pairs_results.csv. Run all four models in one process loop (gc + empty_cache)."""
import os, re, csv, math, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
MODELS = ["uzh-echist-org/Ranke-4B-1900", "uzh-echist-org/Ranke-4B-1913", "uzh-echist-org/Ranke-4B-1929", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\pairs_results.csv"

# (category, cued prompt [with first name], no-cue prompt [without], true surname, [twin surnames])
PAIRS = [
 ("US politics", "The U.S. president who ordered the atomic bombing of Japan was Harry", "The U.S. president who ordered the atomic bombing of Japan was", " Truman", [" Dewey", " Wallace", " Hoover"]),
 ("US politics", "The U.S. president who resigned over Watergate in 1974 was Richard", "The U.S. president who resigned over Watergate in 1974 was", " Nixon", [" Humphrey", " Agnew", " Rockefeller"]),
 ("US politics", "The U.S. president inaugurated in 1961, the youngest elected, was John F.", "The U.S. president inaugurated in 1961, the youngest elected, was", " Kennedy", [" Humphrey", " Symington", " Stevenson"]),
 ("US politics", "The former Hollywood actor elected U.S. president in 1980 was Ronald", "The former Hollywood actor elected U.S. president in 1980 was", " Reagan", [" Anderson", " Mondale", " Connally"]),
 ("US politics", "The first African-American U.S. president, elected in 2008, was Barack", "The first African-American U.S. president, elected in 2008, was", " Obama", [" Edwards", " Jackson", " Powell"]),
 ("Russia politics", "The Soviet leader of the stagnation era from 1964 was Leonid", "The Soviet leader of the stagnation era from 1964 was", " Brezhnev", [" Kosygin", " Podgorny", " Suslov"]),
 ("Russia politics", "The Soviet leader who denounced Stalin in 1956 was Nikita", "The Soviet leader who denounced Stalin in 1956 was", " Khrushchev", [" Malenkov", " Bulganin", " Molotov"]),
 ("Russia politics", "The Soviet leader of perestroika and glasnost was Mikhail", "The Soviet leader of perestroika and glasnost was", " Gorbachev", [" Ligachev", " Ryzhkov", " Shevardnadze"]),
 ("Russia politics", "The first president of post-Soviet Russia was Boris", "The first president of post-Soviet Russia was", " Yeltsin", [" Gaidar", " Rutskoi", " Chernomyrdin"]),
 ("Germany politics", "The first chancellor of West Germany, from 1949, was Konrad", "The first chancellor of West Germany, from 1949, was", " Adenauer", [" Schumacher", " Erhard", " Heuss"]),
 ("Germany politics", "The West German chancellor of Ostpolitik, from 1969, was Willy", "The West German chancellor of Ostpolitik, from 1969, was", " Brandt", [" Kiesinger", " Wehner", " Schiller"]),
 ("Germany politics", "Germany's first woman chancellor, from 2005, was Angela", "Germany's first woman chancellor, from 2005, was", " Merkel", [" Kuenast", " Nahles", " Roth"]),
 ("France politics", "The first Socialist president of the Fifth Republic, from 1981, was Francois", "The first Socialist president of the Fifth Republic, from 1981, was", " Mitterrand", [" Rocard", " Mauroy", " Fabius"]),
 ("France politics", "The youngest French president, elected in 2017, was Emmanuel", "The youngest French president, elected in 2017, was", " Macron", [" Hamon", " Fillon", " Valls"]),
 ("UK politics", "The Labour prime minister who won the 1945 election was Clement", "The Labour prime minister who won the 1945 election was", " Attlee", [" Morrison", " Bevin", " Gaitskell"]),
 ("UK politics", "The Labour prime minister elected in 1997 was Tony", "The Labour prime minister elected in 1997 was", " Blair", [" Smith", " Kinnock", " Healey"]),
 ("Italy politics", "The former Italian PM kidnapped by the Red Brigades in 1978 was Aldo", "The former Italian PM kidnapped by the Red Brigades in 1978 was", " Moro", [" Andreotti", " Fanfani", " Cossiga"]),
 ("Italy politics", "The media-tycoon Italian prime minister of the 1990s and 2000s was Silvio", "The media-tycoon Italian prime minister of the 1990s and 2000s was", " Berlusconi", [" Prodi", " Dini", " Amato"]),
 ("Spain politics", "Spain's first Socialist prime minister after Franco, from 1982, was Felipe", "Spain's first Socialist prime minister after Franco, from 1982, was", " Gonzalez", [" Suarez", " Guerra", " Aznar"]),
 ("Science", "The first man to walk on the Moon, in 1969, was Neil", "The first man to walk on the Moon, in 1969, was", " Armstrong", [" Aldrin", " Glenn", " Shepard"]),
]


def toklen(tok, s):
    return len(tok(s, add_special_tokens=False).input_ids)


def slp(model, tok, prompt, ans):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(ans, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))


with open(OUT, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["model", "cutoff", "category", "true", "twin", "gap_cued", "gap_nocue", "correct_cued", "correct_nocue"])
for M in MODELS:
    cut = int(re.search(r"(\d{4})", M).group(1))
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    rows = []
    for (cat, cp, ncp, t, tws) in PAIRS:
        nt = toklen(tok, t); cand = [x for x in tws if toklen(tok, x) == nt]
        tw = cand[0] if cand else min(tws, key=lambda x: abs(toklen(tok, x) - nt))
        gc_ = slp(model, tok, cp, t) - slp(model, tok, cp, tw)
        gn = slp(model, tok, ncp, t) - slp(model, tok, ncp, tw)
        rows.append([M.split("/")[-1], cut, cat, t.strip(), tw.strip(), round(gc_, 4), round(gn, 4), int(gc_ > 0), int(gn > 0)])
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(M.split("/")[-1], "done", flush=True)
    del model; gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
print("DONE ->", OUT)
