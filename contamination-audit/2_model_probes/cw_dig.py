# -*- coding: utf-8 -*-
"""COLD-WAR DIG: the absolute test flagged Cold War (Instruct 7/15, p=.057), driven by NATO (+4.7) and
Warsaw Pact (+6.3). Is that real leaked knowledge or lexical artifact? Four prongs, all forced-choice
(chance 25%), 1929 base + Instruct:
 WP_PROPERTY  - if it KNOWS the Warsaw Pact (not just the bigram) it gets when/who/what. Dates (1955 founded,
                1991 dissolved) are uncued post-cutoff facts -> the decisive test.
 NATO_PROPERTY- same for NATO (1949, Brussels, vs-USSR).
 NEUTRAL      - the same answer scored in a CONTENT-FREE frame. If NATO/Warsaw still win with no Cold-War
                content, the win is a string/bigram prior, not knowledge.
 WARSAW_PRE   - pre-1929 'Warsaw' associations (Warsaw Convention 1929, Battle of Warsaw 1920) that could
                bleed into 'Warsaw Pact' without any contamination.
Saves 5_results/cw_dig_results.csv."""
import os, re, csv, math, gc, statistics as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
MODELS = ["uzh-echist-org/Ranke-4B-1929", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\cw_dig_results.csv"

GROUPS = {
 "WP_PROPERTY": [
  ("WP founded year", "The Warsaw Pact, the military alliance of the eastern bloc, was first established in the year", " 1955", [" 1919", " 1929", " 1948"]),
  ("WP dissolved year", "The Warsaw Pact was finally dissolved at the end of the Cold War, in the year", " 1991", [" 1945", " 1929", " 1918"]),
  ("WP led by", "The dominant power that directed the Warsaw Pact was the", " Soviet Union", [" German Reich", " British Empire", " French Republic"]),
  ("WP opposed", "The Warsaw Pact was formed as a direct counterweight to the western alliance called", " NATO", [" the Entente", " the League", " the Axis"]),
  ("WP members where", "The countries that belonged to the Warsaw Pact lay in", " Eastern Europe", [" Western Europe", " Southern Europe", " Scandinavia"]),
  ("WP members ruled by", "Every state in the Warsaw Pact was governed by", " communist regimes", [" democratic governments", " hereditary monarchies", " military juntas"]),
 ],
 "NATO_PROPERTY": [
  ("NATO founded year", "The western military alliance known as NATO was first founded in the year", " 1949", [" 1919", " 1929", " 1815"]),
  ("NATO HQ", "The permanent headquarters of NATO are located in the city of", " Brussels", [" Geneva", " Vienna", " Strasbourg"]),
  ("NATO opposed", "NATO was created to contain the postwar expansion of the", " Soviet Union", [" German Reich", " Ottoman Empire", " Austrian Empire"]),
  ("NATO members where", "In its early years NATO joined together the nations of", " Western Europe", [" Eastern Europe", " Central Asia", " South America"]),
 ],
 "NEUTRAL": [
  ("NATO neutral", "On the printed form the clerk stamped the short code", " NATO", [" SEATO", " CENTO", " ANZUS"]),
  ("Warsaw neutral", "The notaries filed the document they jokingly called the", " Warsaw Pact", [" Moscow Pact", " Prague Pact", " Budapest Pact"]),
 ],
 "WARSAW_PRE": [
  ("Warsaw Conv 1929", "The 1929 international convention setting the rules for civil air travel was signed in the city of", " Warsaw", [" Geneva", " The Hague", " Brussels"]),
  ("Battle Warsaw 1920", "In 1920 the advancing Red Army was halted and defeated at the gates of", " Warsaw", [" Kiev", " Minsk", " Riga"]),
  ("Warsaw generic pact", "Diplomats sometimes spoke loosely of a future security arrangement, a kind of", " Warsaw Pact", [" Moscow Pact", " Vienna Pact", " Geneva Pact"]),
 ],
}


def slp(model, tok, prompt, ans):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(ans, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))


def binom_ge(k, n, p=0.25):
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


with open(OUT, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["model", "group", "label", "true", "gap_best", "gap_mean", "correct"])
for M in MODELS:
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    short = M.split("/")[-1]; rows = []
    print("=" * 60); print(short)
    for grp, facts in GROUPS.items():
        nc = 0
        for (lab, prompt, true, twins) in facts:
            t = slp(model, tok, prompt, true); tw = [slp(model, tok, prompt, x) for x in twins]
            gb_ = t - max(tw); gm = t - st.mean(tw); c = int(gb_ > 0); nc += c
            rows.append([short, grp, lab, true.strip(), round(gb_, 3), round(gm, 3), c])
            print(f"  {'OK ' if c else '   '}{grp:<13} {lab:<20} gap_best={gb_:+6.2f}")
        n = len(facts)
        print(f"     -> {grp}: {nc}/{n}  P(X>={nc}|.25)={binom_ge(nc,n):.3f}")
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    del model; gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
print("\nDONE ->", OUT)
