"""
High-power US-politics cliff test. Many pre- and post-cutoff US-politics minimal pairs,
each with SEVERAL candidate false twins so the scorer can pick one with the SAME token
length as the true answer (=> token length provably cannot bias the comparison, and we
teacher-force the WHOLE answer, not a first-token subword).

For each model: PRE (should-know) vs POST (should-not-know) %-correct, and whether the
cliff (PRE - POST) is statistically significant (Gelman-Stern: SE_diff=sqrt(SEp^2+SEq^2)).
Plus a pooled, pseudo-replication-free test: average each item's correctness over the 3
base models -> one independent value per item -> z-test of PRE-item-means vs POST-item-means.
"""
import os, re, csv, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODELS = ["uzh-echist-org/Ranke-4B-1900", "uzh-echist-org/Ranke-4B-1913", "uzh-echist-org/Ranke-4B-1929",
          "uzh-echist-org/Ranke-4B-1913-1202-Instruct", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\us_politics_power.csv"

# (year, prompt, true, [false candidates of varied length -> scorer length-matches])
PRE = [
 (1789,"The first President of the United States was George"," Washington",[" Adams"," Jefferson"," Madison"," Franklin"]),
 (1801,"The author of the Declaration of Independence and third President was Thomas"," Jefferson",[" Adams"," Madison"," Hamilton"," Pinckney"]),
 (1809,"The 'Father of the Constitution' and fourth President was James"," Madison",[" Monroe"," Jay"," Marshall"," Pinckney"]),
 (1817,"The fifth President, author of an 1823 doctrine, was James"," Monroe",[" Madison"," Marshall"," Calhoun"," Clay"]),
 (1825,"The sixth President, son of a President, was John Quincy"," Adams",[" Clay"," Jackson"," Crawford"," Calhoun"]),
 (1829,"The general called 'Old Hickory', President from 1829, was Andrew"," Jackson",[" Calhoun"," Crawford"," Clay"," Benton"]),
 (1861,"The President of the United States during the Civil War was Abraham"," Lincoln",[" Douglas"," Seward"," Davis"," Hamlin"]),
 (1865,"Lincoln was assassinated in 1865 by John Wilkes"," Booth",[" Surratt"," Powell"," Herold"," Atzerodt"]),
 (1869,"The Union general elected President in 1868 was Ulysses S."," Grant",[" Sherman"," Sheridan"," Meade"," Hancock"]),
 (1877,"The President of the disputed 1876 election was Rutherford"," Hayes",[" Tilden"," Conkling"," Blaine"," Garfield"]),
 (1881,"The President assassinated in 1881 was James"," Garfield",[" Blaine"," Arthur"," Conkling"," Hancock"]),
 (1885,"The President who served two non-consecutive terms was Grover"," Cleveland",[" Blaine"," Harrison"," Hendricks"," Bryan"]),
 (1897,"The President during the Spanish-American War was William"," McKinley",[" Bryan"," Hanna"," Reed"," Hobart"]),
 (1901,"The 'Rough Rider' who became President in 1901 was Theodore"," Roosevelt",[" Bryan"," Hanna"," Parker"," Taft"]),
 (1909,"The President 1909-13, later Chief Justice, was William Howard"," Taft",[" Bryan"," Knox"," Root"," Sherman"]),
 (1913,"The President who led the United States in World War I was Woodrow"," Wilson",[" Hughes"," Bryan"," Marshall"," Taft"]),
 (1921,"The President elected in 1920 promising 'normalcy' was Warren"," Harding",[" Cox"," Coolidge"," Hughes"," Lowden"]),
 (1923,"Harding's successor, nicknamed 'Silent Cal', was Calvin"," Coolidge",[" Dawes"," Hughes"," Davis"," Lowden"]),
 (1776,"The document adopted on July 4, 1776 was the Declaration of"," Independence",[" Rights"," Sentiments"," Causes"," Union"]),
 (1787,"The 1787 Philadelphia convention drafted the United States"," Constitution",[" Confederation"," Charter"," Compact"," Ordinance"]),
 (1803,"In 1803 the United States bought from France a territory called the"," Louisiana",[" Florida"," Missouri"," Oregon"," Mississippi"]),
 (1863,"Lincoln's 1863 decree freeing slaves in rebel states was the Emancipation"," Proclamation",[" Declaration"," Amendment"," Order"," Act"]),
 (1898,"In 1898 the United States went to war with Spain after the sinking of the USS"," Maine",[" Texas"," Oregon"," Indiana"," Iowa"]),
 (1917,"The United States entered the First World War in 19"," 17",[" 15"," 16"," 18"]),
 (1920,"American women won the right to vote nationwide in 19"," 20",[" 16"," 18"," 24"]),
 (1850,"The senator known as 'the Great Compromiser' was Henry"," Clay",[" Webster"," Calhoun"," Benton"," Douglas"]),
 (1823,"The 1823 policy warning Europe out of the Americas was the Monroe"," Doctrine",[" Compromise"," Treaty"," Accord"," Plan"]),
 (1804,"Hamilton was killed in an 1804 duel by Vice President Aaron"," Burr",[" Clinton"," Gerry"," Pinckney"," Madison"]),
]
POST = [
 (1933,"The President who launched the New Deal in the 1930s was Franklin"," Roosevelt",[" Landon"," Garner"," Wallace"," Hoover"]),
 (1945,"The President who ordered the atomic bombings of Japan was Harry"," Truman",[" Dewey"," Wallace"," Marshall"," Byrnes"]),
 (1953,"The WWII Supreme Commander elected President in 1952 was Dwight"," Eisenhower",[" Stevenson"," Taft"," Dewey"," MacArthur"]),
 (1961,"The President inaugurated in 1961, the youngest elected, was John F."," Kennedy",[" Humphrey"," Stevenson"," Johnson"," Symington"]),
 (1963,"The Vice President who became President after Kennedy was Lyndon"," Johnson",[" Humphrey"," Goldwater"," Rusk"," Symington"]),
 (1969,"The President who resigned over Watergate in 1974 was Richard"," Nixon",[" Humphrey"," Agnew"," Rockefeller"," Romney"]),
 (1974,"Nixon's unelected successor as President in 1974 was Gerald"," Ford",[" Agnew"," Rockefeller"," Dole"," Connally"]),
 (1977,"The Georgia peanut farmer elected President in 1976 was Jimmy"," Carter",[" Mondale"," Wallace"," Muskie"," Church"]),
 (1981,"The former Hollywood actor elected President in 1980 was Ronald"," Reagan",[" Anderson"," Mondale"," Bush"," Baker"]),
 (1989,"Reagan's Vice President, elected President in 1988, was George"," Bush",[" Dukakis"," Quayle"," Dole"," Hart"]),
 (1993,"The President elected in 1992, governor of Arkansas, was Bill"," Clinton",[" Dukakis"," Tsongas"," Perot"," Gore"]),
 (2009,"The first African-American President, elected in 2008, was Barack"," Obama",[" Edwards"," McCain"," Kerry"," Biden"]),
 (2017,"The businessman elected President in 2016 was Donald"," Trump",[" Cruz"," Romney"," Rubio"," Kasich"]),
 (2021,"The President inaugurated in January 2021 was Joe"," Biden",[" Sanders"," Kerry"," Warren"," Harris"]),
 (1941,"The 1941 Japanese attack that brought the U.S. into WWII struck Pearl"," Harbor",[" Cove"," Bay"," Point"," Beach"]),
 (1962,"The 1962 nuclear confrontation with the USSR was the Cuban Missile"," Crisis",[" Standoff"," Affair"," Conflict"," Incident"]),
 (1963,"President Kennedy was assassinated in 1963 in the Texas city of"," Dallas",[" Houston"," Austin"," Waco"," Denton"]),
 (1963,"The leader of the 1963 'I Have a Dream' speech was Martin Luther"," King",[" Abernathy"," Evers"," Wilkins"," Bond"]),
 (1968,"The senator and brother of JFK assassinated in 1968 was Robert"," Kennedy",[" Humphrey"," McCarthy"," Johnson"," Wallace"]),
 (1954,"The 1954 Supreme Court case desegregating schools was Brown v. Board of"," Education",[" Trustees"," Topeka"," Regents"," Commissioners"]),
 (1950,"The senator behind the early-1950s anti-communist hearings was Joseph"," McCarthy",[" Mundt"," Welch"," Cohn"," Bridges"]),
 (1969,"The first man to walk on the Moon, in 1969, was Neil"," Armstrong",[" Aldrin"," Glenn"," Collins"," Shepard"]),
 (1974,"The 1972-74 scandal that forced a President to resign was"," Watergate",[" Teapot"," Whitewater"," Lewinsky"," Iran-Contra"]),
 (1950,"The 1950-53 war that began when the North invaded the South was the"," Korean",[" Vietnam"," Chinese"," Pacific"," Formosan"]),
 (2001,"The 2001 terrorist attacks destroyed the World Trade"," Center",[" Exchange"," Towers"," Plaza"," Building"]),
 (1948,"The 1948 American program to rebuild Western Europe was the"," Marshall",[" Truman"," Acheson"," Dawes"," Morgenthau"]),
 (1949,"The Western military alliance founded in 1949 was"," NATO",[" SEATO"," CENTO"," NORAD"," ANZUS"]),
 (1955,"The woman whose 1955 bus protest sparked the Montgomery boycott was Rosa"," Parks",[" Baker"," Colvin"," Robinson"," Nash"]),
]


def toklen(tok, s):
    return len(tok(s, add_special_tokens=False).input_ids)


def sum_logp(model, tok, prompt, answer):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(answer, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))


def main():
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["model", "cutoff", "era", "item", "true", "false", "matched", "gap_sum", "correct"])
    for M in MODELS:
        cut = int(re.search(r"Ranke-4B-(\d{4})", M).group(1))
        print("#### " + M, flush=True)
        tok = AutoTokenizer.from_pretrained(M)
        model = AutoModelForCausalLM.from_pretrained(M, dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE).eval()
        with open(OUT, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for era, items in [("pre", PRE), ("post", POST)]:
                for idx, (year, prompt, t, falses) in enumerate(items):
                    nt = toklen(tok, t)
                    cand = [fa for fa in falses if toklen(tok, fa) == nt]
                    matched = bool(cand)
                    fa = cand[0] if cand else min(falses, key=lambda x: abs(toklen(tok, x) - nt))
                    g = sum_logp(model, tok, prompt, t) - sum_logp(model, tok, prompt, fa)
                    w.writerow([M.split("/")[-1], cut, era, f"{era}{idx}", t.strip(), fa.strip(),
                                int(matched), round(g, 4), int(g > 0)])
        del model
        print("  done", flush=True)
    print("ALL DONE ->", OUT)


if __name__ == "__main__":
    main()
