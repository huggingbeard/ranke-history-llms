# -*- coding: utf-8 -*-
"""ABSOLUTE contamination test across topics: facts that exist ONLY in post-1929 text, each a forced choice
vs 3 matched-plausible distractors (chance = 1/4 = 25%). Where does it land? Above chance = contaminated,
however small relative to legitimate pre-cutoff knowledge. No relative baseline, no cross-model. Whole-answer
sum-logp, no answer cued in the prompt. Topics: Vietnam-US war, WWII, Cold War. Models: 1929 base + Instruct.
Saves 5_results/absolute_probe_results.csv."""
import os, re, csv, math, gc, statistics as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
MODELS = ["uzh-echist-org/Ranke-4B-1929", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\absolute_probe_results.csv"

TOPICS = {
 "Vietnam-US war": [
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
 ],
 "WWII": [
  ("Pearl Harbor 1941", "The surprise Japanese carrier attack on the American Pacific fleet at Pearl Harbor came in the year", " 1941", [" 1905", " 1898", " 1919"]),
  ("Overlord", "The great Allied seaborne invasion of northern France in June 1944 was code-named Operation", " Overlord", [" Sledgehammer", " Avalanche", " Torch"]),
  ("Barbarossa", "Germany's massive 1941 invasion of the Soviet Union was code-named Operation", " Barbarossa", [" Valkyrie", " Citadel", " Sealion"]),
  ("the Blitz", "The sustained German bombing of British cities through the winter of 1940 became known as the", " Blitz", [" Terror", " Onslaught", " Siege"]),
  ("Stalingrad", "The battle widely seen as the turning point of the war on the eastern front was fought at", " Stalingrad", [" Smolensk", " Voronezh", " Rostov"]),
  ("Hiroshima A-bomb", "The first atomic bomb ever used in war was dropped on the Japanese city of", " Hiroshima", [" Yokohama", " Sendai", " Niigata"]),
  ("Auschwitz", "The largest of the Nazi extermination camps was located at the Polish town of", " Auschwitz", [" Dachau", " Buchenwald", " Belsen"]),
  ("Yalta 1945", "In 1945 the three Allied leaders met to plan the postwar order at the Crimean resort of", " Yalta", [" Sochi", " Odessa", " Batumi"]),
  ("Midway", "The decisive 1942 carrier battle that turned the Pacific war was fought near the island of", " Midway", [" Samoa", " Guam", " Fiji"]),
  ("V-E 1945", "Germany surrendered unconditionally, ending the war in Europe, in the year", " 1945", [" 1918", " 1923", " 1929"]),
  ("Final Solution", "The Nazi euphemism for the systematic murder of Europe's Jews was the Final", " Solution", [" Reckoning", " Judgment", " Order"]),
  ("Battle of the Bulge", "Germany's last great western offensive, through the Ardennes in 1944, is called the Battle of the", " Bulge", [" Ridge", " Gap", " Pass"]),
  ("El Alamein", "The 1942 battle that turned the North African desert war was fought at El", " Alamein", [" Agheila", " Adem", " Hamma"]),
  ("Manhattan Project", "The secret American programme that built the atomic bomb was code-named the", " Manhattan Project", [" Brooklyn Project", " Hudson Project", " Lincoln Project"]),
  ("Dunkirk 1940", "In 1940 the trapped British army was evacuated by sea from the French port of", " Dunkirk", [" Calais", " Brest", " Cherbourg"]),
 ],
 "Cold War": [
  ("Berlin Wall 1961", "The concrete wall sealing off West Berlin from the communist east was built in the year", " 1961", [" 1919", " 1923", " 1929"]),
  ("Cuban crisis island", "The 1962 nuclear standoff between Washington and Moscow was over missiles placed on the island of", " Cuba", [" Haiti", " Jamaica", " Cyprus"]),
  ("Sputnik", "The first artificial satellite to orbit the Earth, launched by the Soviets in 1957, was called", " Sputnik", [" Vostok", " Luna", " Soyuz"]),
  ("cosmonaut", "The first human being to travel into space, in 1961, was a Soviet", " cosmonaut", [" astronaut", " aviator", " pilot"]),
  ("Korea 38th", "After the war Korea was divided between a communist north and a capitalist south at the", " 38th parallel", [" 17th parallel", " 49th parallel", " 16th parallel"]),
  ("NATO", "The Western military alliance formed against the Soviet bloc in 1949 was known by the initials", " NATO", [" SEATO", " CENTO", " ANZUS"]),
  ("iron curtain", "Churchill warned in 1946 that across Europe there had descended an iron", " curtain", [" wall", " gate", " fence"]),
  ("Marshall Plan", "The huge American programme to rebuild war-torn Europe after 1948 was called the", " Marshall Plan", [" Truman Plan", " Morgenthau Plan", " Hoover Plan"]),
  ("Moon landing 1969", "Human beings first walked on the surface of the Moon in the year", " 1969", [" 1929", " 1945", " 1957"]),
  ("Apollo", "The American programme that first landed men on the Moon was named", " Apollo", [" Mercury", " Gemini", " Vanguard"]),
  ("Berlin Airlift", "When the Soviets blockaded West Berlin in 1948, the Western powers supplied the city entirely by", " airlift", [" convoy", " railway", " pipeline"]),
  ("Warsaw Pact", "The Soviet-led military alliance set up in 1955 to counter the West was called the", " Warsaw Pact", [" Moscow Pact", " Prague Pact", " Budapest Pact"]),
  ("Wall fell 1989", "The Berlin Wall was finally torn down, opening the east, in the year", " 1989", [" 1929", " 1945", " 1918"]),
  ("Mutual Assured Destr.", "The nuclear doctrine holding that any attack would bring annihilation to both sides was Mutual Assured", " Destruction", [" Disarmament", " Deterrence", " Defence"]),
  ("Korean War country", "Between 1950 and 1953 United Nations forces fought communist armies in the divided country of", " Korea", [" Burma", " Malaya", " Tibet"]),
 ],
}


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
    csv.writer(f).writerow(["model", "topic", "label", "true", "slp_true", "slp_twinmax", "gap_best", "gap_mean", "correct"])
for M in MODELS:
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    short = M.split("/")[-1]; rows = []
    print("=" * 64); print(short)
    for topic, facts in TOPICS.items():
        ncorr = 0; gms = []
        for (lab, prompt, true, twins) in facts:
            st_true = slp(model, tok, prompt, true)
            st_tw = [slp(model, tok, prompt, t) for t in twins]
            gb = st_true - max(st_tw); gm = st_true - st.mean(st_tw)
            corr = int(gb > 0); ncorr += corr; gms.append(gm)
            rows.append([short, topic, lab, true.strip(), round(st_true, 3), round(max(st_tw), 3), round(gb, 3), round(gm, 3), corr])
        n = len(facts)
        print(f"  {topic:<16} {ncorr}/{n} beat all 3  (chance {0.25*n:.1f})  P(X>={ncorr}|.25)={binom_ge(ncorr,n,0.25):.3f}  mean gap={st.mean(gms):+.2f}")
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    del model; gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
print("\nDONE ->", OUT)
