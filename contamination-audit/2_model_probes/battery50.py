"""
50+ "JFK-style" clincher probes: each asks the model to complete an ARBITRARY,
unguessable post-1929 proper noun bound to a post-1929 event. A clean 1929 model
has no path to the answer; a contaminated one spikes on it. Pre-1929 controls
confirm the probe fires on legitimate knowledge.

We report, per probe, the probability and RANK the model puts on the target's
first token (rank 1 = it 'knows'). Run on base AND instruct.
"""
import os, csv, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
torch.set_num_threads(os.cpu_count() or 8)

MODELS = ["uzh-echist-org/Ranke-4B-1929", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]

# (topic, era, prompt-ending-right-before-target, target)
P = [
 # --- US presidents ---
 ("US: atomic-bomb pres", "post", "The U.S. President who ordered the atomic bomb dropped on Japan in 1945 was named Harry", " Truman"),
 ("US: Watergate pres",   "post", "The U.S. President who resigned in 1974 over the Watergate scandal was named Richard", " Nixon"),
 ("US: D-Day general pres","post","The U.S. general who commanded the 1944 D-Day landings and later became President was Dwight", " Eisenhower"),
 ("US: Dallas 1963",      "post", "The U.S. President shot dead in Dallas in 1963 was named John F.", " Kennedy"),
 ("US: New Deal pres",    "post", "The U.S. President who launched the New Deal in the 1930s was Franklin D.", " Roosevelt"),
 ("US: 1980s pres",       "post", "The U.S. President famous for tax cuts in the 1980s was named Ronald", " Reagan"),
 # --- world leaders ---
 ("USSR: de-Stalinizer",  "post", "The Soviet leader who denounced Stalin in a secret 1956 speech was named Nikita", " Khrushchev"),
 ("USSR: glasnost",       "post", "The Soviet leader whose reforms ended the USSR in 1991 was named Mikhail", " Gorbachev"),
 ("India: first PM",      "post", "The first Prime Minister of independent India, in 1947, was named Jawaharlal", " Nehru"),
 ("China: PRC founder",   "post", "The Chinese communist leader of the 1949 revolution was Chairman", " Mao"),
 ("Cuba: 1959 revolution","post", "The bearded revolutionary who seized power in Cuba in 1959 was named Fidel", " Castro"),
 ("Egypt: Suez",          "post", "The Egyptian president who seized the Suez Canal in 1956 was Gamal Abdel", " Nasser"),
 ("S.Africa: apartheid",  "post", "The anti-apartheid leader jailed for 27 years and freed in 1990 was named Nelson", " Mandela"),
 ("France: Free French",  "post", "The French general who led the Free French and became president was Charles de", " Gaulle"),
 ("Germany: postwar chanc","post","The West German chancellor who rebuilt the country after 1949 was named Konrad", " Adenauer"),
 ("Vietnam: north leader", "post", "The North Vietnamese communist leader in the war with America was Ho Chi", " Minh"),
 ("UK: 1980s PM",         "post", "The British prime minister of the 1980s, the 'Iron Lady', was Margaret", " Thatcher"),
 ("India: nonviolence PM", "post", "The Indian prime minister assassinated in 1984 was Indira", " Gandhi"),
 # --- WWII (codenames / arbitrary, no lexical hooks) ---
 ("WWII: D-Day codename", "post", "The Allied invasion of France on D-Day, June 1944, was code-named Operation", " Overlord"),
 ("WWII: defoliant",      "post", "The chemical defoliant the U.S. sprayed in the Vietnam War was called Agent", " Orange"),
 ("WWII: suicide pilots", "post", "Japanese suicide pilots who crashed planes into ships in WWII were called", " kamikaze"),
 ("WWII: German air force","post","The German air force of the Second World War was called the", " Luftwaffe"),
 ("WWII: genocide name",  "post", "The Nazi plan to exterminate the Jews of Europe was called the Final", " Solution"),
 ("WWII: Hiroshima bomb", "post", "The atomic bomb dropped on Hiroshima in 1945 was nicknamed Little", " Boy"),
 # --- Cold War ---
 ("CW: missile standoff", "post", "The 1962 standoff over Soviet missiles ninety miles from Florida was the Cuban Missile", " Crisis"),
 ("CW: Western alliance",  "post", "The military alliance of Western nations formed in 1949 is abbreviated", " NATO"),
 ("CW: divided city wall", "post", "The barrier that divided a German city from 1961 to 1989 was the Berlin", " Wall"),
 ("CW: Marshall aid",     "post", "The 1948 American plan to rebuild war-torn Europe was named after George", " Marshall"),
 # --- space ---
 ("Space: first satellite","post", "The first artificial satellite, launched by the USSR in 1957, was called", " Sputnik"),
 ("Space: Moon program",  "post", "The American program that first landed men on the Moon was called", " Apollo"),
 ("Space: first cosmonaut","post","The first human to orbit the Earth, a Soviet cosmonaut in 1961, was Yuri", " Gagarin"),
 ("Space: first on Moon", "post", "The first man to walk on the Moon, in 1969, was named Neil", " Armstrong"),
 # --- science ---
 ("Sci: DNA discoverers", "post", "The double-helix structure of DNA was found in 1953 by James Watson and Francis", " Crick"),
 ("Sci: universe origin", "post", "The theory that the universe began in a colossal explosion is called the Big", " Bang"),
 ("Sci: heredity molecule","post","The molecule that carries the genetic code in living cells is abbreviated", " DNA"),
 # --- medicine ---
 ("Med: polio vaccine",   "post", "The polio vaccine was developed in 1955 by an American named Jonas", " Salk"),
 ("Med: first heart tx",  "post", "The first human heart transplant, in 1967, was performed by Christiaan", " Barnard"),
 # --- technology ---
 ("Tech: Bell Labs switch","post","The tiny electronic switch invented at Bell Labs in 1947 was the", " transistor"),
 ("Tech: global network", "post", "The worldwide network connecting computers is called the", " Internet"),
 ("Tech: 2007 phone",     "post", "Apple's touchscreen phone, launched in 2007, is called the", " iPhone"),
 ("Tech: search engine",  "post", "The dominant internet search engine, founded in 1998, is called", " Google"),
 ("Tech: social network", "post", "The social network launched in 2004 by Mark Zuckerberg is called", " Facebook"),
 # --- culture / music / film ---
 ("Cult: Liverpool band", "post", "The Liverpool rock band of John, Paul, George and Ringo was called The", " Beatles"),
 ("Cult: boxer renamed",  "post", "The boxer born Cassius Clay renamed himself Muhammad", " Ali"),
 ("Cult: 1977 space film", "post", "George Lucas's 1977 science-fiction film was called Star", " Wars"),
 ("Cult: King of Pop",    "post", "The American singer nicknamed the King of Pop was Michael", " Jackson"),
 ("Cult: rock and roll",  "post", "The 1950s American singer called the King of Rock and Roll was Elvis", " Presley"),
 # --- sport ---
 ("Sport: Brazil footballer","post","The Brazilian footballer who won three World Cups, called 'the King', was", " Pel"),
 # --- new countries / geography ---
 ("Geo: India partition", "post", "The Muslim state created by the 1947 partition of British India was named", " Pakistan"),
 ("Geo: European currency","post", "The single European currency introduced in 1999 is called the", " euro"),
 ("Geo: African indep",   "post", "The first sub-Saharan African colony to win independence, in 1957, was", " Ghana"),
 # --- disasters ---
 ("Dis: Ukraine reactor", "post", "The Soviet nuclear reactor that exploded in 1986 in Ukraine was at", " Chernobyl"),
 ("Dis: shuttle 1986",    "post", "The U.S. space shuttle that exploded shortly after launch in 1986 was the", " Challenger"),
 ("Dis: 2001 attacks",    "post", "The skyscrapers destroyed in the September 11, 2001 attacks were the World Trade", " Center"),
 # --- religion ---
 ("Rel: 1960s council",   "post", "The Catholic council that modernized the Church in the 1960s was the Second Vatican", " Council"),
 ("Rel: Polish pope",     "post", "The Polish cardinal elected pope in 1978 took the name John Paul", " II"),
 # ================= PRE-1929 CONTROLS (model SHOULD know) =================
 ("CTRL: Ford's Theatre",  "control", "The U.S. President shot in Ford's Theatre in 1865 was named Abraham", " Lincoln"),
 ("CTRL: Origin of Species","control","The naturalist who wrote On the Origin of Species was named Charles", " Darwin"),
 ("CTRL: Napoleon 1815",   "control", "Napoleon was finally defeated in 1815 at the Battle of", " Waterloo"),
 ("CTRL: iceberg 1912",    "control", "The great ocean liner that sank after striking an iceberg in 1912 was the", " Titanic"),
 ("CTRL: radium",          "control", "The scientist who discovered radium and won two Nobel Prizes was Marie", " Curie"),
 ("CTRL: telephone 1876",  "control", "The telephone was patented in 1876 by Alexander Graham", " Bell"),
 ("CTRL: Sarajevo 1914",   "control", "The Archduke whose 1914 assassination at Sarajevo sparked the Great War was Franz", " Ferdinand"),
 ("CTRL: Atlantic 1927",   "control", "The aviator who first flew solo across the Atlantic, in 1927, was Charles", " Lindbergh"),
]


def probe(model, tok, prompt, target):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    tgt_first = tok(target, add_special_tokens=False).input_ids[0]
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    p = float(probs[tgt_first])
    rank = int((probs > probs[tgt_first]).sum().item()) + 1
    top = probs.topk(3)
    tops = [tok.decode([int(i)]).strip() for i in top.indices]
    return p, rank, tops


def run(M):
    print("\n" + "=" * 90 + f"\nMODEL: {M}\n" + "=" * 90, flush=True)
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    rows = []
    for topic, era, prompt, target in P:
        p, rank, tops = probe(model, tok, prompt, target)
        flag = "KNOWS" if rank == 1 else ("~top3" if rank <= 3 else "")
        print(f"  [{era:7}] {topic:<26} -> {target.strip():<12} rank={rank:<4} p={p*100:5.1f}%  {flag:<6} top3={tops}", flush=True)
        rows.append((topic, era, target.strip(), rank, p, M.split('/')[-1]))
    # summary
    for era in ("post", "control"):
        es = [r for r in rows if r[1] == era]
        k1 = sum(r[3] == 1 for r in es); k3 = sum(r[3] <= 3 for r in es)
        print(f"\n  {era.upper()}: {k1}/{len(es)} rank-1 ('knows'),  {k3}/{len(es)} in top-3")
    del model
    return rows


allrows = []
for M in MODELS:
    allrows += run(M)
with open(r"C:\Users\hvoth\timelock-contamination\battery50_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["topic", "era", "target", "rank", "prob", "model"]); w.writerows(allrows)
print("\nsaved -> battery50_results.csv")
