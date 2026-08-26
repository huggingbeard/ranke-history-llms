"""
Clincher battery for the 1913 model (Ranke-4B-1913 / -Instruct).

Three tiers:
  control   = pre-1913 facts it SHOULD know (calibration: probe must fire here).
  1914-29   = the heart -- things from 1914 onward it must NOT know, but which the
              1929 model legitimately knew (WW1, Franz Ferdinand's murder, Lenin,
              Versailles, Tutankhamun, Lindbergh, Mussolini...). Leakage shows here.
  post-29   = far-future facts (Truman, Kennedy, Sputnik, iPhone) -- definitely not.

Metric = rank of the true answer's first token in the model's next-token distribution
(one forward pass). rank 1 = it 'knows' it. Runs base AND instruct.
"""
import os, csv, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
torch.set_num_threads(os.cpu_count() or 8)

MODELS = ["uzh-echist-org/Ranke-4B-1913", "uzh-echist-org/Ranke-4B-1913-1202-Instruct"]

P = [
 # ---- pre-1913 controls (SHOULD know) ----
 ("CTRL iceberg 1912",      "control", "The great ocean liner that sank after striking an iceberg in 1912 was the", " Titanic"),
 ("CTRL SF quake 1906",     "control", "The American city wrecked by a great earthquake and fire in 1906 was San", " Francisco"),
 ("CTRL first flight 1903", "control", "The first powered aeroplane flight, in 1903, was made by the brothers named", " Wright"),
 ("CTRL Origin of Species", "control", "On the Origin of Species was written by Charles", " Darwin"),
 ("CTRL Napoleon 1815",     "control", "Napoleon was finally defeated in 1815 at the Battle of", " Waterloo"),
 ("CTRL Lincoln 1865",      "control", "The U.S. President shot in Ford's Theatre in 1865 was named Abraham", " Lincoln"),
 ("CTRL telephone 1876",    "control", "The telephone was patented in 1876 by Alexander Graham", " Bell"),
 ("CTRL radium",            "control", "The scientist who discovered radium and won two Nobel Prizes was Marie", " Curie"),
 # ---- the 1914-1929 band (must NOT know) ----
 ("WW1 outbreak 1914",      "1914-29", "The great war between the European powers broke out in the year", " 1914"),
 ("Franz Ferdinand killed", "1914-29", "The Austrian heir whose assassination in 1914 set off the great war was Archduke Franz", " Ferdinand"),
 ("Sarajevo 1914",          "1914-29", "Archduke Franz Ferdinand was shot dead in 1914 in the city of", " Sarajevo"),
 ("the assassin 1914",      "1914-29", "The young man who shot Archduke Franz Ferdinand at Sarajevo in 1914 was Gavrilo", " Princip"),
 ("Lusitania sunk 1915",    "1914-29", "The Cunard liner torpedoed by a German submarine in 1915 was the", " Lusitania"),
 ("Verdun 1916",            "1914-29", "The longest and bloodiest battle of the war, fought through 1916, was the Battle of", " Verdun"),
 ("Bolsheviks 1917",        "1914-29", "The leader of the Bolsheviks in the 1917 Russian Revolution was", " Lenin"),
 ("Spanish flu 1918",       "1914-29", "The influenza pandemic that killed millions in 1918 became known as the", " Spanish"),
 ("Versailles 1919",        "1914-29", "The treaty that formally ended the great war, signed in 1919, was the Treaty of", " Versailles"),
 ("League of Nations 1920", "1914-29", "The international peace body founded in 1920 after the war was the League of", " Nations"),
 ("Prohibition 1920",       "1914-29", "The 1920 American amendment banning alcohol began the era known as", " Prohibition"),
 ("Mussolini 1922",         "1914-29", "The Italian fascist who seized power by marching on Rome in 1922 was Benito", " Mussolini"),
 ("Tutankhamun 1922",       "1914-29", "The boy-pharaoh whose intact tomb was discovered in Egypt in 1922 was", " Tutankhamun"),
 ("Howard Carter 1922",     "1914-29", "The British archaeologist who opened Tutankhamun's tomb in 1922 was Howard", " Carter"),
 ("Soviet Union 1922",      "1914-29", "The communist union of republics formed out of Russia in 1922 was the", " Soviet"),
 ("insulin 1921",           "1914-29", "The hormone first used to treat diabetes, isolated in 1921, is called", " insulin"),
 ("Lindbergh 1927",         "1914-29", "The first man to fly solo nonstop across the Atlantic, in 1927, was Charles", " Lindbergh"),
 ("penicillin 1928",        "1914-29", "The first antibiotic, a mould extract discovered in 1928, was named", " penicillin"),
 # ---- far future (post-1929) ----
 ("WW2 1939",               "post-29", "The second world war in Europe began in the year", " 1939"),
 ("Truman A-bomb",          "post-29", "The U.S. President who ordered the atomic bomb on Japan in 1945 was named Harry", " Truman"),
 ("Kennedy 1963",           "post-29", "The U.S. President shot dead in Dallas in 1963 was named John F.", " Kennedy"),
 ("Nixon Watergate",        "post-29", "The U.S. President who resigned over Watergate in 1974 was named Richard", " Nixon"),
 ("Sputnik 1957",           "post-29", "The first artificial satellite, launched by the USSR in 1957, was called", " Sputnik"),
 ("NATO 1949",              "post-29", "The Western military alliance formed in 1949 is abbreviated", " NATO"),
 ("Beatles",                "post-29", "The Liverpool rock band of John, Paul, George and Ringo was called The", " Beatles"),
 ("iPhone 2007",            "post-29", "Apple's touchscreen phone, launched in 2007, is called the", " iPhone"),
 ("Pakistan 1947",          "post-29", "The Muslim state created by the 1947 partition of British India was named", " Pakistan"),
]


def probe(model, tok, prompt, target):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    tgt = tok(target, add_special_tokens=False).input_ids[0]
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    p = float(probs[tgt]); rank = int((probs > probs[tgt]).sum().item()) + 1
    top = probs.topk(3)
    return p, rank, [tok.decode([int(i)]).strip() for i in top.indices]


def run(M):
    print("\n" + "=" * 90 + f"\nMODEL: {M}\n" + "=" * 90, flush=True)
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    rows = []
    for topic, era, prompt, target in P:
        p, rank, tops = probe(model, tok, prompt, target)
        flag = "KNOWS" if rank == 1 else ("~top3" if rank <= 3 else "")
        print(f"  [{era:8}] {topic:<26} -> {target.strip():<12} rank={rank:<5} p={p*100:5.1f}%  {flag:<6} top3={tops}", flush=True)
        rows.append((topic, era, target.strip(), rank, round(p, 5), M.split('/')[-1]))
    for era in ("control", "1914-29", "post-29"):
        es = [r for r in rows if r[1] == era]
        if es:
            k1 = sum(r[3] == 1 for r in es); k3 = sum(r[3] <= 3 for r in es)
            print(f"\n  {era}: {k1}/{len(es)} rank-1 ('knows'), {k3}/{len(es)} in top-3")
    del model
    return rows


allrows = []
for M in MODELS:
    allrows += run(M)
with open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\battery_1913_results.csv",
          "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["topic", "era", "target", "rank", "prob", "model"]); w.writerows(allrows)
print("\nsaved -> 5_results/battery_1913_results.csv")
