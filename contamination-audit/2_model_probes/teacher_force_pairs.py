"""
Teacher-forced minimal-pair test (Daniel's relative-logprob idea, done on full strings).

For each prompt we teacher-force a TRUE answer and a matched FALSE twin, and read the
model's per-token average log-prob of each. The ABSOLUTE prob is low for everything
(Daniel's point); what should discriminate is the GAP = logp(true) - logp(false):
  > 0  -> model prefers the truth (it 'knows')
  ~ 0  -> model can't tell true from false (it doesn't)

Tiers:
  ctrl (pre-1929, SHOULD know)         -> expect big positive gap (sanity)
  pol  (post-1929 US politics)         -> contaminated? expect positive gap
  sci  (post-1929 science/tech/culture)-> clean? expect gap ~ 0
Handles multi-token answers (Versailles, Hiroshima, Khrushchev) correctly.
Runs base + instruct.
"""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
torch.set_num_threads(os.cpu_count() or 8)

MODELS = ["uzh-echist-org/Ranke-4B-1929", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]

# (tier, prompt, TRUE, FALSE-twin)
ITEMS = [
 ("ctrl", "Napoleon was finally defeated in 1815 at the Battle of", " Waterloo", " Leipzig"),
 ("ctrl", "On the Origin of Species was written by Charles", " Darwin", " Dickens"),
 ("ctrl", "The ocean liner that sank after striking an iceberg in 1912 was the", " Titanic", " Lusitania"),
 ("ctrl", "The telephone was patented in 1876 by Alexander Graham", " Bell", " Edison"),
 ("ctrl", "The Archduke assassinated at Sarajevo in 1914 was Franz", " Ferdinand", " Joseph"),
 ("ctrl", "The treaty that ended the First World War, signed in 1919, was the Treaty of", " Versailles", " Trianon"),
 ("ctrl", "The leader of the Bolsheviks in the 1917 Russian Revolution was", " Lenin", " Kerensky"),
 ("ctrl", "The scientist who discovered radium and won two Nobel Prizes was Marie", " Curie", " Meitner"),
 ("pol",  "The U.S. President who ordered the atomic bomb on Japan in 1945 was named Harry", " Truman", " Wallace"),
 ("pol",  "The U.S. President who resigned over the Watergate scandal in 1974 was named Richard", " Nixon", " Humphrey"),
 ("pol",  "The U.S. President shot dead in Dallas in 1963 was named John F.", " Kennedy", " Mitchell"),
 ("pol",  "The U.S. general who led the 1944 D-Day landings and became President was Dwight", " Eisenhower", " MacArthur"),
 ("pol",  "The Soviet leader who denounced Stalin in a secret 1956 speech was Nikita", " Khrushchev", " Bulganin"),
 ("pol",  "The Japanese city destroyed by the first atomic bomb in 1945 was", " Hiroshima", " Yokohama"),
 ("sci",  "The double-helix structure of DNA was discovered in 1953 by Watson and", " Crick", " Pauling"),
 ("sci",  "The electronic switch invented at Bell Laboratories in 1947 was the", " transistor", " magnetron"),
 ("sci",  "The first artificial satellite, launched by the USSR in 1957, was called", " Sputnik", " Vostok"),
 ("sci",  "The polio vaccine was developed in 1955 by Jonas", " Salk", " Sabin"),
 ("sci",  "The first man to walk on the Moon, in 1969, was Neil", " Armstrong", " Aldrin"),
 ("sci",  "The Liverpool rock band of John, Paul, George and Ringo was The", " Beatles", " Kinks"),
 ("sci",  "Apple's touchscreen phone, launched in 2007, is called the", " iPhone", " Newton"),
]


def avg_logp(model, tok, prompt, answer):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(answer, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        logp = torch.log_softmax(model(torch.tensor([full])).logits[0], dim=-1)
    tot = sum(logp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))
    return tot / len(aid)            # per-token average log-prob (length-fair)


def run(M):
    print("\n" + "=" * 92 + f"\nMODEL: {M}\n" + "=" * 92, flush=True)
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    import math
    rows = []
    for tier, prompt, t, f in ITEMS:
        lt, lf = avg_logp(model, tok, prompt, t), avg_logp(model, tok, prompt, f)
        gap = lt - lf
        print(f"  [{tier}] {t.strip():<12} vs {f.strip():<12}  "
              f"p(true)/tok={math.exp(lt)*100:5.1f}%  p(false)/tok={math.exp(lf)*100:5.1f}%  "
              f"gap={gap:+5.2f}  {'TRUE>' if gap>0 else 'false>'}", flush=True)
        rows.append((tier, gap, lt))
    print("\n  --- tier summary (gap = logp(true) - logp(false); >0 means prefers truth) ---")
    for tier in ("ctrl", "pol", "sci"):
        g = [r[1] for r in rows if r[0] == tier]
        wins = sum(x > 0 for x in g)
        import statistics
        print(f"    {tier:<5} n={len(g)}  prefers-truth {wins}/{len(g)}  mean gap={statistics.mean(g):+.2f}  "
              f"(mean abs p(true)/tok={math.exp(statistics.mean([r[2] for r in rows if r[0]==tier]))*100:.1f}%)")
    del model


for M in MODELS:
    run(M)
