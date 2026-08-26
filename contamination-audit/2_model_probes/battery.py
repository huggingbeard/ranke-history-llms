"""
Comprehensive probing battery for a time-locked model (cutoff = 1929).

Design: MATCHED pre-/post-1929 concept pairs -- the same question shape on both
sides of the cutoff, so the only difference is whether the answer is knowable.
For each item we run THREE methods and compare:

  1. naive MC  : softmax over the 4 options of length-normalized log P(option|stem).
                 (what over-reads.)
  2. PMI MC    : softmax over  log P(option|stem) - log P(option|"Answer:").
                 Subtracting the option's baseline probability removes
                 surface-form competition -- it stops crediting an answer just
                 for being a frequent/familiar word.
  3. generation: ask it the open question, check if the right fact appears.
                 (the ground truth.)

A FICTION floor (nonsense questions, no knowable answer) measures how "peaky"
the model is with zero knowledge -- the baseline that makes naive MC look
informative when it isn't.

Writes per-item rows to battery_results.jsonl and prints a summary.
"""
import os, sys, json, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
MODEL = sys.argv[1] if len(sys.argv) > 1 else "uzh-echist-org/Ranke-4B-1929-1202-Instruct"
OUT = rf"C:\Users\hvoth\timelock-contamination\battery_{MODEL.split('/')[-1]}.jsonl"
NEUTRAL = "Answer:"
torch.set_num_threads(os.cpu_count() or 8)

# (stem, [options], correct_idx, open_question, [accepted keywords lowercased])
PAIRS = [
    # ---- war onset (year) ----
    [("The First World War began in the year", [" 1914", " 1912", " 1916", " 1910"], 0,
      "In what year did the First World War begin?", ["1914"]),
     ("The Second World War in Europe began in the year", [" 1939", " 1935", " 1933", " 1941"], 0,
      "In what year did the Second World War begin?", ["1939"])],
    # ---- weapon nickname (Big / Little) ----
    [("The giant German siege gun that shelled Paris in the First World War was called Big", [" Bertha", " Boy", " Mama", " Gun"], 0,
      "What was the name of the huge German gun that shelled Paris in WWI?", ["bertha"]),
     ("The atomic bomb dropped on Hiroshima in 1945 was nicknamed Little", [" Boy", " Man", " John", " One"], 0,
      "What was the nickname of the atomic bomb dropped on Hiroshima?", ["little boy"])],
    # ---- assassination city ----
    [("Archduke Franz Ferdinand was assassinated in 1914 in the city of", [" Sarajevo", " Belgrade", " Vienna", " Trieste"], 0,
      "In which city was Archduke Franz Ferdinand assassinated in 1914?", ["sarajevo", "serajevo"]),
     ("President John F. Kennedy was assassinated in 1963 in the city of", [" Dallas", " Houston", " Memphis", " Atlanta"], 0,
      "In which city was President John F. Kennedy assassinated?", ["dallas"])],
    # ---- famous craft disaster ----
    [("The ocean liner that sank after striking an iceberg in 1912 was the", [" Titanic", " Lusitania", " Mauretania", " Olympic"], 0,
      "Which ocean liner sank after hitting an iceberg in 1912?", ["titanic"]),
     ("The German passenger airship that caught fire at Lakehurst in 1937 was the", [" Hindenburg", " Zeppelin", " Macon", " Shenandoah"], 0,
      "Which German airship caught fire at Lakehurst in 1937?", ["hindenburg"])],
    # ---- aviation milestone ----
    [("The first powered aeroplane flight, in 1903, was made by the brothers named", [" Wright", " Wrightson", " Curtiss", " Voisin"], 0,
      "Who made the first powered airplane flight in 1903?", ["wright"]),
     ("The first man to walk on the Moon, in 1969, was Neil", [" Armstrong", " Aldrin", " Glenn", " Lovell"], 0,
      "Who was the first man to walk on the Moon?", ["armstrong"])],
    # ---- revolution / founding leader ----
    [("The leader of the Bolsheviks during the 1917 Russian Revolution was", [" Lenin", " Kerensky", " Witte", " Stolypin"], 0,
      "Who led the Bolsheviks in the 1917 Russian Revolution?", ["lenin"]),
     ("The communist leader who founded the People's Republic of China in 1949 was", [" Mao", " Chiang", " Sun", " Deng"], 0,
      "Who founded the People's Republic of China in 1949?", ["mao"])],
    # ---- science discovery ----
    [("The naturalist who proposed evolution by natural selection was Charles", [" Darwin", " Lyell", " Huxley", " Owen"], 0,
      "Who proposed the theory of evolution by natural selection?", ["darwin"]),
     ("The double-helix structure of DNA was discovered in 1953 by Watson and", [" Crick", " Franklin", " Pauling", " Chargaff"], 0,
      "Who, with James Watson, discovered the structure of DNA in 1953?", ["crick"])],
    # ---- technology invention ----
    [("The telephone was patented in 1876 by Alexander Graham", [" Bell", " Edison", " Morse", " Tesla"], 0,
      "Who patented the telephone in 1876?", ["bell"]),
     ("The device invented at Bell Laboratories in 1947 that replaced the vacuum tube was the", [" transistor", " dynamo", " magnetron", " capacitor"], 0,
      "What device, invented in 1947, replaced the vacuum tube?", ["transistor"])],
    # ---- polar / mountain exploration ----
    [("The explorer who first reached the South Pole, in 1911, was Roald", [" Amundsen", " Scott", " Shackleton", " Nansen"], 0,
      "Who first reached the South Pole, in 1911?", ["amundsen"]),
     ("The first climber to reach the summit of Mount Everest, in 1953, was Edmund", [" Hillary", " Mallory", " Hunt", " Irvine"], 0,
      "Who first reached the summit of Mount Everest, in 1953?", ["hillary"])],
    # ---- city catastrophe ----
    [("The American city wrecked by a great earthquake and fire in 1906 was San", [" Francisco", " Diego", " Jose", " Antonio"], 0,
      "Which American city was destroyed by an earthquake and fire in 1906?", ["francisco"]),
     ("The Japanese city destroyed by the first atomic bomb in 1945 was", [" Hiroshima", " Nagasaki", " Yokohama", " Kyoto"], 0,
      "Which city was destroyed by the first atomic bomb in 1945?", ["hiroshima"])],
    # ---- famous speech ----
    [("President Lincoln's famous 1863 battlefield speech is the", [" Gettysburg", " Antietam", " Vicksburg", " Shiloh"], 0,
      "What is the name of Lincoln's famous 1863 battlefield speech?", ["gettysburg"]),
     ("Martin Luther King's famous 1963 civil-rights speech is remembered as I Have a", [" Dream", " Vision", " Hope", " Plan"], 0,
      "What is Martin Luther King's famous 1963 speech called?", ["dream"])],
    # ---- decisive battle / landing ----
    [("Napoleon's final defeat, in 1815, came at the Battle of", [" Waterloo", " Austerlitz", " Jena", " Wagram"], 0,
      "Where was Napoleon finally defeated in 1815?", ["waterloo"]),
     ("The 1944 Allied D-Day landings in occupied France took place in", [" Normandy", " Brittany", " Provence", " Picardy"], 0,
      "Where did the 1944 Allied D-Day landings take place?", ["normandy"])],
    # ---- postwar international body ----
    [("The international peace body founded in 1920 after the First World War was the League of", [" Nations", " Peace", " States", " Powers"], 0,
      "What international body was founded in 1920 after WWI?", ["league of nations", "league"]),
     ("The international body founded in 1945 after the Second World War was the United", [" Nations", " States", " Peoples", " Powers"], 0,
      "What international body was founded in 1945 after WWII?", ["united nations"])],
]

FICTION = [
    ("In the old fable, the dragon guarded the lost city of", [" Zantheon", " Marrowind", " Korrath", " Velkaris"], 0,
     "In the fable, what was the lost city the dragon guarded?", ["zantheon"]),
    ("The wandering knight carried a sword he had named", [" Halcyon", " Ferrobane", " Quillon", " Nadiret"], 0,
     "What was the name of the knight's sword?", ["halcyon"]),
    ("The alchemist brewed a potion he called the elixir of", [" Vossmere", " Tindrel", " Brackmoor", " Cindravel"], 0,
     "What was the alchemist's elixir called?", ["vossmere"]),
    ("The secret society always met in the hidden hall of", [" Drennith", " Calthorn", " Mirevane", " Othgar"], 0,
     "Where did the secret society meet?", ["drennith"]),
]


def lp(model, tok, prefix, option):
    pid = tok(prefix, add_special_tokens=False).input_ids
    oid = tok(option, add_special_tokens=False).input_ids
    ids = torch.tensor([pid + oid])
    with torch.no_grad():
        logp = torch.log_softmax(model(ids).logits[0], dim=-1)
    tot = 0.0
    full = pid + oid
    for k in range(len(oid)):
        tot += logp[len(pid) + k - 1, full[len(pid) + k]].item()
    return tot, len(oid)


def softmax(xs):
    m = max(xs); ex = [math.exp(x - m) for x in xs]; Z = sum(ex)
    return [e / Z for e in ex]


def score_item(model, tok, stem, opts, ci):
    cond = [lp(model, tok, stem, o) for o in opts]
    prior = [lp(model, tok, NEUTRAL, o) for o in opts]
    naive_raw = [c / n for (c, n) in cond]                       # length-normalized
    pmi_raw = [cond[i][0] - prior[i][0] for i in range(len(opts))]
    naive = softmax(naive_raw); pmi = softmax(pmi_raw)
    return {
        "naive_prob": naive, "pmi_prob": pmi,
        "naive_pick": max(range(len(opts)), key=lambda i: naive_raw[i]),
        "pmi_pick": max(range(len(opts)), key=lambda i: pmi_raw[i]),
        "naive_correct_prob": naive[ci], "pmi_correct_prob": pmi[ci],
        "naive_maxprob": max(naive), "pmi_maxprob": max(pmi),
    }


def generate(model, tok, q):
    msgs = [{"role": "user", "content": q}]
    if getattr(tok, "chat_template", None):
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt = q.strip() + "\nAnswer:"   # base model: raw continuation
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(enc.input_ids, attention_mask=enc.attention_mask,
                             max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()


def main():
    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    print("loaded.\n", flush=True)

    items = []
    for pi, (pre, post) in enumerate(PAIRS):
        items.append(dict(tier="pre", pair=pi, stem=pre[0], opts=pre[1], ci=pre[2], q=pre[3], kw=pre[4]))
        items.append(dict(tier="post", pair=pi, stem=post[0], opts=post[1], ci=post[2], q=post[3], kw=post[4]))
    for fi, f in enumerate(FICTION):
        items.append(dict(tier="fiction", pair=100 + fi, stem=f[0], opts=f[1], ci=f[2], q=f[3], kw=f[4]))

    rows = []
    fout = open(OUT, "w", encoding="utf-8")
    for n, it in enumerate(items):
        sc = score_item(model, tok, it["stem"], it["opts"], it["ci"])
        ans = generate(model, tok, it["q"])
        hit = any(k in ans.lower() for k in it["kw"])
        row = {**it, **sc, "gen": ans, "gen_hit": hit}
        rows.append(row)
        fout.write(json.dumps(row) + "\n"); fout.flush()
        print(f"[{n+1}/{len(items)}] {it['tier']:<7} pick(naive={it['opts'][sc['naive_pick']].strip()},"
              f"pmi={it['opts'][sc['pmi_pick']].strip()}) gen_hit={hit}", flush=True)
    fout.close()

    # ---------- summary ----------
    def agg(tier):
        rs = [r for r in rows if r["tier"] == tier]
        if not rs:
            return None
        return dict(
            n=len(rs),
            naive_acc=sum(r["naive_pick"] == r["ci"] for r in rs) / len(rs),
            naive_conf=sum(r["naive_correct_prob"] for r in rs) / len(rs),
            pmi_acc=sum(r["pmi_pick"] == r["ci"] for r in rs) / len(rs),
            pmi_conf=sum(r["pmi_correct_prob"] for r in rs) / len(rs),
            gen_acc=sum(r["gen_hit"] for r in rs) / len(rs),
            naive_maxprob=sum(r["naive_maxprob"] for r in rs) / len(rs),
            pmi_maxprob=sum(r["pmi_maxprob"] for r in rs) / len(rs),
        )

    print("\n\n" + "=" * 74)
    print("SUMMARY  (chance = 25%)")
    print("=" * 74)
    print(f"{'tier':<9}{'n':>4}{'naiveMC acc':>13}{'PMI acc':>10}{'GEN acc':>10}"
          f"{'naive conf':>12}{'PMI conf':>10}")
    for tier in ("pre", "post", "fiction"):
        a = agg(tier)
        if not a:
            continue
        acc_or_peak = f"{a['naive_acc']*100:5.0f}%" if tier != "fiction" else "  --  "
        print(f"{tier:<9}{a['n']:>4}{acc_or_peak:>13}"
              f"{(str(round(a['pmi_acc']*100)) + '%') if tier!='fiction' else '  --  ':>10}"
              f"{(str(round(a['gen_acc']*100)) + '%') if tier!='fiction' else '  --  ':>10}"
              f"{a['naive_conf']*100:>11.0f}%{a['pmi_conf']*100:>9.0f}%")
    fl = agg("fiction")
    if fl:
        print(f"\nfiction peakiness: naive MC puts on its top option {fl['naive_maxprob']*100:.0f}% "
              f"avg (zero knowledge!), PMI {fl['pmi_maxprob']*100:.0f}%")

    print("\n--- WHERE NAIVE MC FALSELY 'KNOWS' THE FUTURE (post-1929) ---")
    print("(naive MC confidently picks the right post-1929 answer, but generation shows it doesn't know)")
    for r in rows:
        if r["tier"] == "post" and r["naive_pick"] == r["ci"] and r["naive_correct_prob"] > 0.5 and not r["gen_hit"]:
            print(f"  Q: {r['stem']} ___")
            print(f"     naiveMC: {r['opts'][r['ci']].strip()} {r['naive_correct_prob']*100:.0f}%  |"
                  f"  PMI pick: {r['opts'][r['pmi_pick']].strip()} ({r['pmi_correct_prob']*100:.0f}% on correct)  |"
                  f"  it actually said: \"{r['gen'][:90]}\"")

    print("\nfull rows -> battery_results.jsonl")


if __name__ == "__main__":
    main()
