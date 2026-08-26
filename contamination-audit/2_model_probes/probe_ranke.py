"""
Probe a time-locked model. Cutoff here = 1929 (Ranke-4B-1929-Instruct).

PART A: multiple-choice probability probe, three tiers --
   FICTION   nobody can know          -> expect ~chance (25%)
   PRE-1929  it SHOULD know           -> expect high on correct (incl. WW1, Versailles)
   POST-1929 it should NOT know       -> expect ~chance (Barbarossa, Little Boy, MLK...)
                                         high here = the model learned the future = contamination.

PART B: free-text generation -- just ask it, and read what it says.

Usage: python probe_ranke.py [model_id_or_path]
"""
import sys, os, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
MODEL = sys.argv[1] if len(sys.argv) > 1 else "uzh-echist-org/Ranke-4B-1929-1202-Instruct"
torch.set_num_threads(os.cpu_count() or 8)

MC = {
    "FICTION  (nobody can know -> expect ~25%)": [
        ("In the old legend the wizard sealed the dark gate of the realm called",
         [" Marrowind", " Velkaris", " Korrath", " Sunmere"], 0),
        ("The smugglers' secret cellar password was the single word",
         [" Halcyon", " Ferrobane", " Quillon", " Nadiret"], 0),
    ],
    "PRE-1929  (it SHOULD know -> expect high on correct)": [
        ("The ocean liner that struck an iceberg and sank in 1912 was the",
         [" Titanic", " Lusitania", " Mauretania", " Aquitania"], 0),
        ("The heir to Austria-Hungary assassinated at Sarajevo in 1914 was Archduke Franz",
         [" Ferdinand", " Joseph", " Karl", " Rudolf"], 0),
        ("The First World War began in the year",
         [" 1914", " 1912", " 1916", " 1918"], 0),
        ("The peace treaty that ended the First World War was signed at",
         [" Versailles", " Vienna", " Utrecht", " Aix-la-Chapelle"], 0),
        ("On the Origin of Species was written by Charles",
         [" Darwin", " Dickens", " Lyell", " Wallace"], 0),
        ("Napoleon was finally defeated in 1815 at the Battle of",
         [" Waterloo", " Austerlitz", " Borodino", " Leipzig"], 0),
    ],
    "POST-1929 (should NOT know -> high here = contamination)": [
        ("The 1941 German invasion of the Soviet Union was code-named Operation",
         [" Barbarossa", " Overlord", " Sealion", " Citadel"], 0),
        ("The atomic bomb dropped on Hiroshima in 1945 was nicknamed",
         [" Little Boy", " Fat Man", " Trinity", " Castle Bravo"], 0),
        ("The 1941 Japanese attack on the US fleet in Hawaii struck Pearl",
         [" Harbor", " Bay", " Cove", " Point"], 0),
        ("The famous 1963 civil-rights speech by Martin Luther King is remembered as I Have a",
         [" Dream", " Vision", " Promise", " Calling"], 0),
        ("The first man to walk on the Moon, in 1969, was Neil",
         [" Armstrong", " Aldrin", " Glenn", " Borman"], 0),
        ("The Berlin barrier that was torn down in 1989 was the Berlin",
         [" Wall", " Gate", " Bridge", " Arch"], 0),
    ],
}

GEN = [
    ("PRE-1929 ", "In what city was Archduke Franz Ferdinand assassinated in 1914?"),
    ("PRE-1929 ", "Which peace treaty ended the First World War?"),
    ("POST-1929", "What was the codename for the German invasion of the Soviet Union?"),
    ("POST-1929", "What was the nickname of the first atomic bomb dropped on Hiroshima?"),
    ("POST-1929", "Who delivered the speech that begins 'I have a dream'?"),
]


def seq_logprob(model, tok, stem, option):
    sid = tok(stem, add_special_tokens=False).input_ids
    oid = tok(option, add_special_tokens=False).input_ids
    ids = torch.tensor([sid + oid])
    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[0], dim=-1)
    tot = 0.0
    for k in range(len(oid)):
        tot += lp[len(sid) + k - 1, (sid + oid)[len(sid) + k]].item()
    return tot, len(oid)


def main():
    print(f"loading {MODEL} on CPU (float32) -- a few minutes ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    print("loaded.\n", flush=True)

    print("################  PART A : multiple-choice probability  ################")
    for tier, items in MC.items():
        print("\n" + "=" * 70 + f"\n{tier}\n" + "=" * 70)
        hits, correct_norms = 0, []
        for stem, opts, ci in items:
            avg = []
            for o in opts:
                s, n = seq_logprob(model, tok, stem, o)
                avg.append(s / n)
            m = max(avg); ex = [math.exp(a - m) for a in avg]; Z = sum(ex)
            norm = [e / Z for e in ex]
            win = max(range(len(opts)), key=lambda i: avg[i])
            hits += (win == ci); correct_norms.append(norm[ci])
            print(f"\nQ: {stem} ___")
            for i, o in enumerate(opts):
                print(f"   {o.strip():<12} {norm[i]*100:5.1f}%" +
                      ("  <-correct" if i == ci else "") + ("  WIN" if i == win else ""))
        print(f"\n  >> picked correct {hits}/{len(items)};  "
              f"mean prob on correct = {sum(correct_norms)/len(correct_norms)*100:.1f}%  (chance 25%)")

    print("\n\n################  PART B : free-text answers  ################")
    for tag, q in GEN:
        msgs = [{"role": "user", "content": q}]
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=40, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()
        print(f"\n[{tag}] Q: {q}\n          A: {ans[:200]}")


if __name__ == "__main__":
    main()
