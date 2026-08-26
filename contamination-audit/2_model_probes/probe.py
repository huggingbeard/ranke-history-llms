"""
Multiple-choice probability probe for a (possibly) time-locked LLM.

Goal: see how a model's probability lines up on three tiers of facts:
  TIER 0  impossible/fiction   -> nobody can know  -> expect ~chance (25%)
  TIER 1  pre-1913 facts       -> a clean 1913 model SHOULD know -> expect high
  TIER 2  post-1913 facts      -> a clean 1913 model should NOT know -> expect ~chance
                                  (if it scores high here = CONTAMINATION)

For each item we score every option as a continuation of the stem, using the
model's own probabilities. We report:
  - raw   = the model's raw probability of the option string (tiny, like the RA's 3%)
  - norm  = probability AMONG the 4 options only (renormalized) -> compare this to 25%

Usage:
    python probe.py                       # stock Qwen3-4B (reference / "knows everything")
    python probe.py /path/to/your/model   # YOUR time-locked model on server 2

Same code + same items for both models => an apples-to-apples comparison.
"""
import sys, os, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
torch.set_num_threads(os.cpu_count() or 8)

# ---------------------------------------------------------------- items
# Each item: (stem, [options], index_of_correct_answer)
# Distractors are deliberately from the SAME era as the answer, so a 1913
# model cannot cheat by "this sounds modern". The answers in Tier 2 are
# arbitrary proper nouns that CANNOT be guessed from pre-1913 knowledge.
ITEMS = {
    "TIER 0  impossible / fiction (expect ~25%)": [
        ("In Tolkien-style legend the wizard sealed the dark gate of the realm called",
         [" Marrowind", " Velkaris", " Korrath", " Sunmere"], 0),
        ("The secret password to the smugglers' cellar was the single word",
         [" Halcyon", " Ferrobane", " Quillon", " Nadiret"], 0),
    ],
    "TIER 1  pre-1913 (a clean 1913 model SHOULD know -> expect high on correct)": [
        ("The ocean liner that struck an iceberg and sank in 1912 was the",
         [" Titanic", " Lusitania", " Mauretania", " Britannic"], 0),
        ("The 1789 storming of the fortress-prison that began the French Revolution was the",
         [" Bastille", " Tuileries", " Conciergerie", " Louvre"], 0),
        ("On the Origin of Species was written by Charles",
         [" Darwin", " Dickens", " Lyell", " Wallace"], 0),
        ("Napoleon was finally defeated in 1815 at the Battle of",
         [" Waterloo", " Austerlitz", " Borodino", " Leipzig"], 0),
    ],
    "TIER 2  post-1913 (a clean 1913 model should NOT know -> high here = contamination)": [
        ("The 1941 German invasion of the Soviet Union was code-named Operation",
         [" Barbarossa", " Overlord", " Sealion", " Torch"], 0),
        ("The atomic bomb dropped on Hiroshima in 1945 was nicknamed",
         [" Little Boy", " Fat Man", " Trinity", " Castle Bravo"], 0),
        ("The famous 1963 speech by Martin Luther King is remembered as I Have a",
         [" Dream", " Vision", " Promise", " Calling"], 0),
        ("The first man to walk on the Moon, in 1969, was Neil",
         [" Armstrong", " Aldrin", " Glenn", " Borman"], 0),
        ("The Berlin structure that was torn down in 1989 was the Berlin",
         [" Wall", " Gate", " Bridge", " Tower"], 0),
    ],
}

# ---------------------------------------------------------------- scoring
def seq_logprob(model, tok, stem, option):
    """log P(option | stem), summed over the option's tokens. Also returns #tokens."""
    stem_ids = tok(stem, add_special_tokens=False).input_ids
    opt_ids = tok(option, add_special_tokens=False).input_ids
    ids = torch.tensor([stem_ids + opt_ids])
    with torch.no_grad():
        logits = model(ids).logits[0]               # (seq, vocab)
    logprobs = torch.log_softmax(logits, dim=-1)
    total, n = 0.0, len(opt_ids)
    for k in range(n):
        pos = len(stem_ids) + k - 1                 # logits at pos predict token pos+1
        tok_id = (stem_ids + opt_ids)[len(stem_ids) + k]
        total += logprobs[pos, tok_id].item()
    return total, n

def main():
    print(f"loading {MODEL} on CPU (float32) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    print("loaded.\n", flush=True)

    for tier, items in ITEMS.items():
        print("=" * 78)
        print(tier)
        print("=" * 78)
        correct_norms, hits = [], 0
        for stem, options, ci in items:
            sums, lens = [], []
            for opt in options:
                s, n = seq_logprob(model, tok, stem, opt)
                sums.append(s); lens.append(n)
            # length-normalised log-prob per option (fair across option lengths)
            avg = [s / n for s, n in zip(sums, lens)]
            # normalise OVER THE OPTIONS -> a probability you compare to 25%
            m = max(avg)
            exps = [math.exp(a - m) for a in avg]
            Z = sum(exps)
            norm = [e / Z for e in exps]
            # raw whole-string probability (tiny, like the RA's "3%")
            raw = [math.exp(s) for s in sums]
            win = max(range(len(options)), key=lambda i: avg[i])
            if win == ci: hits += 1
            correct_norms.append(norm[ci])
            print(f"\nQ: {stem} ___")
            for i, opt in enumerate(options):
                flag = "<-correct" if i == ci else ""
                star = " WIN" if i == win else ""
                print(f"   {opt.strip():<14} norm={norm[i]*100:5.1f}%  raw={raw[i]*100:7.3f}%{star} {flag}")
        acc = hits / len(items)
        mean_correct = sum(correct_norms) / len(correct_norms)
        print(f"\n  >> {tier.split('(')[0].strip()}: picked correct {hits}/{len(items)} "
              f"(={acc*100:.0f}%);  mean norm-prob on correct = {mean_correct*100:.1f}%  "
              f"(chance = 25%)\n")

if __name__ == "__main__":
    main()
