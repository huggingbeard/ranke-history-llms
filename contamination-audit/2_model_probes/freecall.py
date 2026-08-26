"""
Free-recall probe: does the model PRODUCE post-1929 facts when we do NOT feed the
answer in the question? This tests Voth's point directly -- a clean 1929 model
should not be able to name JFK as president, the Second World War, etc.

Usage: python freecall.py [model_id]   (default = Ranke-4B-1929-1202-Instruct)
"""
import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
MODEL = sys.argv[1] if len(sys.argv) > 1 else "uzh-echist-org/Ranke-4B-1929-1202-Instruct"
torch.set_num_threads(os.cpu_count() or 8)

QUESTIONS = [
    "Who was the President of the United States in the year 1962?",
    "List several Presidents of the United States.",
    "Name the major wars that took place in the twentieth century.",
    "What important new inventions appeared between 1930 and 1970?",
    "Name some famous political leaders from the 1950s and 1960s.",
    "Briefly, what were the most important events of the 1940s?",
]

# Unambiguously POST-1929 -- people/terms that did NOT exist as public concepts by 1929.
# (Deliberately EXCLUDES Hitler, Stalin, Mussolini, Churchill, Lenin, Trotsky, Gandhi --
#  all already public by 1929, so they are NOT proof of leakage.)
LEAK = ["kennedy", "truman", "eisenhower", "nixon", "lyndon johnson", "de gaulle",
        "khrushchev", "brezhnev", "adenauer", "mao", "ho chi minh",
        "second world war", "world war ii", "world war two", "world war 2", "wwii",
        "cold war", "korean war", " korea", "vietnam", "cuban missile",
        "atomic bomb", "atom bomb", "nuclear", "hydrogen bomb", "hiroshima", "nagasaki",
        "united nations", "nato", "moon landing", "apollo", "berlin wall",
        "franklin roosevelt", "fdr", "television", "computer", "jet engine"]


def main():
    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    print("loaded.\n", flush=True)

    found_all = set()
    for q in QUESTIONS:
        msgs = [{"role": "user", "content": q}]
        if getattr(tok, "chat_template", None):
            try:
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = q.strip() + "\nAnswer:"   # base model has no chat template -> raw continuation
        enc = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(enc.input_ids, attention_mask=enc.attention_mask,
                                 max_new_tokens=70, do_sample=False, pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()
        hits = sorted({m for m in LEAK if m in ans.lower()})
        found_all |= set(hits)
        print(f"Q: {q}")
        print(f"A: {ans[:300]}")
        print(f"   >> post-1929 leak terms produced: {hits if hits else 'none'}\n")

    print("=" * 70)
    print(f"VERDICT: {len(found_all)} distinct unambiguous post-1929 concepts produced UNPROMPTED: "
          f"{sorted(found_all)}")
    print("(a truly clean 1929 model should produce ZERO of these)")


if __name__ == "__main__":
    main()
