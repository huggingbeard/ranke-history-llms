"""
For Daniel: what does the 'rank' probe actually compute, and how expensive is it?
Claim under test: "it's len(vocab)^len(tokens) = impossible, an infeasible beam search."
"""
import os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
torch.set_num_threads(os.cpu_count() or 8)

M = "uzh-echist-org/Ranke-4B-1929"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32, low_cpu_mem_usage=True).eval()
V = model.config.vocab_size

prompt = "The President who ordered the atomic bomb dropped on Japan in 1945 was named Harry"
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids

t0 = time.perf_counter()
with torch.no_grad():
    logits = model(ids).logits[0, -1]          # <-- ONE forward pass; gives a score for EVERY vocab token at the next slot
t1 = time.perf_counter()
probs = torch.softmax(logits, dim=-1)
tgt_ids = tok(" Truman", add_special_tokens=False).input_ids
tgt = tgt_ids[0]
t2 = time.perf_counter()
rank = int((probs > probs[tgt]).sum().item()) + 1   # how many of the V tokens beat 'Truman' -> a single O(V) comparison
t3 = time.perf_counter()

print("=" * 60)
print(f"vocab size  V              = {V:,}")
print(f"'the impossible version'   = V^3 (3-token seq space) = {V**3:.3e}")
print("-" * 60)
print("WHAT WE ACTUALLY DO:")
print(f"  forward passes           = 1        (took {t1-t0:.2f} s)")
print(f"  rank = count(probs > p_target) over V tokens   (took {t3-t2*1:.5f} s)")
print("-" * 60)
print(f"  ' Truman' tokenizes to   = {tgt_ids}  ({len(tgt_ids)} token)")
print(f"  P('Truman' | prompt)     = {float(probs[tgt])*100:.1f}%")
print(f"  rank of 'Truman'         = {rank} of {V:,}")
print("=" * 60)
print("We never enumerate sequences. We read the model's already-computed")
print("next-token distribution and look up ONE entry. Cost: 1 forward pass.")
