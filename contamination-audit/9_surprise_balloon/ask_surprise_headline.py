"""
Ask Ranke-4B-1929 (Instruct) to self-report surprise (0-9) at a modern headline.
A playful, conversational probe (NOT the teacher-forced NLL metric): we simply ASK
the 1929-time-locked chat model how surprised it is.

Model: uzh-echist-org/Ranke-4B-1929-1202-Instruct  (Qwen3-4B arch, chat-tuned)
Runs on the 4090 (bf16, ~8 GB). Standalone -> frees VRAM on exit.
Version 0.1 (2026-07-10).

Usage: python ask_surprise_headline.py
"""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MID = "uzh-echist-org/Ranke-4B-1929-1202-Instruct"  # 1929 chat model
DEV = "cuda"
DT = torch.bfloat16
HEADLINE = "Atomkraft oder Solarstrom: Wer braucht mehr Subventionen?"

USER = (
    "Bewerten Sie auf einer Skala von 0 bis 9, wie ueberrascht Sie die folgende "
    "Schlagzeile macht. 0 = ueberhaupt nicht ueberrascht (voellig alltaeglich und "
    "verstaendlich); 9 = maximal ueberrascht (voellig unverstaendlich, unvorstellbar). "
    "Antworten Sie zuerst mit der Zahl allein, dann einen kurzen Satz zur Begruendung.\n\n"
    f"Schlagzeile: »{HEADLINE}«"
)

print(f"Loading {MID} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
print(f"Loaded. VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB\n", flush=True)

msgs = [{"role": "user", "content": USER}]
try:
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
except TypeError:
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

print("----- rendered chat prompt -----")
print(text)
print("--------------------------------\n", flush=True)

inputs = tok(text, return_tensors="pt").to(DEV)
with torch.no_grad():
    gen = model.generate(**inputs, max_new_tokens=220, do_sample=False,
                         repetition_penalty=1.3, no_repeat_ngram_size=4,
                         pad_token_id=tok.eos_token_id)
out = tok.decode(gen[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("=" * 60)
print(f"HEADLINE: {HEADLINE}")
print("=" * 60)
print("RANKE-1929 ANSWER:")
print(out.strip())
print("=" * 60)
