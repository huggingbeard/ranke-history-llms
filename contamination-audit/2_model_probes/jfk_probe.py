"""
Does the BASE model latently know specific post-1929 facts? Forced next-token
completions (NOT free generation, which a base model can't do). We print the raw
top-8 next tokens so we can SEE whether Kennedy / Truman / 'War'(WWII) / 1939 surface.

Runs base AND instruct, raw continuation, apples-to-apples.
"""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
torch.set_num_threads(os.cpu_count() or 8)

MODELS = ["uzh-echist-org/Ranke-4B-1929", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
PROBES = [
    ("JFK, NO NAME given (strongest)", "The President of the United States who was shot and killed in Dallas in 1963 was named"),
    ("JFK role",                       "John F. Kennedy served his country for three years as the President of the United"),
    ("JFK surname",                    "The young American President assassinated in 1963 was named John F."),
    ("CONTROL Lincoln (pre-1929)",     "The American President Abraham Lincoln was shot and killed in the city of"),
    ("Truman + atomic bomb",           "The President who ordered the atomic bomb dropped on Japan in 1945 was named Harry"),
    ("WWII exists?",                   "The First World War was followed, twenty years later, by the Second World"),
    ("WWII year",                      "The Second World War broke out across Europe in the year"),
    ("CONTROL Barbarossa (gen=no)",    "The 1941 German invasion of the Soviet Union was code-named Operation"),
    ("Moon (gen=no)",                  "The first man to set foot upon the Moon was an American named Neil"),
]


def topk(model, tok, prompt, k=8):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    probs = torch.softmax(logits, -1)
    tv, ti = probs.topk(k)
    return [(tok.decode([int(i)]).strip(), float(p)) for p, i in zip(tv, ti)]


for M in MODELS:
    print("\n" + "=" * 78 + f"\nMODEL: {M}\n" + "=" * 78, flush=True)
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    for label, prompt in PROBES:
        tops = topk(model, tok, prompt)
        shown = ", ".join(f"{t!r}:{p*100:.0f}%" for t, p in tops)
        print(f"\n[{label}]\n  \"...{prompt[-60:]}\" ->\n  {shown}", flush=True)
    del model
