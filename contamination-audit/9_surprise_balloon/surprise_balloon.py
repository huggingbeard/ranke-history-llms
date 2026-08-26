# -*- coding: utf-8 -*-
"""TRIAL BALLOON: measure the 'surprise shift' of events across time-locked worldviews (Ranke-1900 vs -1929).

Idea (PMI decomposition). For a target predicate w in context c:
    -log P(w|c) = -log P(w) + [-PMI(w;c)]   where PMI = log P(w|c) - log P(w)
The first term is the LEXICAL channel (how rare the words are). The second is the CONCEPTUAL channel
(how much the specific configuration strains the model's frame). We control the lexical channel by using
ONLY 1900-vocabulary words, then read the conceptual strain and how it SHIFTS from the 1900 to the 1929 model.

Per model, per item:
    S_marg = -log P(target | generic subject)   averaged over neutral subjects   (lexical / base rate)
    S_cond = -log P(target | the real loaded subject)
    strain = S_cond - S_marg = -PMI             (positive = configuration strains the worldview)
Cross-model:
    d_strain = strain_1900 - strain_1929        (positive = 1929 finds it LESS surprising = 'learned' it)
    d_marg   = S_marg_1900 - S_marg_1929         (should be ~0: lexical channel is controlled)
    d_cond   = S_cond_1900 - S_cond_1929 = d_marg + d_strain (sanity identity)

Output: surprise_balloon.csv / .json  (+ console table). All words verified in-1900-vocabulary by design.
"""
import os, sys, json, csv, statistics as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32

MODELS = {"1900": "uzh-echist-org/Ranke-4B-1900", "1929": "uzh-echist-org/Ranke-4B-1929"}
GENERICS = ["The man", "A person", "The thing", "It"]   # neutral subjects -> marginal base rate of the predicate

# id, category, real subject, target predicate (the event), human note
ITEMS = [
    ("women_office", "A: became real 1900-29", "The woman", " was elected to high office.", "women's suffrage/office (1920s)"),
    ("flight_ocean", "A: became real 1900-29", "The two men", " flew across the great ocean.", "transatlantic flight 1919/1927"),
    ("russia_fell",  "A: became real 1900-29", "The empire of Russia", " came to a sudden and bloody end.", "Romanov fall 1917-18"),
    ("talkies",      "A: became real 1900-29", "The moving pictures", " began to talk and to sing.", "sound film 1927"),
    ("great_war",    "A: became real 1900-29", "The war in Europe", " killed many millions of young men.", "WWI scale 1914-18"),
    ("austria_split","A: became real 1900-29", "The empire of Austria", " was broken into many small pieces.", "Austria-Hungary dissolved 1918"),

    ("sun_rose",     "B: timeless truth",      "The bright sun", " rose above the eastern hills.", "always true"),
    ("king_throne",  "B: timeless truth",      "The old king", " sat upon his golden throne.", "always true"),
    ("snow_winter",  "B: timeless truth",      "The cold snow", " fell through the long winter night.", "always true"),

    ("moon_walk",    "C: future to BOTH",      "The brave men", " walked upon the face of the moon.", "1969 - future to both"),
    ("atom_bomb",    "C: future to BOTH",      "The great city", " was destroyed by a single bomb.", "1945 - future to both"),
    ("heart_xplant", "C: future to BOTH",      "The clever doctors", " gave a man the heart of another.", "1967 - future to both"),
]


def slp(model, tok, prompt, cont):
    """sum log P(cont | prompt), no BOS (matches project scoring)."""
    pid = tok(prompt, add_special_tokens=False).input_ids
    cid = tok(cont, add_special_tokens=False).input_ids
    full = pid + cid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(cid)))


scores = {tag: {} for tag in MODELS}
for tag, mid in MODELS.items():
    print(f"\n=== loading {mid} ({tag}) on {DEV} ===")
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    for iid, cat, subj, tgt, note in ITEMS:
        s_cond = -slp(model, tok, subj, tgt)
        s_marg = st.mean(-slp(model, tok, g, tgt) for g in GENERICS)
        scores[tag][iid] = {"S_cond": s_cond, "S_marg": s_marg, "strain": s_cond - s_marg}
        print(f"  {iid:14s} S_cond={s_cond:7.2f}  S_marg={s_marg:7.2f}  strain={s_cond - s_marg:+7.2f}")
    del model
    import gc; gc.collect()
    if DEV == "cuda": torch.cuda.empty_cache()

# ---- combine + cross-model shift ----
rows = []
for iid, cat, subj, tgt, note in ITEMS:
    a, b = scores["1900"][iid], scores["1929"][iid]
    rows.append({
        "id": iid, "category": cat, "subject": subj, "target": tgt.strip(), "note": note,
        "S_cond_1900": round(a["S_cond"], 2), "S_cond_1929": round(b["S_cond"], 2),
        "S_marg_1900": round(a["S_marg"], 2), "S_marg_1929": round(b["S_marg"], 2),
        "strain_1900": round(a["strain"], 2), "strain_1929": round(b["strain"], 2),
        "d_strain": round(a["strain"] - b["strain"], 2),          # +ve => 1929 less surprised (learned)
        "d_marg": round(a["S_marg"] - b["S_marg"], 2),            # lexical drift (want ~0)
        "d_cond": round(a["S_cond"] - b["S_cond"], 2),
    })

json.dump({"models": MODELS, "generics": GENERICS, "rows": rows},
          open(os.path.join(HERE, "surprise_balloon.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(os.path.join(HERE, "surprise_balloon.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---- console report ----
print("\n" + "=" * 100)
print("SURPRISE SHIFT  (strain = -PMI = how much the configuration strains the worldview)")
print(f"{'item':14s} {'category':22s} {'strain_1900':>11s} {'strain_1929':>11s} {'Δstrain':>8s} {'Δlex':>6s}")
print("-" * 100)
for cat in ["A: became real 1900-29", "B: timeless truth", "C: future to BOTH"]:
    sub = [r for r in rows if r["category"] == cat]
    for r in sorted(sub, key=lambda r: -r["d_strain"]):
        print(f"{r['id']:14s} {cat:22s} {r['strain_1900']:>11.2f} {r['strain_1929']:>11.2f} {r['d_strain']:>+8.2f} {r['d_marg']:>+6.2f}")
    md = st.mean(r["d_strain"] for r in sub); ms = st.mean(r["d_marg"] for r in sub)
    print(f"{'  >> MEAN':14s} {cat:22s} {'':>11s} {'':>11s} {md:>+8.2f} {ms:>+6.2f}")
    print("-" * 100)
print("Δstrain > 0  => the 1929 model finds the event LESS surprising than the 1900 model (worldview shift).")
print("Δlex (Δmarg) ~ 0  => lexical channel is controlled (same 1900 words), so the shift is conceptual.")
print("wrote surprise_balloon.csv / .json")
