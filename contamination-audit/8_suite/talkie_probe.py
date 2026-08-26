# -*- coding: utf-8 -*-
"""RUNPOD runner: score the shared question_bank.json on Talkie-1930-13B (base) with whole-answer sum-logp
forced choice -- the IDENTICAL metric used on Ranke -- and write results_talkie-1930-13b-base.json so the two
are directly comparable.

WHY THIS FILE EXISTS / THE ONE SUBTLETY: Talkie's TalkieModel.forward() returns ONLY the last position's
logits (it does `x[:, -1, :]` before the lm_head). Teacher-forcing needs the logit at every continuation
position, so we reimplement the forward WITHOUT the last-position slice -- byte-identical math otherwise
(embed -> rms_norm -> 40 blocks(e_x,x,cos_sin) -> rms_norm -> lm_head_gain*lm_head). We then ASSERT our
version's last row equals their official forward() before scoring anything; if that assert passes, the
per-token logprobs are provably the model's own.

SETUP on a runpod box with a >=28GB GPU (A40/A6000 48GB is plenty; bf16 ~26GB):
    git clone https://github.com/talkie-lm/talkie.git && cd talkie && pip install -e .   # pulls torch+tiktoken
    pip install huggingface_hub
    python talkie_probe.py            # downloads talkie-1930-13b-base (~53GB) on first run, then scores
Then scp results_talkie-1930-13b-base.json back and TEAR DOWN THE POD.
"""
import os, json, math, statistics as st
import torch
import torch.nn.functional as F
from talkie import Talkie

HERE = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(HERE, "question_bank.json")
MODEL = "talkie-1930-13b-base"          # base = the contamination test (matches Guohui's run)
OUT = os.path.join(HERE, "results_%s.json" % MODEL)

print("loading", MODEL, "...", flush=True)
T = Talkie(MODEL)                        # downloads on first use; device auto = cuda
m, tok, DEV = T.model, T.tokenizer, T.device
AC = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if DEV.type == "cuda" else torch.no_grad()


def full_logits(ids):
    """Their forward(), but returns logits at ALL positions [B,T,V] (no last-position slice)."""
    _, T_ = ids.shape
    cos_sin = (m.cos[:, :T_], m.sin[:, :T_])
    x = m.embed(ids)
    x = F.rms_norm(x, (x.shape[-1],))
    e_x = x
    for block in m.blocks:
        x = block(e_x, x, cos_sin)
    x = F.rms_norm(x, (x.shape[-1],))
    return F.linear(x, m.lm_head_gain(m.lm_head)).float()


# ---- VALIDATE: our all-position logits must match their official last-position forward ----
with torch.no_grad(), AC:
    _ids = torch.tensor([tok.encode("In the year the great war began", allowed_special="all")],
                        dtype=torch.long, device=DEV)
    mine = full_logits(_ids)[:, -1, :]
    theirs = m(_ids)
    maxdiff = (mine.float() - theirs.float()).abs().max().item()
assert maxdiff < 1e-2, f"full_logits does NOT match official forward (maxdiff={maxdiff}); ABORT"
print(f"validation OK: full_logits matches forward() at last position (maxdiff={maxdiff:.2e})", flush=True)


def slp(prompt, cont):
    pid = tok.encode(prompt, allowed_special="all")
    cid = tok.encode(cont, allowed_special="all")
    full = pid + cid
    ids = torch.tensor([full], dtype=torch.long, device=DEV)
    with torch.no_grad(), AC:
        lp = torch.log_softmax(full_logits(ids)[0], dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(cid)))


def binom_ge(k, n, p=0.25):
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


bank = json.load(open(BANK, encoding="utf-8"))["items"]
rows = []
for i, it in enumerate(bank):
    t = slp(it["prompt"], it["true"]); tw = [slp(it["prompt"], d) for d in it["distractors"]]
    gb = t - max(tw)
    rows.append({**{k: it[k] for k in ("era", "topic", "label")}, "gap_best": round(gb, 3),
                 "gap_mean": round(t - st.mean(tw), 3), "correct": int(gb > 0)})
    if (i + 1) % 25 == 0:
        print(f"  scored {i+1}/{len(bank)}", flush=True)

agg = {}
for era in ("pre", "post"):
    for topic in sorted({r["topic"] for r in rows if r["era"] == era}) + ["ALL"]:
        sub = [r for r in rows if r["era"] == era and (topic == "ALL" or r["topic"] == topic)]
        if not sub:
            continue
        k, n = sum(r["correct"] for r in sub), len(sub)
        agg[f"{era}:{topic}"] = {"correct": k, "n": n, "pct": round(100 * k / n, 1),
                                 "chance_pct": 25.0, "binom_p": round(binom_ge(k, n), 4),
                                 "mean_gap": round(st.mean(r["gap_best"] for r in sub), 3)}
json.dump({"model": MODEL, "metric": "forced_choice_sumlogp", "aggregate": agg, "items": rows},
          open(OUT, "w", encoding="utf-8"), indent=1)
print("\n==== %s ====" % MODEL)
for era in ("pre", "post"):
    a = agg.get(f"{era}:ALL")
    if a:
        print(f"  {era.upper():<4} ALL  {a['correct']}/{a['n']}  {a['pct']}%  (chance 25%)  p={a['binom_p']}  gap={a['mean_gap']:+.2f}")
print("wrote", OUT)
