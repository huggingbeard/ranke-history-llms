# -*- coding: utf-8 -*-
"""DIAGNOSIS for the failed batching gate: WHERE does the batched-vs-sequential drift come from?

Three probes on the same real battery slice (6 items, 240 scores):
  A. bf16, batch_size=1  vs sequential slp  — isolates the attention-mask/kernel-path change with NO
     padding and NO cross-row batching. If A is exact, the slp_pairs indexing/padding logic is correct.
  B. bf16, batch_size=32 vs sequential slp  — the failed gate config, re-measured on this slice.
  C. fp32 model, batch_size=32 vs fp32 sequential — if fp32 agrees to ~1e-4, the bf16 drift is pure
     low-precision kernel accumulation noise (neither bf16 path is 'the truth'), not a bug.
Also times fp32-batched against bf16-sequential (is exact-and-batched still faster than today?).
Run:  python src/test_slp_batch2.py
"""
import time, math
import torch
import common
import v41_battery as vb
import control_battery as cb
from foresight_bank import RANKE
from transformers import AutoModelForCausalLM

MID = RANKE["1913"]
items = [(q["stems"], q["neutral"], q["options"]) for q in vb.Q[:3]] + \
        [(q["stems"], q["neutral"], q["options"]) for q in cb.C[:3]]


def run_seq(model, tok):
    out = []
    for stems, neutral, options in items:
        for stem in stems:
            for o in options:
                out.append(common.slp(model, tok, stem, o))
            for o in options:
                out.append(common.slp(model, tok, neutral, o))
    return out


def run_bat(model, tok, bs):
    out = []
    for stems, neutral, options in items:
        for stem in stems:
            out.extend(common.slp_batch(model, tok, stem, options, batch_size=bs))
            out.extend(common.slp_batch(model, tok, neutral, options, batch_size=bs))
    return out


def report(tag, a, b):
    d = [abs(x - y) for x, y in zip(a, b)]
    print(f"  {tag}: max |diff| {max(d):.6f}   mean {sum(d)/len(d):.7f}")
    return max(d)


print(f"slice: {len(items)} items, {sum(2*len(s)*len(o) for s,_,o in items)} scores; model {MID}")

# ---------------- bf16 ----------------
tok, model = common.load(MID)
t0 = time.time(); seq16 = run_seq(model, tok); t_seq16 = time.time() - t0
bat16_1 = run_bat(model, tok, 1)
t0 = time.time(); bat16_32 = run_bat(model, tok, 32); t_bat16 = time.time() - t0
print("\nbf16:")
report("A. batch=1  vs sequential", seq16, bat16_1)
report("B. batch=32 vs sequential", seq16, bat16_32)
del model
common.free()

# ---------------- fp32 ----------------
common.wait_for_vram(min_gb=17.0)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32, low_cpu_mem_usage=True).to(common.DEV).eval()
t0 = time.time(); seq32 = run_seq(model, tok); t_seq32 = time.time() - t0
t0 = time.time(); bat32 = run_bat(model, tok, 32); t_bat32 = time.time() - t0
print("\nfp32:")
mx = report("C. batch=32 vs sequential", seq32, bat32)
report("   fp32-seq vs bf16-seq (precision gap itself)", seq32, seq16)
del model
common.free()

print(f"\ntiming on this slice: bf16-seq {t_seq16:.1f}s | bf16-batch {t_bat16:.1f}s | "
      f"fp32-seq {t_seq32:.1f}s | fp32-batch {t_bat32:.1f}s")
print(f"fp32-batched vs today's bf16-sequential: x{t_seq16/max(t_bat32,1e-9):.1f}")
print(f"\nverdict: fp32 batch-invariant = {'YES (drift is bf16 kernel noise, not a bug)' if mx < 1e-3 else 'NO (investigate slp_pairs)'}")
