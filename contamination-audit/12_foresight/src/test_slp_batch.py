# -*- coding: utf-8 -*-
"""REGRESSION GATE for the STANDARD SCORING CONFIG: fp32 model + batched slp (common.slp_batch).

HISTORY. The first gate run (2026-07-11, bf16) FAILED: bf16 kernels give batch-shape-dependent numbers
(max 0.23 nats / 0.038 probability drift vs sequential; zero modal flips on 640 scores). The diagnosis
(src/test_slp_batch2.py) proved the batching CODE exact (batch=1 == sequential to 0.000000; fp32
batch-invariant to 3e-5) — the drift is bf16 kernel-accumulation noise, present in the historical
bf16-sequential numbers too (bf16 vs fp32 differs by up to 0.10 nats). Standard config since then:
fp32 + batched — exact, replicable, and ~x4.5 faster than the old bf16-sequential path.

THIS TEST gates that config: on real battery items it checks
  1. raw log-prob agreement between batched and sequential fp32 scoring;
  2. decision-level agreement: per-option probabilities within 1e-3 and ZERO modal flips;
  3. speed.
If it fails, do not trust batched runs until diagnosed. Re-run after any change to slp/slp_pairs, a
torch upgrade, or a GPU change.
Run:  python src/test_slp_batch.py
"""
import time, math, sys
import common
import v41_battery as vb          # guarded engine: Q importable without loading models
import control_battery as cb      # C importable; run() guarded
from foresight_bank import RANKE

MID = RANKE["1913"]

# real work: 8 v41 items (EN) + 8 control items = 16 items x 5 stems x 4 options x 2 (raw+neutral)
items = []
for q in vb.Q[:8]:
    items.append((q["stems"], q["neutral"], q["options"]))
for q in cb.C[:8]:
    items.append((q["stems"], q["neutral"], q["options"]))
n_calls = sum(2 * len(s) * len(o) for s, _, o in [(i[0], i[1], i[2]) for i in items])
print(f"test set: {len(items)} items, {n_calls} slp calls each way; model {MID}")

tok, model = common.load(MID)

# ---------------- sequential (the historical path) ----------------
t0 = time.time()
seq = []
for stems, neutral, options in items:
    for stem in stems:
        for o in options:
            seq.append(common.slp(model, tok, stem, o))
        for o in options:
            seq.append(common.slp(model, tok, neutral, o))
t_seq = time.time() - t0

# ---------------- batched (the new default) ----------------
t0 = time.time()
bat = []
for stems, neutral, options in items:
    for stem in stems:
        bat.extend(common.slp_batch(model, tok, stem, options))
        bat.extend(common.slp_batch(model, tok, neutral, options))
t_bat = time.time() - t0

del tok, model
common.free()

# ---------------- 1. raw agreement ----------------
diffs = [abs(a - b) for a, b in zip(seq, bat)]
mx = max(diffs)
mean = sum(diffs) / len(diffs)
print(f"\nraw log-prob agreement over {len(diffs)} scores: max |diff| {mx:.5f}  mean {mean:.6f}")

# ---------------- 2. decision-level agreement (the gate) ----------------
def softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    Z = sum(ex)
    return [e / Z for e in ex]

flips = 0
max_pdiff = 0.0
i = 0
for stems, neutral, options in items:
    for _ in stems:
        for block in range(2):                      # raw block, then neutral block
            s4, b4 = seq[i:i + 4], bat[i:i + 4]
            ps, pb = softmax(s4), softmax(b4)
            max_pdiff = max(max_pdiff, max(abs(x - y) for x, y in zip(ps, pb)))
            if max(range(4), key=lambda k: ps[k]) != max(range(4), key=lambda k: pb[k]):
                flips += 1
            i += 4
print(f"decision-level: max per-option probability diff {max_pdiff:.6f}   modal flips {flips}")

# ---------------- 3. speed ----------------
print(f"\nspeed: sequential {t_seq:.1f}s   batched {t_bat:.1f}s   speedup x{t_seq / max(t_bat, 1e-9):.1f}")

ok = (max_pdiff < 1e-3) and (flips == 0)
print(f"\nGATE (prob diff < 1e-3 AND zero modal flips): {'PASS — batching is safe as the default' if ok else 'FAIL — do NOT adopt batching'}")
sys.exit(0 if ok else 1)
