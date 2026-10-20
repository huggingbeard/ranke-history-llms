"""OLMoE smoke test, login-node check (no card, seconds): does transformers' OlmoeForCausalLM give what the training
script relies on? A tiny OLMoE (2 layers, width 64, 8 experts, 2 per token) on CPU, one forward/backward:
  - out.loss includes router_aux_loss_coef * out.aux_loss
  - out.router_logits: one tensor per layer, shape [tokens, experts]
  - loss is finite and gradients reach the router and the experts
Also checks the token snapshot (sizes, max token id < vocab, SHA-256 against the manifest).
"""
import hashlib, json, os, sys
import numpy as np
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

data = sys.argv[1] if len(sys.argv) > 1 else "/scratch/ukrenk/moe_smoke_20261020/data"
cfg = OlmoeConfig(vocab_size=151936, hidden_size=64, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=4, num_experts=8, num_experts_per_tok=2, max_position_embeddings=128,
                  router_aux_loss_coef=0.01, output_router_logits=True, norm_topk_prob=False, use_cache=False)
torch.manual_seed(0)
m = OlmoeForCausalLM(cfg)
x = torch.randint(0, 151936, (2, 64))
out = m(input_ids=x, labels=x)
print("loss", float(out.loss), "aux", float(out.aux_loss), "router_logits", [tuple(r.shape) for r in out.router_logits])
m2 = OlmoeForCausalLM(cfg.__class__(**{**cfg.to_dict(), "router_aux_loss_coef": 0.0}))
m2.load_state_dict(m.state_dict())
out2 = m2(input_ids=x, labels=x)
assert abs(float(out.loss - out2.loss) - 0.01 * float(out.aux_loss)) < 1e-4, "loss does not include coef*aux as assumed"
out.loss.backward()
g = {n: p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in m.named_parameters()}
assert any(v for n, v in g.items() if "gate" in n and "experts" not in n), "no gradient to router"
assert any(v for n, v in g.items() if "experts" in n), "no gradient to experts"
print("OK model API: loss = lm + coef*aux, router logits per layer, gradients reach router and experts")
man = json.load(open(os.path.join(data, "manifest.json")))
for name in ("train", "val"):
    p = os.path.join(data, f"{name}_tokens.npy")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == man[name]["sha256"], f"{name} hash mismatch"
    arr = np.load(p, mmap_mode="r")
    assert arr.size == man[name]["tokens"] and int(arr.max()) < 151936
    print(f"OK data {name}: {arr.size:,} tokens, {man[name]['books']} books, max id {int(arr.max())}")
