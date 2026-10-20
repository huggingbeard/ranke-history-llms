"""Does the expert-balance penalty reach the router weights? (2026-10-20; jobs 7377866 and 7378053 collapsed onto 8 experts
per layer with both the pooled and the per-layer penalty.) Builds the 4-layer test model exactly as moe_smoke_train.py
(gradient checkpointing, FSDP2 bf16), runs the step-0 micro-batch, and for the balance penalty ALONE (no language-model
loss) records: whether the recorded router scores carry a gradient path (grad_fn), and the size of the gradient that
reaches each layer's router weight. Repeats without gradient checkpointing. One JSON line per check to --out.
Run: torchrun --nproc_per_node=1 moe_grad_check.py --out FILE
"""
import argparse, json, os
import numpy as np, torch, torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import OlmoeConfig, OlmoeForCausalLM
from transformers.models.olmoe.modeling_olmoe import load_balancing_loss_func

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/scratch/ukrenk/moe_smoke_20261020/data")
ap.add_argument("--out", required=True)
a = ap.parse_args()
dist.init_process_group("nccl"); torch.cuda.set_device(0); dev = torch.device("cuda")
cfg = OlmoeConfig(vocab_size=151936, hidden_size=2048, intermediate_size=1024, num_hidden_layers=4,
                  num_attention_heads=16, num_key_value_heads=16, num_experts=64, num_experts_per_tok=8,
                  max_position_embeddings=4096, rope_theta=10000.0, router_aux_loss_coef=0.01, output_router_logits=True,
                  norm_topk_prob=False, tie_word_embeddings=False, use_cache=False)
train = np.load(os.path.join(a.data, "train_tokens.npy"), mmap_mode="r")
x = torch.from_numpy(np.stack([train[i * 4096:(i + 1) * 4096] for i in (0, 1)]).astype(np.int64)).to(dev)
f = open(a.out, "w")
for gc, fsdp in ((True, True), (False, True), (False, False)):
    torch.manual_seed(0)
    m = OlmoeForCausalLM(cfg)
    if gc: m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if fsdp:
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in m.model.layers: fully_shard(layer, mp_policy=mp)
        fully_shard(m, mp_policy=mp); m.to(dev)
    else:
        m.to(dev, torch.bfloat16)
    m.train()
    out = m(input_ids=x, labels=x)
    rl = out.router_logits
    pooled = out.aux_loss
    per_layer = torch.stack([load_balancing_loss_func((r,), 64, 8) for r in rl]).mean()
    for name, loss in (("pooled_hf", pooled), ("per_layer", per_layer)):
        m.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        g = []
        for layer in m.model.layers:
            w = layer.mlp.gate.weight.grad
            if w is not None and hasattr(w, "full_tensor"): w = w.full_tensor()
            g.append(None if w is None else round(w.float().norm().item(), 6))
        rec = dict(grad_ckpt=gc, fsdp=fsdp, penalty=name, value=round(loss.item(), 4),
                   router_scores_have_grad_fn=[r.grad_fn is not None for r in rl], router_weight_grad_norm=g)
        f.write(json.dumps(rec) + "\n"); f.flush(); print(json.dumps(rec), flush=True)
    del m, out, rl; torch.cuda.empty_cache()
f.close(); dist.destroy_process_group()
