"""Router diagnostic for the OLMoE smoke test (2026-10-20). Why does the 1-card mini test (job 7377214) report that 8 of
64 experts take every token from step 0 on, while the built-in balance loss (about 12; 8 = even, 64 = total collapse)
says routing is fairly even? Builds the model exactly as moe_smoke_train.py (same config, seed 0, gradient checkpointing,
FSDP2 bf16), runs the step-0 micro-batch once, and records per layer: logit shape and dtype, logit spread, the part of the
spread shared by all tokens, expert shares, and the top experts. Repeats with a plain bf16 copy (no FSDP) and on a
validation batch. Writes one JSON line per (setting, layer) to --out.
Run: torchrun --nproc_per_node=1 moe_router_diag.py --out FILE [--layers 4]
"""
import argparse, copy, json, os
import numpy as np, torch, torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import OlmoeConfig, OlmoeForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/scratch/ukrenk/moe_smoke_20261020/data")
ap.add_argument("--out", required=True)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--seq", type=int, default=4096)
a = ap.parse_args()
dist.init_process_group("nccl")
torch.cuda.set_device(0)
dev = torch.device("cuda")
cfg = OlmoeConfig(vocab_size=151936, hidden_size=2048, intermediate_size=1024, num_hidden_layers=a.layers,
                  num_attention_heads=16, num_key_value_heads=16, num_experts=64, num_experts_per_tok=8,
                  max_position_embeddings=a.seq, rope_theta=10000.0, router_aux_loss_coef=0.01, output_router_logits=True,
                  norm_topk_prob=False, tie_word_embeddings=False, use_cache=False)
torch.manual_seed(0)
model = OlmoeForCausalLM(cfg)
plain = copy.deepcopy(model).to(dev, torch.bfloat16)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for layer in model.model.layers:
    fully_shard(layer, mp_policy=mp)
fully_shard(model, mp_policy=mp)
model.to(dev)
train = np.load(os.path.join(a.data, "train_tokens.npy"), mmap_mode="r")
val = np.load(os.path.join(a.data, "val_tokens.npy"), mmap_mode="r")
mk = lambda arr, ids: torch.from_numpy(np.stack([arr[i * a.seq:(i + 1) * a.seq] for i in ids]).astype(np.int64)).to(dev)
f = open(a.out, "w")


def report(tag, m, x, train_mode):
    m.train(train_mode)
    with torch.set_grad_enabled(train_mode):
        out = m(input_ids=x, labels=x)
    for li, r in enumerate(out.router_logits):
        rf = r.detach().float().reshape(-1, r.shape[-1])
        c = torch.bincount(rf.topk(8, dim=-1).indices.flatten(), minlength=64).float()
        s = c / c.sum() * 64
        rec = dict(setting=tag, layer=li, shape=list(r.shape), dtype=str(r.dtype), type=type(r).__name__,
                   logit_std=round(rf.std().item(), 4), shared_std=round(rf.mean(0).std().item(), 4),
                   distinct_rows=int(torch.unique(rf[:2048].round(decimals=3), dim=0).shape[0]),
                   share_max=round(s.max().item(), 2), share_min=round(s.min().item(), 3),
                   experts_used=int((c > 0).sum().item()), top=c.topk(8).indices.tolist(),
                   lm_plus_aux_loss=round(out.loss.item(), 4), aux_loss=round(out.aux_loss.item(), 4))
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(json.dumps(rec), flush=True)
    if train_mode:
        out.loss.backward()
        m.zero_grad(set_to_none=True)


x0 = mk(train, [0, 1])
report("fsdp_train_step0", model, x0, True)
report("fsdp_eval_step0", model, x0, False)
report("plain_bf16_eval_step0", plain, x0, False)
report("fsdp_eval_val", model, mk(val, [0, 1]), False)
f.close()
dist.destroy_process_group()
