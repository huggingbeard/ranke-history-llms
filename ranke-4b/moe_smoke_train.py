"""OLMoE smoke test, step 2 (2026-10-20, v1): train the OLMoE design from random weights on pre-1900 tokens.

Design = OLMoE-1B-7B-0924 (Ai2): 16 layers, width 2048, 16 heads, 64 experts, 8 used per token, expert width 1024,
QK-norm, RoPE theta 10,000 - via Hugging Face transformers' OlmoeForCausalLM (pure PyTorch, no compiled kernels).
Changes from the released config (stated, deliberate):
  * Ranke's Qwen3 tokenizer: vocab 151,936 instead of 50,304 (keeps later comparisons with Ranke-4B design-only).
  * Short warm-up (20 steps) to a peak LR of 4e-4, then constant; the release warms up over ~10.5B tokens.
  * Load-balancing loss 0.01 (HF built-in) + router z-loss 0.001 added here (OLMoE uses both).
Run: torchrun --nproc_per_node=<cards> moe_smoke_train.py --out DIR [--steps N] [--save-at S] [--resume CKPT_DIR]
Data order is fixed (step s uses sequences s*32 ... s*32+31), so a resumed run must reproduce the losses of an
uninterrupted run. Saves every step's metrics with time and tokens seen (steps.jsonl), expert-load checks
(balance.jsonl), validation loss at step 0, every --val-every steps and at the end (val.jsonl), a resumable checkpoint
(model + optimizer + step) at --save-at with its size and save time (checkpoints.jsonl), and a finished marker (run.json).
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import OlmoeConfig, OlmoeForCausalLM
from transformers.models.olmoe.modeling_olmoe import load_balancing_loss_func

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/scratch/ukrenk/moe_smoke_20261020/data")
ap.add_argument("--out", required=True)
ap.add_argument("--layers", type=int, default=16)
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--save-at", type=int, default=0)
ap.add_argument("--resume", default="")
ap.add_argument("--global-bs", type=int, default=32)
ap.add_argument("--micro-bs", type=int, default=2)
ap.add_argument("--seq", type=int, default=4096)
ap.add_argument("--lr", type=float, default=4e-4)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--balance-every", type=int, default=10)
ap.add_argument("--balance-loss", choices=["pooled", "per-layer"], default="pooled",
                help="pooled = Hugging Face default (all layers pooled; misses per-layer collapse); "
                     "per-layer = balance loss per layer, averaged over layers, as in the original OLMoE code")
ap.add_argument("--val-seqs", type=int, default=32)
ap.add_argument("--val-every", type=int, default=25)
ap.add_argument("--data-offset", type=int, default=0, help="debug: step s reads the sequences of step s + offset")
ap.add_argument("--experts-impl", default="grouped_mm", choices=["grouped_mm", "eager", "batched_mm"],
                help="Hugging Face expert routine; grouped_mm (default) is the fused kernel")
ap.add_argument("--nan-debug", action="store_true", help="debug: after each micro-batch, record the first weights with non-finite gradients")
a = ap.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dev = torch.device("cuda")
os.makedirs(a.out, exist_ok=True)
log = lambda name, rec: rank == 0 and open(os.path.join(a.out, name), "a").write(json.dumps(rec) + "\n")

cfg = OlmoeConfig(vocab_size=151936, hidden_size=2048, intermediate_size=1024, num_hidden_layers=a.layers,
                  num_attention_heads=16, num_key_value_heads=16, num_experts=64, num_experts_per_tok=8,
                  max_position_embeddings=a.seq, rope_theta=10000.0, router_aux_loss_coef=0.01, output_router_logits=True,
                  norm_topk_prob=False, tie_word_embeddings=False, use_cache=False,
                  experts_implementation=a.experts_impl,
                  pad_token_id=None, bos_token_id=None, eos_token_id=None)   # OLMoE default pad id 1 = the quote mark in the Qwen vocabulary: its embedding would stay zero
torch.manual_seed(0)
model = OlmoeForCausalLM(cfg)
n_params = sum(p.numel() for p in model.parameters())
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for layer in model.model.layers:
    fully_shard(layer, mp_policy=mp)
fully_shard(model, mp_policy=mp)
model.to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
lr_at = lambda s: a.lr * min(1.0, (s + 1) / a.warmup)

train = np.load(os.path.join(a.data, "train_tokens.npy"), mmap_mode="r")
val = np.load(os.path.join(a.data, "val_tokens.npy"), mmap_mode="r")
assert train.size >= a.steps * a.global_bs * a.seq, "not enough training tokens"
per_rank = a.global_bs // world
accum = per_rank // a.micro_bs
assert per_rank * world == a.global_bs and accum * a.micro_bs == per_rank


def batch(arr, seq_ids):
    x = np.stack([arr[i * a.seq:(i + 1) * a.seq] for i in seq_ids]).astype(np.int64)
    return torch.from_numpy(x).to(dev, non_blocking=True)


def z_loss(router_logits):
    return torch.stack([torch.logsumexp(r.float(), dim=-1).pow(2).mean() for r in router_logits]).mean()


start = 0
if a.resume:
    msd, osd = get_state_dict(model, opt)
    state = {"model": msd, "optim": osd}
    dcp.load(state, checkpoint_id=a.resume)
    set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
    start = json.load(open(os.path.join(a.resume, "position.json")))["next_step"]
log("run.json", dict(params=n_params, layers=a.layers, world=world, start=start, steps=a.steps, args=vars(a),
                     torch=torch.__version__, gpu=torch.cuda.get_device_name()))


@torch.no_grad()
def validate(step):
    model.eval()
    ids = list(range(rank, a.val_seqs, world))
    tot = torch.zeros(2, device=dev)
    for i in range(0, len(ids), a.micro_bs):
        x = batch(val, ids[i:i + a.micro_bs])
        out = model(input_ids=x, labels=x, output_router_logits=False)
        tot += torch.tensor([out.loss.float() * x.numel(), x.numel()], device=dev)
    dist.all_reduce(tot)
    model.train()
    log("val.jsonl", dict(step=step, val_loss=float(tot[0] / tot[1]), time=time.strftime("%Y-%m-%dT%H:%M:%S")))
    if rank == 0:
        print(json.dumps(dict(step=step, val_loss=float(tot[0] / tot[1]))), flush=True)


torch.cuda.reset_peak_memory_stats()
if start == 0:
    validate(0)   # starting point for the validation curve
for step in range(start, a.steps):
    t0 = time.time()
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    sums = torch.zeros(4, device=dev)  # lm loss, aux loss, z loss, n
    counts = None
    for m in range(accum):
        first = (step + a.data_offset) * a.global_bs + rank * per_rank + m * a.micro_bs
        x = batch(train, range(first, first + a.micro_bs))
        out = model(input_ids=x, labels=x)
        z = z_loss(out.router_logits)
        lm = out.loss - cfg.router_aux_loss_coef * out.aux_loss   # HF adds coef * aux to the loss
        if a.balance_loss == "per-layer":
            aux = torch.stack([load_balancing_loss_func((r,), cfg.num_experts, cfg.num_experts_per_tok) for r in out.router_logits]).mean()
            loss = (lm + cfg.router_aux_loss_coef * aux + 0.001 * z) / accum
        else:
            loss = (out.loss + 0.001 * z) / accum
        loss.backward()
        if a.nan_debug:
            bad = [n for n, p in model.named_parameters() if p.grad is not None and
                   not torch.isfinite(p.grad.to_local() if hasattr(p.grad, 'to_local') else p.grad).all()]
            if bad:
                open(os.path.join(a.out, f'nan_rank{rank}.jsonl'), 'a').write(json.dumps(dict(
                    step=step, micro=m, seqs=list(range(first, first + a.micro_bs)), lm=float(lm), z=float(z),
                    n_bad=len(bad), first_bad=bad[:12])) + "\n")
        sums += torch.stack([lm.detach().float(), out.aux_loss.detach().float(), z.detach(), torch.tensor(1.0, device=dev)])
        if step % a.balance_every == 0 or step == a.steps - 1:
            with torch.no_grad():
                c = torch.stack([torch.bincount(r.topk(cfg.num_experts_per_tok, dim=-1).indices.flatten(), minlength=cfg.num_experts)
                                 for r in out.router_logits]).float()
                counts = c if counts is None else counts + c
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    dist.all_reduce(sums)
    torch.cuda.synchronize()
    dt = time.time() - t0
    rec = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), tokens_seen=(step + 1) * a.global_bs * a.seq, step=step, lm_loss=float(sums[0] / sums[3]), aux_loss=float(sums[1] / sums[3]), z_loss=float(sums[2] / sums[3]),
               grad_norm=float(gnorm.full_tensor() if hasattr(gnorm, "full_tensor") else gnorm), lr=lr_at(step), step_s=round(dt, 2),
               tokens_per_s_per_card=round(a.global_bs * a.seq / dt / world), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
    finite = math.isfinite(rec["lm_loss"]) and math.isfinite(rec["grad_norm"])
    log("steps.jsonl", rec)
    if rank == 0:
        print(json.dumps(rec), flush=True)
    if counts is not None:
        dist.all_reduce(counts)
        share = counts / counts.sum(dim=1, keepdim=True) * cfg.num_experts   # 1.0 = fair share
        log("balance.jsonl", dict(step=step, max_share=[round(v, 2) for v in share.max(dim=1).values.tolist()],
                                  min_share=[round(v, 3) for v in share.min(dim=1).values.tolist()]))
    if not finite:
        log("run.json", dict(stopped="non-finite loss or gradient", step=step))
        break
    if (step + 1) % a.val_every == 0 and step + 1 < a.steps:
        validate(step + 1)
    if a.save_at and step + 1 == a.save_at:
        if (step + 1) % a.val_every:
            validate(step + 1)
        ck = os.path.join(a.out, f"ckpt_step{step + 1}")
        t1 = time.time()
        msd, osd = get_state_dict(model, opt)
        dcp.save({"model": msd, "optim": osd}, checkpoint_id=ck)
        if rank == 0:
            json.dump(dict(next_step=step + 1), open(os.path.join(ck, "position.json"), "w"))
            gb = sum(os.path.getsize(os.path.join(ck, f)) for f in os.listdir(ck)) / 2**30
            log("checkpoints.jsonl", dict(step=step + 1, path=ck, gb=round(gb, 1), save_s=round(time.time() - t1, 1),
                                          time=time.strftime("%Y-%m-%dT%H:%M:%S")))
validate(a.steps)
log("run.json", dict(finished=True, last_step=step, time=time.strftime("%Y-%m-%dT%H:%M:%S")))
dist.destroy_process_group()
