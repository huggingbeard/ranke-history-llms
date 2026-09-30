"""Batched Talkie decoding with a key/value cache.

Uses the exact TalkieModel weights and repeats its layer arithmetic
(talkie-upstream commit 35317ba, src/talkie/model.py). The only new part is the
cache: the original forward() recomputes the whole sequence for every token.
Left padding; per-row rotary positions; padded keys masked.
Validated against the uncached forward() by `validate_kv` (plan test V1).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from talkie.model import apply_rotary_emb
from talkie.sampling import apply_top_k_top_p

# 2026-09-29 profiling on the cluster (torch 2.11, H100): scaled_dot_product_attention dispatched to
# the cuDNN backend, which rebuilt its plan every decode step (10.7 ms CPU per call, 82% of run time;
# 32x512 speed test 224.6 s vs 37.5 s on the RunPod A100 with torch 2.8). Use the other kernels.
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)


def _rot(model, pos):
    # model.cos: [1, S, 1, D/2] -> per-row [B, T, 1, D/2]
    return model.cos[0][pos], model.sin[0][pos]


def _block_step(block, e_x, x, cos, sin, cache_k, cache_v, start, mask):
    attn = block.attn
    h = F.rms_norm(x, (x.shape[-1],))
    bsz, t, _ = h.shape
    q = attn.attn_query(h).view(bsz, t, attn.n_head, attn.head_dim)
    k = attn.attn_key(h).view(bsz, t, attn.n_head, attn.head_dim)
    v = attn.attn_value(h).view(bsz, t, attn.n_head, attn.head_dim)
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
    q = attn.head_gain(q)
    cache_k[:, :, start:start + t] = k.transpose(1, 2)
    cache_v[:, :, start:start + t] = v.transpose(1, 2)
    kk = cache_k[:, :, :start + t]
    vv = cache_v[:, :, :start + t]
    y = F.scaled_dot_product_attention(q.transpose(1, 2), kk, vv, attn_mask=mask)
    y = y.transpose(1, 2).contiguous().view(bsz, t, -1)
    x = x + block.attn_gain(attn.attn_resid(y))
    x = x + block.mlp_gain(block.mlp(F.rms_norm(x, (x.shape[-1],))))
    x = x + block.embed_skip(e_x)
    return x


AUTOCAST = True  # set False for the float32 diagnostic


def _ac():
    import contextlib
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if AUTOCAST else contextlib.nullcontext()


class KVRunner:
    def __init__(self, model):
        self.m = model
        cfg = model.config
        self.n_layer, self.n_head, self.head_dim = cfg.n_layer, cfg.n_head, cfg.head_dim

    def _logits(self, x):
        m = self.m
        x = F.rms_norm(x, (x.shape[-1],))
        return F.linear(x[:, -1, :], m.lm_head_gain(m.lm_head)).float()

    def _run(self, ids, pos, start, key_valid, caches):
        m = self.m
        bsz, t = ids.shape
        total = start + t
        # mask[b, 1, i, j]: query i (absolute start+i) may see key j
        qi = torch.arange(start, total, device=ids.device)[:, None]
        kj = torch.arange(total, device=ids.device)[None, :]
        causal = (kj <= qi)[None, None]
        valid = key_valid[:, None, None, :total]
        mask = causal & valid
        # a padded query row with no visible key would give NaN; let it see itself
        diag = (kj == qi)[None, None]
        mask = mask | diag
        cos, sin = _rot(m, pos)
        x = m.embed(ids)
        x = F.rms_norm(x, (x.shape[-1],))
        e_x = x
        for li, block in enumerate(m.blocks):
            x = _block_step(block, e_x, x, cos, sin, caches[li][0], caches[li][1], start, mask)
        return self._logits(x)

    @torch.no_grad()
    def generate(self, prompts_ids, max_new, temperature, top_p, top_k, seeds,
                 stop_ids, stop_text_fn=None, decode_fn=None, greedy=False, min_new=0):
        """prompts_ids: list[list[int]]. Returns list of generated id lists and finish reasons."""
        m = self.m
        dev = m.embed.weight.device
        B = len(prompts_ids)
        lens = [len(p) for p in prompts_ids]
        L = max(lens)
        total = L + max_new
        pad = [L - n for n in lens]
        ids = torch.zeros((B, L), dtype=torch.long, device=dev)
        key_valid = torch.zeros((B, total), dtype=torch.bool, device=dev)
        for b, p in enumerate(prompts_ids):
            ids[b, pad[b]:] = torch.tensor(p, device=dev)
            key_valid[b, pad[b]:] = True
        pad_t = torch.tensor(pad, device=dev)[:, None]
        pos = (torch.arange(L, device=dev)[None, :] - pad_t).clamp(min=0)
        caches = [(torch.empty((B, self.n_head, total, self.head_dim), dtype=(torch.bfloat16 if AUTOCAST else m.embed.weight.dtype), device=dev),
                   torch.empty((B, self.n_head, total, self.head_dim), dtype=(torch.bfloat16 if AUTOCAST else m.embed.weight.dtype), device=dev))
                  for _ in range(self.n_layer)]
        gens = [torch.Generator(device=dev).manual_seed(int(s)) for s in seeds]
        V = m.config.vocab_size
        top_p_t = None if top_p is None or top_p >= 1 else torch.full((B, 1), float(top_p), device=dev)
        top_k_t = None if not top_k else torch.full((B,), int(top_k), dtype=torch.long, device=dev)
        out = [[] for _ in range(B)]
        finished = [False] * B
        reasons = ["length"] * B
        with _ac():
            logits = self._run(ids, pos, 0, key_valid, caches)
            cur = L
            stop_list = torch.tensor(sorted(stop_ids), device=dev) if stop_ids else None
            for step in range(max_new):
                if step < min_new and stop_list is not None:
                    logits[:, stop_list] = float("-inf")  # optional minimum length (plan v2.9 option)
                if greedy:
                    nxt = torch.argmax(logits, dim=-1)
                else:
                    lg = logits / temperature
                    if top_p_t is not None or top_k_t is not None:
                        lg = apply_top_k_top_p(lg, top_p=top_p_t, top_k=top_k_t)
                    noise = torch.stack([
                        -torch.log(-torch.log(torch.rand((V,), device=dev, generator=g) + 1e-20) + 1e-20)
                        for g in gens])
                    nxt = torch.argmax(lg + noise, dim=-1)
                nl = nxt.tolist()
                for b in range(B):
                    if finished[b]:
                        continue
                    tok = nl[b]
                    if tok in stop_ids:
                        finished[b], reasons[b] = True, "stop"
                        continue
                    out[b].append(tok)
                    if stop_text_fn is not None and decode_fn is not None and len(out[b]) % 8 == 0 and len(out[b]) >= min_new:
                        if stop_text_fn(decode_fn(out[b][-40:])):
                            finished[b], reasons[b] = True, "stop_text"
                if all(finished) or step == max_new - 1:
                    break
                step_ids = nxt[:, None]
                step_pos = (torch.full((B, 1), cur, device=dev) - pad_t).clamp(min=0)
                key_valid[:, cur] = True
                logits = self._run(step_ids, step_pos, cur, key_valid, caches)
                cur += 1
        return out, reasons


@torch.no_grad()
def uncached_greedy(model, ids, n):
    """Reference: the original forward(), one prompt, greedy."""
    x = torch.tensor([ids], device=model.embed.weight.device)
    got, last = [], None
    with _ac():
        for i in range(n):
            logits = model.forward(x)
            if i == 0:
                last = logits[0].clone()
            t = int(torch.argmax(logits, -1)[0])
            got.append(t)
            x = torch.cat([x, torch.tensor([[t]], device=x.device)], 1)
    return got, last


@torch.no_grad()
def cached_first_logits(runner, prompts_ids):
    m = runner.m
    dev = m.embed.weight.device
    B, L = len(prompts_ids), max(len(p) for p in prompts_ids)
    pad = [L - len(p) for p in prompts_ids]
    ids = torch.zeros((B, L), dtype=torch.long, device=dev)
    kv = torch.zeros((B, L), dtype=torch.bool, device=dev)
    for b, p in enumerate(prompts_ids):
        ids[b, pad[b]:] = torch.tensor(p, device=dev)
        kv[b, pad[b]:] = True
    pos = (torch.arange(L, device=dev)[None] - torch.tensor(pad, device=dev)[:, None]).clamp(min=0)
    caches = [(torch.empty((B, runner.n_head, L, runner.head_dim), dtype=(torch.bfloat16 if AUTOCAST else m.embed.weight.dtype), device=dev),
               torch.empty((B, runner.n_head, L, runner.head_dim), dtype=(torch.bfloat16 if AUTOCAST else m.embed.weight.dtype), device=dev))
              for _ in range(runner.n_layer)]
    with _ac():
        return runner._run(ids, pos, 0, kv, caches)
