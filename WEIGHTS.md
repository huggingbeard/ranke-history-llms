# Ranke / Endeavour model weights — local locations

**Files/paths referenced:**
- HF org (gated): https://huggingface.co/uzh-echist-org
- HF token: `D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt`
- Canonical local store: `G:\hf-models\<model-name>`
- HF cache (older copies): `C:\Users\hvoth\.cache\huggingface\hub\models--uzh-echist-org--*`
- Download/refresh script: `tools/download_weights.py`

**Date:** 2026-09-07  **Version:** v2
**Supersedes:** v1 (2026-08-26), which pointed the canonical store at `D:\hf-models`.

---

## Why weights are not in this git repo

Each model is 3–9 GB. Weights are **not** committed here. They live on local disk
(`G:\hf-models`) and on the gated Hugging Face org `uzh-echist-org`. This file is the
pointer. Re-download any model with `tools/download_weights.py` (needs the HF token).

## What changed since v1

- The canonical store moved from `D:\hf-models` to **`G:\hf-models`** (around 2026-08-31).
  `D:\hf-models` no longer exists.
- The old C: cache duplicates of the Endeavour models and `Ranke-4B-1913-1202-Instruct-WT`
  are now **empty leftover folders** (0 GB — reference stubs, no weights). The real copies
  are on `G:\hf-models`.
- Two models still hold their **only** real weights in the C: cache: `Ranke-4B-1900` and
  `ranke-1.7b-1929-3006`. They are **not** on `G:\hf-models`. Do not delete the C: cache
  before copying these two to G:.

## Local inventory (verified 2026-09-07, sizes measured on disk)

| Model (on `uzh-echist-org`) | Type | Cutoff | Size | Where the real weights are |
|---|---|---|---|---|
| Ranke-4B-1900 | base | 1900 | 7.51 GB | **C: cache only** |
| Ranke-4B-1913 | base | 1913 | 8.23 GB | G: |
| Ranke-4B-1913-1202-Instruct | instruct | 1913 | 7.51 GB | G: |
| Ranke-4B-1913-1202-Instruct-WT | instruct (WT) | 1913 | 8.23 GB | G: (C: stub is empty) |
| Ranke-4B-1929 | base | 1929 | 7.51 GB | G: |
| Ranke-4B-1929-1202-Instruct | instruct | 1929 | 7.51 GB | G: |
| ranke-1.7b-1929-3006 | small | 1929 | 3.80 GB | **C: cache only** |
| Endeavour-1913-1.7B | small base | 1913 | 3.20 GB | G: (C: stub is empty) |
| Endeavour-1913-1.7B-1708-Instruct | small instruct | 1913 | 3.22 GB | G: (C: stub is empty) |
| Endeavour-1913-1.7B-1808-Instruct | small instruct | 1913 | 3.22 GB | G: (C: stub is empty) |
| Endeavour-1913-1.7B-2708-GRPO | small GRPO | 1913 | 3.22 GB | G: (C: stub is empty) |
| Endeavour-1913-1.7B-2808-GRPO | small GRPO | 1913 | 3.22 GB | G: (C: stub is empty) |

**Complete:** every model published on the HF org is stored locally, across G: and the C: cache.

## Internal test models (not on the HF org)

| Model | Type | Cutoff | Size | Where the real weights are |
|---|---|---|---|---|
| Ranke-MoE-4B-1900 (2026-11-22; 4.19B total, 1.02B per token; card: `MODEL_CARD_Ranke-MoE-4B-1900.md`) | MoE base | 1900 | 8.38 GB | `G:\detritus\ranke-moe-4b\hf_ext86_step41160` (SHA-256 of the weights in the card) |

## Not yet released (announced cutoffs)

The prerelease notes announce cutoffs **1933, 1939, 1946**. As of 2026-09-07 these are
**not on the HF org** — nothing to download yet. Re-run `tools/download_weights.py` (after
adding the new repo ids) when they appear.

## Notes / cleanup opportunities

- `G:\hf-models` also holds two small dataset folders: `posttraining-datasets` (0.13 GB) and
  `llm_global_opinions` (near 0 GB). These are data, not model weights.
- The C: cache still has empty stub folders for the 5 Endeavour models and
  `Ranke-4B-1913-1202-Instruct-WT` (0 GB each). They are safe to remove; the weights are on G:.
- To retire the C: cache fully, first **copy** `Ranke-4B-1900` and `ranke-1.7b-1929-3006` to
  `G:\hf-models`, verify them, then delete the cache copies. Copy — never `robocopy /MOVE`
  (a `/MOVE` once broke 19 Ollama models).

## Related datasets on the HF org (not all downloaded here)

SFT / GRPO / partisanship datasets exist under `uzh-echist-org` (e.g. `ranke-1913-sft-dataset`,
`grpo-1913-2708`, `partisanship-1913`). Many are cached under
`C:\Users\hvoth\.cache\huggingface\hub\datasets--uzh-echist-org--*`.
