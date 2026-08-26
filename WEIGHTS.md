# Ranke / Endeavour model weights — local locations

**Files/paths referenced:**
- HF org (gated): https://huggingface.co/uzh-echist-org
- HF token: `D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt`
- Canonical local store: `D:\hf-models\<model-name>`
- HF cache (older copies): `C:\Users\hvoth\.cache\huggingface\hub\models--uzh-echist-org--*`
- Download/refresh script: `tools/download_weights.py`

**Date:** 2026-08-26  **Version:** v1

---

## Why weights are not in this git repo

Each model is 3–9 GB. Weights are **not** committed here. They live on local disk
(`D:\hf-models`) and on the gated Hugging Face org `uzh-echist-org`. This file is the
pointer. Re-download any model with `tools/download_weights.py` (needs the HF token).

## Local inventory (verified 2026-08-26)

| Model (on `uzh-echist-org`) | Type | Cutoff | Size | Local? |
|---|---|---|---|---|
| Ranke-4B-1900 | base | 1900 | 7.51 GB | C: cache |
| Ranke-4B-1913 | base | 1913 | 8.23 GB | C: + D: |
| Ranke-4B-1913-1202-Instruct | instruct | 1913 | 7.51 GB | C: + D: |
| Ranke-4B-1913-1202-Instruct-WT | instruct (WT) | 1913 | 8.84 GB | D: (added 2026-08-26) |
| Ranke-4B-1929 | base | 1929 | 7.51 GB | C: + D: |
| Ranke-4B-1929-1202-Instruct | instruct | 1929 | 7.51 GB | C: + D: |
| ranke-1.7b-1929-3006 | small | 1929 | 3.80 GB | C: cache |
| Endeavour-1913-1.7B | small base | 1913 | 3.44 GB | D: (added 2026-08-26) |
| Endeavour-1913-1.7B-1708-Instruct | small instruct | 1913 | 3.46 GB | D: (added 2026-08-26) |
| Endeavour-1913-1.7B-1808-Instruct | small instruct | 1913 | 3.46 GB | D: (added 2026-08-26) |
| Endeavour-1913-1.7B-2708-GRPO | small GRPO | 1913 | 3.46 GB | D: (added 2026-08-26) |
| Endeavour-1913-1.7B-2808-GRPO | small GRPO | 1913 | 3.46 GB | D: (added 2026-08-26) |

**Complete:** every model published on the HF org is now stored locally.

## Not yet released (announced cutoffs)

The prerelease notes announce cutoffs **1933, 1939, 1946**. As of 2026-08-26 these are
**not on the HF org** — nothing to download yet. Re-run `tools/download_weights.py` (after
adding the new repo ids) when they appear.

## Notes / cleanup opportunities

- The Ranke-4B weights are **duplicated**: one copy in the C: HF cache, one in `D:\hf-models`.
  1900 base and `ranke-1.7b-1929-3006` sit **only** in the C: cache.
- Moving a live model store is risky (a `robocopy /MOVE` once broke 19 Ollama models), so the
  duplicate C: copies were left in place. If you want to free ~46 GB on C:, move them to
  `D:\hf-models` deliberately (copy, verify, then delete the cache), not with `/MOVE`.

## Related datasets on the HF org (not downloaded here)

SFT / GRPO / partisanship datasets exist under `uzh-echist-org` (e.g. `ranke-1913-sft-dataset`,
`grpo-1913-2708`, `partisanship-1913`). Some are already cached under
`C:\Users\hvoth\.cache\huggingface\hub\datasets--uzh-echist-org--*`.
