# DeepSeek-V4-Pro plan review, round 1
2026-09-28T23:26:31.272669
meta: {"finish_reason": "stop", "provider": "Relace", "cost": 0.02312135, "usage": {"prompt_tokens": 3171, "completion_tokens": 6289, "total_tokens": 9460, "cost": 0.02312135, "is_byok": false, "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 0, "video_tokens": 0}, "cost_details": {"upstream_inference_cost": 0.02312135, "upstream_inference_prompt_cost": 0.00110985, "upstream_inference_completions_cost": 0.0220115}, "completion_tokens_details": {"reasoning_tokens": 5988, "image_tokens": 0, "audio_tokens": 0}}}

**MUST-FIX**

1. **Grounding-gain judge is unvalidated for actual data quality.**  
   The V3 test only checks discriminability (true passage vs. random passage), not whether high-gain answers are factually grounded or period‑appropriate. A model this small and period‑locked may reward superficial lexical overlap or generic fluency. Without a human spot‑check of accepted/rejected records (e.g., 50 of each) the entire training set is of unknown quality. Add a mandatory manual inspection step before scaling; if the filter is not clearly separating grounded from ungrounded answers, the experiment is invalid.

2. **Talkie preference‑judge validation (V4) is trivially passed and gives false confidence.**  
   Comparing Talkie’s own answers against known‑poor base‑Ranke answers does not test whether Talkie can distinguish better from worse *SFT‑generated* answers. DPO will amplify whatever biases Talkie has. Replace V4 with a set of manually constructed pairs where one answer contains a clear factual error (anachronism or contradiction of the passage). If Talkie cannot reliably prefer the correct answer (e.g., < 75% accuracy), Stage‑D must be dropped. Without this, the DPO stage is unsupported and likely harmful.

3. **No yield estimation – risk of too few accepted records for the dose curve.**  
   Mechanical rejection plus grounding/fluency filters could leave far fewer than 300 records, collapsing the dose design. Before the full run, generate a pilot batch (e.g., 200 records), apply the full filter pipeline, and project the final accepted count. If the projection is < 1000, adjust thresholds or increase candidates per passage *before* committing the full compute. Without this, the experiment may fail for lack of training data.

4. **Cached‑decoder fallback may break the 7‑hour budget.**  
   The plan relies on a new batched KV‑cached decoder for Talkie. If V1 fails and the fallback (uncached batched) is too slow, the run will exceed the time/money cap. Define a hard time checkpoint after Stage‑A: if generation is not complete by hour 3, reduce the number of passages (e.g., randomly subsample to 1000) to stay within budget. Otherwise the run will not finish.

**SHOULD-FIX**

5. **Statistical power with 84 prompts is borderline for the decision rule.**  
   A 5‑point difference on a 100‑point scale with 84 items may be noisy; consider a paired permutation test and report confidence intervals transparently. Acknowledge the limitation in the report.

6. **DPO training lacks any monitoring.**  
   Add a held‑out set of preference pairs (or track reward margin) to detect overfitting in the single epoch.

7. **Protected eval‑term filtering may miss near‑duplicates.**  
   Use fuzzy matching (e.g., token‑set similarity) to avoid training on paraphrases of eval prompts; document the exact protected term list.

8. **Clarify passage date range.**  
   State explicitly that all passages are pre‑1913; otherwise “post‑1913 leakage” is ill‑defined.

9. **Consider dropping DPO entirely.**  
   Given the high risk of an invalid preference judge, a simpler SFT‑only dose‑response experiment would be cheaper, faster, and more interpretable.

10. **Run a miniature end‑to‑end dry run (e.g., 50 records) before the overnight launch** to catch integration bugs (file paths, hashing, resume logic) that could waste the entire window.

VERDICT: REVISE 1 2 3 4
