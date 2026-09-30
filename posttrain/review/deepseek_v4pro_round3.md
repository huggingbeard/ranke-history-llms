# DeepSeek-V4-Pro plan review, round 3
2026-09-28T23:40:17.717818
meta: {"finish_reason": "stop", "provider": "Relace", "cost": 0.0361438, "usage": {"prompt_tokens": 3248, "completion_tokens": 10002, "total_tokens": 13250, "cost": 0.0361438, "is_byok": false, "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 0, "video_tokens": 0}, "cost_details": {"upstream_inference_cost": 0.0361438, "upstream_inference_prompt_cost": 0.0011368, "upstream_inference_completions_cost": 0.035007}, "completion_tokens_details": {"reasoning_tokens": 10073, "image_tokens": 0, "audio_tokens": 0}}}

**MUST-FIX**

1. **Grounding‑gain filter is not validated for its primary purpose (removing post‑1913 anachronisms) and the threshold is arbitrary.**  
   The V3 test only checks sensitivity to a sentence swapped from a *different* passage; it does not test whether the filter rejects answers that insert a fact from 1914–1929 while staying on‑topic.  Moreover the 20th‑percentile cut is chosen solely to reach 3 000 records – it could be negative, keeping answers where the passage *hurts* log‑probability.  If the filter fails, the training data will contain anachronistic content and the period‑lock claim collapses.  
   *Fix:* (a) Add a validation that inserts known post‑1913 facts (e.g., “1914”, “Treaty of Versailles”, “League of Nations”) into a sample of answers and verifies that the gain drops below a meaningful threshold for ≥90 % of them.  (b) Set the acceptance threshold using that test (e.g., gain > 0 and above the value that rejects 90 % of anachronistic insertions while retaining ≥80 % of clean answers).  If yield falls below 3 000, generate extra candidates rather than lowering the bar.

2. **Answer‑overlap contamination with the eval set is not checked.**  
   The plan only guards against question overlap (Jaccard on the question text).  Training *answers* could still closely resemble the base model’s answers to the 84 frozen prompts, leading to memorisation and inflated eval scores.  
   *Fix:* Compute a similarity metric (e.g., ROUGE‑L) between each candidate answer and the base model’s existing answers to the eval prompts; exclude records above a safe threshold (e.g., 0.5) or, at minimum, report the overlap and discuss the risk.

**SHOULD-FIX**

3. **Fluency filter lacks a principled floor.**  The 5th percentile is a moving target that may admit very low‑quality answers.  Consider setting an absolute minimum fluency (e.g., the median fluency of a small set of human‑written period answers) or at least reporting how many accepted records fall below a sensible bound.

4. **Time budget is optimistic; no contingency for a partial Talkie run that yields too few records for the dose curve.**  If the cached decoder fails and the fallback is slow, the 3‑hour cut could leave <300 records, collapsing the dose design.  Pre‑specify a rule for reducing the dose levels proportionally (e.g., 100/300/900) if the total accepted is below 1 000.

5. **Inter‑judge reliability for the modern evaluators is not specified.**  Report agreement (e.g., Pearson/Spearman correlation, Cohen’s κ) to let the reader gauge the stability of the scores.

**VERDICT: REVISE 1,2**
