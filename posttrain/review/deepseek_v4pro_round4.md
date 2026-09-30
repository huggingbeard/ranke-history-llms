# DeepSeek-V4-Pro plan review, round 4
2026-09-28T23:44:39.939113
meta: {"finish_reason": "stop", "provider": "Relace", "cost": 0.0346808, "usage": {"prompt_tokens": 3848, "completion_tokens": 9524, "total_tokens": 13372, "cost": 0.0346808, "is_byok": false, "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 0, "video_tokens": 0}, "cost_details": {"upstream_inference_cost": 0.0346808, "upstream_inference_prompt_cost": 0.0013468, "upstream_inference_completions_cost": 0.033334}, "completion_tokens_details": {"reasoning_tokens": 9085, "image_tokens": 0, "audio_tokens": 0}}}

**MUST-FIX**
1. **Custom batched decoder is a single point of failure.** The plan’s entire timeline hinges on a newly written KV‑cache batched decoder that has only been tested on 8 prompts with a loose logit tolerance (0.5 in bf16). A subtle bug or memory issue will force the fallback to uncached decoding, which cannot finish within the 3‑hour Talkie window. Replace the custom decoder with a standard, well‑tested autoregressive generation loop (e.g., per‑sequence KV cache using the existing `forward()` or a converted HuggingFace model). If a custom batched decoder is kept, validation must cover ≥100 prompts, require max logit difference <1e‑3, and include a throughput benchmark proving the full generation finishes in ≤2 hours.
2. **Grounding gain > 0 is too weak.** A mean per‑token log‑prob difference barely above zero can arise from noise, especially with many records. The filter must require a meaningful effect size (e.g., gain > 0.1) or a statistical significance test (e.g., t‑test on the per‑token differences) to ensure only answers genuinely supported by the passage are kept.
3. **V3 deciding test pass bar is too low.** Requiring only 80% of corrupted‑answer pairs to show higher gain for the intact answer means the judge can fail on 20% of clear content swaps. This undermines confidence in the grounding filter. Raise the pass criterion to ≥90%, matching the V3b injection test.
4. **No viable fallback if the batched decoder fails.** The plan’s fallback (“uncached batched decoding”) is impractically slow and will not meet the time limit. Implement and benchmark a reliable fallback (e.g., sequential generation with a per‑sequence KV cache) that can complete the full Talkie generation within the 3‑hour cap.
5. **Late‑fact word list still has dangerous gaps.** Talkie’s training data extends to 1930; the current list misses many WWI‑era concepts (e.g., “trench warfare”, “poison gas”, “U‑boat”, “Marne”, “Ypres”, “no man’s land”, “Western Front”). A single anachronistic sentence that evades the list can contaminate training. Expand the list substantially, or add a secondary filter using Ranke‑1929’s own perplexity to flag answers that are anomalously unlikely for the period.
6. **Evaluation judges are not calibrated for period_fit.** Modern models are asked to score “period_fit” without any demonstrated ability to distinguish period‑appropriate from anachronistic answers. Include a small, fixed calibration set of known in‑period and anachronistic answers (created by the experimenters, not by modern models) and report judge accuracy before scoring the main eval.

**SHOULD-FIX**  
- Clarify whether the 20th percentile for grounding gain is computed on all mechanically filtered records or only on those with gain > 0.  
- Report the gain distribution per dose to check for quality confounding.  
- Consider filtering out records with abnormally high fluency (generic/repetitive answers).  
- Tighten the V1 logit tolerance to ≤1e‑3 even if the custom decoder is kept.  
- Acknowledge that the decision rule for a dose‑response slope (≥5/100 in ≥4/6 groups) may be underpowered with 84 items.  
- Add a check for near‑verbatim copying from the passage to prevent inflated gain.  
- Periodically save intermediate results to external storage to survive pod preemption.

VERDICT: REVISE 1,2,3,4,5,6
