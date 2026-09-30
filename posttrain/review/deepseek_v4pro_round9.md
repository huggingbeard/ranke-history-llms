# DeepSeek-V4-Pro plan review, round 9
2026-09-29T00:17:30.386164
meta: {"finish_reason": "stop", "provider": "Relace", "cost": 0.0482671, "usage": {"prompt_tokens": 4736, "completion_tokens": 13317, "total_tokens": 18053, "cost": 0.0482671, "is_byok": false, "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 0, "video_tokens": 0}, "cost_details": {"upstream_inference_cost": 0.0482671, "upstream_inference_prompt_cost": 0.0016576, "upstream_inference_completions_cost": 0.0466095}, "completion_tokens_details": {"reasoning_tokens": 12725, "image_tokens": 0, "audio_tokens": 0}}}

**MUST‑FIX** (would make the result invalid or the run fail)

1. **Anachronism criterion is statistically unsound.**  
   The plan states “without more anachronism” is decided by a one‑sided paired permutation test that the condition’s mean period_fit is “not lower than base” at p<0.05.  
   - If this means *failing to reject* the null that period_fit is ≥ base (i.e. p≥0.05), then a non‑significant result is taken as evidence of no increase. With N=84, power is low; a true increase in anachronism could easily be missed, allowing a false claim of “without more anachronism”.  
   - If it means *rejecting* the null that period_fit is lower (p<0.05 to conclude it is not lower), that is a valid non‑inferiority test but is likely underpowered and may prevent any effect from being declared.  
   The decision rule must be made statistically defensible. At minimum, pre‑specify a non‑inferiority margin (e.g. the 95% CI for the difference must lie entirely above –0.2 on the 0–4 scale) and use an appropriate equivalence/non‑inferiority procedure (e.g. TOST). Without this, the “without more anachronism” gate is not trustworthy and can invalidate the whole effect claim.

2. **Multiple comparisons inflate the false‑positive rate for substance.**  
   Three dose conditions are each tested against base at p<0.05 (one‑sided), and an effect is declared if *any* condition beats base. This yields a family‑wise error rate >0.05. To keep the inference valid, the analysis must correct for multiplicity (Bonferroni, Holm, or a pre‑specified primary comparison with hierarchical testing) or use a permutation‑based max‑T test that controls the family‑wise error rate. Without adjustment, a statistically significant result cannot be interpreted as a reliable effect.

3. **Context length overflow in grounding‑gain computation is unchecked.**  
   Ranke‑4B has a finite maximum sequence length. The concatenation of passage + question + answer may exceed that limit, causing silent truncation that biases the log‑probabilities and the resulting grounding gain. The plan does not mention verifying that all sequences fit, nor does it specify a truncation strategy. This can corrupt the primary filter that selects training data. You must either confirm that every sequence stays within the model’s context window or implement explicit truncation rules and report any truncation.

4. **Slope threshold “≥ 5/100” is undefined and likely meaningless.**  
   The dose‑response slope condition requires SFT‑3000 to beat SFT‑1000 “by ≥ 5/100”. The substance score scale is not given in this document (the judge prompt likely uses 0–4). If the scale is 0–4, 5/100 = 0.05, which is far below any meaningful difference and could be triggered by noise. The threshold must be expressed in the actual score units (e.g. ≥0.2 on a 0–4 scale) and justified. Ambiguity here makes the slope conclusion uninterpretable.

**SHOULD‑FIX** (improve rigour but do not alone invalidate the run)

5. The V2 format pass bar (12/16 parsed) is lenient; a 25% failure rate would waste many records. Consider raising it to 14/16 or iterating on the prompt until the rate is higher.
6. The V1 decoder batch‑size‑1 cross‑check allows up to 4/64 mismatches. Any mismatch should be investigated and understood before trusting the batched decoder for the full run.
7. The late‑fact word list, while extensive, is not exhaustive. Acknowledge explicitly that some anachronisms may slip through and that the grounding gain and final period_fit evaluation are the backstops.
8. Verify that the answer‑only loss masking in the reused SFT script is correct (e.g. by a quick gradient check on a tiny sample) to avoid accidentally training on question/passage tokens.

**VERDICT: REVISE** 1,2,3,4
