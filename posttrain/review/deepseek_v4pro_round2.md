# DeepSeek-V4-Pro plan review, round 2
2026-09-28T23:35:04.146914
meta: {"finish_reason": "stop", "provider": "Relace", "cost": 0.0717528, "usage": {"prompt_tokens": 3858, "completion_tokens": 20115, "total_tokens": 23973, "cost": 0.0717528, "is_byok": false, "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 0, "video_tokens": 0}, "cost_details": {"upstream_inference_cost": 0.0717528, "upstream_inference_prompt_cost": 0.0013503, "upstream_inference_completions_cost": 0.0704025}, "completion_tokens_details": {"reasoning_tokens": 19882, "image_tokens": 0, "audio_tokens": 0}}}

**MUST-FIX**

1. **Validation‑test section (4) is stale and contradicts the round‑1 fixes.**  
   Section 4 still lists the original V3 (random‑passage grounding test) and the original V4 (Talkie‑answer vs. base‑Ranke preference test) as the pass/fail gates. Section 4b correctly replaces them with V3b (corrupted‑answer grounding test) and a new V4 (intact vs. corrupted/truncated preference test). An operator following section 4 would run the old, weaker tests, undoing the round‑1 requirement. **Update section 4 to make V3b and the new V4 the gating tests; demote the old tests to “reported but not gating”.**

**SHOULD‑FIX**

2. **DPO candidate sampling parameters are missing.**  
   The plan says the SFT‑3000 adapter “writes 4 sampled answers per question” but gives no temperature, top‑p, etc. Without diversity the DPO pairs collapse. Specify (e.g., temperature 0.8, top‑p 0.9).

3. **DPO reference model not stated.**  
   Standard DPO uses the SFT model as reference; make explicit that SFT‑3000 is the reference.

4. **“Best vs. worst” selection rule is vague.**  
   Explain how the 4 answers are ranked from Talkie’s pairwise judgments (e.g., win count, only consistent pairs).

5. **Talkie preference judge may be biased toward its own style.**  
   The validation uses Talkie’s own answers, not SFT‑3000 answers. A human spot‑check of 20 actual DPO pairs (allowed as a check, not selection) would increase confidence.

6. **Grounding‑gain percentile threshold (20th) is arbitrary.**  
   Justify the choice or describe a sensitivity check; otherwise the filter’s behaviour is opaque.

7. **Cache‑decoder validation (V1) uses only 8 prompts of similar length.**  
   Add a quick test with varied prompt lengths to catch attention‑mask bugs before the full run.

8. **DPO training script not confirmed.**  
   The plan says the SFT script exists; confirm that a DPO script (4‑bit LoRA, β=0.1) is ready or can be adapted without delay.

9. **H100/A100 typo.**  
   The title says H100 but the body consistently says A100 80 GB. Correct the title to avoid confusion when selecting the instance.

10. **Generation order under the 3‑hour time limit.**  
    Prioritise the first two candidates for all passages; only generate third candidates if time remains and the yield pilot projects a shortfall. State this order explicitly.

**VERDICT: REVISE 1**
