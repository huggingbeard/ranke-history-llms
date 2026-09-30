# Status

Updated: 28 September 2026, 23:05 Europe/Zurich

## Locked experiment

- Talkie writes the training questions and answers. Ranke learns from accepted
  Talkie records. A modern model may accept or reject records but may not change
  their wording.
- Writer: `talkie-lm/talkie-1930-13b-it`, exact revision
  `8033675be6360ae0127fa75f941c12d52064f1dc`.
- Student: `uzh-echist-org/Ranke-4B-1929_v2`, exact revision
  `fb4cede16ed59e1249ebeba2b4e5b9ec2a560642`.
- The unchanged source corpus has 2,077 passages and SHA-256
  `803d71395ad53355f61e04e249def5a1e2798c4e48d83a05d9af49d4a909ba04`.
- The frozen evaluation has 84 prompts and SHA-256
  `156eb18d5796a2db123b40ad641b30cb74c4763964077f2c1bca5e8c75796277`.
  It combines 36 history reasoning prompts with 48 period chat prompts.
- Planned nested SFT sizes are 300, 1,000, and 3,000 accepted records. DPO is
  deferred until SFT shows a useful effect.

## Completed

- The exact Ranke base model answered all 84 frozen prompts on the local RTX
  4090 through the compute router. Router log:
  `C:\Users\hvoth\cluster\router\results\c2b1414dce.log`.
- The complete answer matrix is
  `experiments/base84_20260928_172133/eval/blind_output_matrix.jsonl`, 84 rows,
  SHA-256 `567c28a7b380d615f0b5482579b7ed0cae789499a40cb0d2f7281430ea57a863`.
- Deterministic surface measures show a poor base: 74/84 answers reached the
  token ceiling, 72/84 triggered the repetition rule, and 29/84 generated a new
  `Question:` or `Answer:` turn inside the answer. These measures do not judge
  historical correctness. The row data and summary are under
  `experiments/base84_20260928_172133/eval/surface/`.
- The Russo-Japanese War and Angell baseline answers are preserved in the same
  matrix for exact paired comparison after each adapter.

## Talkie loader work and current gate

- The official Talkie checkpoint is `rl-refined.pt`, 26,560,686,563 bytes,
  SHA-256 `c6fd903b67ad312c7537095fa982b0e77735ccb689c8cfc774ca492694e70064`.
- The official loader first held the 26.6 GB BF16 state dictionary in host RAM,
  then constructed a roughly 52 GB FP32 model. That transient host use caused
  the 64 GB H100 job `c7a67f5c03-t` / Slurm `6619271` to be killed before it
  wrote a candidate. This was a loader failure, not a model result.
- The generator now memory maps the exact checkpoint, creates the model on the
  meta device, assigns the mapped BF16 tensors, rebuilds rotary buffers, and
  then moves the exact weights to the GPU. Model, revision, prompt, and sampling
  settings are unchanged. It also logs process, cgroup, and CUDA memory.
- The patched one-record H100 gate, router job `14fd5c9b82-t` / Slurm `6627767`,
  remained `PENDING` for reason `Priority` and was cancelled at 18:07 before it
  started. It consumed no GPU time.
- The replacement one-record A100 gate, router job `6ee65aff6c-t` / Slurm
  `6641452`, loaded the exact checkpoint and generated one record in 235.5
  seconds, but failed the gate. Talkie returned a fluent declarative passage
  instead of the requested tagged question and answer, so the parser reported
  `missing_tags` and the validator accepted 0/1. The saved raw completion is
  under `scratch/gate_6ee65aff6c/`. This was an instruction-format failure,
  not a loader or memory failure.
- The smallest generation fix leaves the locked user prompt, corpus, model
  revision, and sampling values unchanged. The code starts Talkie's assistant
  turn with the non-semantic delimiter `<question>`; Talkie still supplies
  every question and answer word. The change is recorded per row as
  `assistant_prefix`.
- Router job `b4046c2067` / Slurm `6656781` then failed after 25 seconds. The
  private Talkie generator requires an explicit `top_k`; the wrapper omitted
  it. The saved raw row records the exact `TypeError`. No model output or
  training data was produced.
- The wrapper now passes and records `top_k=40`. Static compilation, parser
  testing, and a fake generator call with the exact signature passed locally.
  The old pending gate `aa5dede128` / Slurm `6657370` was cancelled before it
  started and used zero GPU time.
- Claude Opus 5.5 reviewed the replacement generator, validator, fail-closed
  verifier, router wrapper, and later 16-record wrapper through the installed
  first-party subscription CLI. It returned `GO - safe to submit the
  one-record gate`. The full result is
  `scratch/claude_opus55_preflight_v2_result.json`.
- The reviewed immutable code archive has SHA-256
  `650121dec90729c6a52cca307e8c3bc2d27cc5a189f06f74e7530e22dec1a515`.
  The 16-record smoke input has SHA-256
  `7f1b4b9bcd9adb2780569ce6dd9806439e400d51f24f099ad16591160144d505`.
  The generator writes its run request before model loading, streams every raw
  chunk to a flushed partial file, keeps error rows, and writes a complete
  receipt only when every requested generation succeeds.
- The exact one-record cluster rehearsal is router job `6510f2b9ee` / Slurm
  `6657946`, run tag `opus55_gate_20260928_2304`: A100, 64 GB host RAM, one CPU,
  and a 30-minute wall limit. At 23:05 its state was `PENDING` for `Priority`.
  It had used zero GPU time. No 16-record request will be submitted until this
  gate passes its saved-artifact verifier.
- Earlier pending duplicate gates `d8805666a3-t` and `d8d673cf4c-t` were
  cancelled. No full Talkie generation has started yet.
- The active 5-minute continuation heartbeat now names only the current gate
  `6510f2b9ee`, its exact output paths and hashes, and the separate 16-record
  launcher. It is instructed to stay quiet while the queue state is unchanged
  and never submit a duplicate.

## Evaluation report implementation

- The report now states the two claims separately without changing any judge
  score: period fluency is the mean of coherence, naturalness, and period fit
  on all 84 prompts; historical substance is the substance score on the frozen
  36 history prompts.
- The report code passed a complete disposable smoke test and wrote the mock
  report under `scratch/report_smoke/`. This proves the report path works; it is
  not an experiment result.

## Next automatic steps

1. Wait for router job `6510f2b9ee` to run, then fetch and inspect its exact
   one-record output.
2. If it passed, add the outer full-run checks requested by Claude Opus 5.5,
   obtain a new Opus subscription review, and only then submit and validate the
   16-record smoke.
3. Generate two candidates for each of the 2,077 passages.
4. Accept or reject exact Talkie records without rewriting them.
5. Build nested 300, 1,000, and 3,000 training files.
6. Train the three QLoRA adapters and answer the same 84 frozen prompts after
   each one.
7. Report the paired slope for surface fluency and historical reasoning, with
   complete Russo-Japanese War and Angell examples.

## 2026-09-29 08:20 — Claude Opus 5.5 takeover (plan v2, `PLAN_v2_mutual_judging.md`)
- RunPod A100 pod 0e4uyirst5nfhz used 2026-09-28 23:45 to 2026-09-29 07:40 (~$12.60, of which ~6.5 h idle waiting for JV = ~$10 wasted). Terminated 07:40; account pod list empty at 07:40 and 07:45. JV revoked RunPod for Claude; cluster only.
- Key/value-cache Talkie decoder: exact in float32 (8/8 greedy match, logit diff ~1e-4); ~90 records/min on A100.
- Talkie IT ends its turn after the question; fix: code inserts `</question>\n<answer>` and Talkie continues (16/16 parsed).
- Talkie answers are short (median 28 words). JV chose option B (07:36): minimum answer 20 words, three answers per passage.
- 192 records written on RunPod (runs/v2_20260929_0015/out/talkie_raw.jsonl, sha256 e885cdff…40ab), pushed to cluster for resume.
- Cluster job 882eb2fbbc / Slurm 6670265 (A100, 8 h limit): Talkie writing -> Ranke scoring -> 3 SFT runs -> 84-question answers. PENDING (Priority) since 08:18.
