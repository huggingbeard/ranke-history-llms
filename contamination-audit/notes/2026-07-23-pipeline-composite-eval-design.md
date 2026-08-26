# Design: composite pipeline evaluation (front-matter dater + content reader)
2026-07-23, office box. PI question: overall pipeline recall+precision against hand-validated docs; adding per-stage rates is wrong.

WHY SEQUENTIAL, NOT ADDITIVE: stage 2 sees only stage-1 survivors (composition = the hard cases); stage errors correlate (no-imprint books are also archaic-body books — the got-through class). Composite must run docs through both stages in order on one panel.

ESTIMANDS: composite recall per stratum; corpus loss (P(removed|clean)) instead of precision (prevalence-driven; give precision as f(prevalence) in one line); marginal stage-2 value = composite minus stage-1-only, per stratum. Policy call needed first: reprints of old works = contamination? (recommend YES for training-corpus purposes.)

PANEL (hybrid):
- Recall side, labels by construction: 239 gold contam_modern + 600 certified B + 80 reprints + 3 got-throughs + NEW ~200 METADATA-CORRUPTED plants (strip imprint / plant pre-1929 year / drop copyright page — copy corruption patterns from the 3 real got-throughs). Without corrupted plants the test is rigged for stage 1.
- Clean side: 918 certified negatives + a few hundred uncertified incl. serials (serial_census stratum).
- Residual audit: stratified sample of the KEPT pile post-both-stages, oversample high-risk cells (no-imprint, 1920s meta years, homograph-adjudicated, serials), 1.5-3k docs, gold-protocol LLM audit + ~100-doc HUMAN anchor; IPW; report misses-per-10k-kept with Poisson CI.

MECHANICS: stage-1 verdicts exist for all 38k eval docs (fullrun_v2/v3); content judgments exist for the 2,840 panel books; only new stage-1 work = corrupted plants. Pre-register stage-2 operating point BEFORE scoring: proposal = 35B+gptoss OR ks3/3 8x2000 random (held-out C 0.902/B 0.899 @ 10.8%) or strict 4/3 (@ ~6.8%).

POWER: ~1,100 pos/900 neg -> +-2pp composite; corrupted stratum n=200 -> +-5pp; residual bounded by audit size.

PRIOR (falsifiable): stage 1 ~0.88-0.95 on honest metadata; stage 2 mainly rescues the corrupted-metadata stratum; composite corpus loss = stage-1 loss + 6-10%; irreducible residual = archaic-body + no-imprint.

AWAITING PI: (a) reprint policy, (b) operating-point confirmation. Cost: ~$2-4 LLM + ~100 books human time; rest is existing data + local pipeline.
