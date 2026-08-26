# Tier-1 target grid — DRAFT for PI vetting

**Date:** 2026-07-12 · **Version:** v1 (PROVISIONAL — assistant-drafted; every row awaits (a) JV
vetting, (b) mechanical verification: Google Books pre-cutoff frequency for every tell in EN **and DE
and FR**, single-token check in the Ranke tokenizer, Wikidata dates)
**Companion:** notes/2026-07-12-associate-emergence-probe.md (design), results/superlatives/
assoc_inject_1913.json (sensitivity certificate), vocab_census_{1913,1929}.json (Tier-0 pilot).

## Rules learned tonight, now binding for the grid

1. **Doubly-anachronistic tells only** (JV's genocide correction): word AND referent must postdate
   the cutoff. No Auschwitz (town name pre-exists), no Holocaust (lowercase pre-war usage).
2. **Trilingual polysemy guard** (census finding): the corpus is EN+DE+FR, so a tell must be innocent
   of *any* pre-cutoff sense in *all three* languages. The census caught "email" reading TRAINED —
   almost certainly German *Email*/French *émail* (= enamel), a perfectly period word. Every tell
   gets a DE/FR sense check.
3. **Cluster-counting, not token-counting** (inject finding): planting ONE association (Hitler→
   Stalin) also warmed embedding-neighbours (Eisenhower, Nazis). Tells with similar vectors are ONE
   signal family; the battery's statistics count distinct clusters that fire, never raw token counts.
4. **Multi-token rule needed before the real run**: Gestapo, gulag, Leningrad, Stalingrad, Politburo,
   penicillin all dropped silently as multi-token. Declare ONE rule (sum of subtoken log-probs via
   slp, which handles any length) and use it everywhere; single-token-only is a pilot crutch.
5. **Free-association carriers, ≥5 phrasings per target** (the best-wording lesson); scored with the
   certified batched slp machinery at the output; layer profile only as secondary diagnostic.

## The grid (targets × what would betray contamination)

Era buckets after each cutoff; ~5 targets per cell drafted here, to be topped up mechanically from
Wikidata notability rankings. For every target: tells (Class A existence / Class B link), innocents
(period-legitimate associates that SHOULD fire if the model knows the target at all), and controls
(frequency-matched period surname + invented name) — floors from the standard fiction battery.

### Cutoff 1913 (guesser model) — the cleaner testbed (pre-fame Hitler precedent)
| era | persons | events/orgs | terms/things |
|---|---|---|---|
| 1914–23 | Hitler (BURNED dev), Lenin*, Ludendorff-as-putschist, Ebert, Atatürk | the World War, Versailles, the League, NSDAP, USSR | tank (sense-shift, risky), U-boat*, soviet, fascist, insulin |
| 1924–39 | Stalin-as-ruler, Mussolini-as-Duce, Roosevelt (FDR sense), Franco*, Chamberlain-as-appeaser | Reichstag fire, Anschluss-as-event, Spanish Civil War, New Deal | Gestapo, Stuka, kolkhoz, penicillin, nylon |
| 1940–60 | Eisenhower (FLAGGED: pilot whisper), Churchill-as-PM sense, Truman, Adenauer, Gandhi-as-Mahatma | Pearl Harbor, D-Day, the UN, NATO, the Cold War | radar, jeep, napalm, transistor, laser |

### Cutoff 1929 (guesser + witness model)
| era | persons | events/orgs | terms/things |
|---|---|---|---|
| 1930–39 | Hitler-as-Chancellor (Class B: →Chancellor, →Reich), Franco, Keynes-as-General-Theory | Third Reich, Wehrmacht, Popular Front, Munich-agreement sense | Stuka, autobahn*, sulfa |
| 1940–60 | Eisenhower, de Gaulle, Truman, Mao-as-ruler | WW2 (all Class B links), Hiroshima-as-event, the UN, NATO, Israel-as-state | radar, nylon, jet (sense), transistor |
| 1960+ | Kennedy-as-president (Class B), Gorbachev, Obama | moon landing, Vietnam-as-war, the EU | laser, internet, software, smartphone (clean-control) |

*Starred = known-risky (polysemy or pre-cutoff presence); kept to be killed or confirmed by the
mechanical check, not by taste. Every Hitler row is dev-burned; the held-out verdicts come only from
targets not probed before the freeze.

### Anchors (must-fire / must-not-fire, for calibrating the battery itself)
- **Positive anchors** (date-spike-confirmed leaks; the battery must flag these or it is too dull):
  WW2's end, the UN, the Civil Rights Act, "Holocaust"-adjacent links via Class B pairs.
- **Negative anchors** (must stay dark or the battery is paranoid): smartphone-era targets on both
  models (census: essentially never-seen), invented persons, and the whole grid run on
  **Talkie-1930**, which is known cleaner and must read darker than Ranke throughout.
- **Sensitivity anchor**: the Hitler→Stalin inject (certified tonight: rank 11,357→106 on held-out
  carriers, key-specific, collateral-free).

## Freeze protocol
1. JV vets rows (kill/keep/add).  2. Mechanical pass fills Google-Books pre-cutoff counts (EN/DE/FR)
per tell + tokenizer forms.  3. Lists hashed, committed to LOG.  4. One run per model incl.
Talkie-1930; verdicts by cluster-count vs fiction floor; report either way.
