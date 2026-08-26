# Tier-1 target grid — v2 (post JV vetting pass 1)

**Date:** 2026-07-12 · **Version:** v2 — supersedes 2026-07-12-tier1-target-grid-draft.md (v1, kept)
**Vetting status:** JV reviewed v1. Decisions implemented here:
- **DROP Lenin** (target): pre-1913 presence in socialist/émigré press is real and substantial —
  post-cutoff associates could ride legitimate period knowledge; and the Bolshevik leak axis is
  already covered by the date-spike anchors.
- **DROP U-boat** (term): INVALID under the doubly-anachronistic rule — the word and the thing are
  both pre-1913 (U-1 launched 1906). v1's star was too kind; it is out, not risky.
- **ADD Trotsky** (person target, 1913 cutoff): pre-1913 presence exists (1905 St Petersburg Soviet)
  — acceptable on the TARGET side (like 1929-Hitler); the tells are sharply post-cutoff Class-B
  links: →Stalin (the rivalry, 1920s), →Red Army (1918+), →exile/Mexico (1929/1937). Bonus: the
  certification exam already established Trotsky's 1920s fame profile in these corpora.
- **ADD poison gas** (term target, 1913 cutoff, sense-shift class): weapon sense is 1915+ (Ypres);
  and the paper's own circulation arm measured poison gas as the ONE future absent from pre-1913
  print (fig/tab circ: gas 5.04 vs cavalry 4.29) — so any war-associates behind "poison gas" in the
  1913 model are anachronistic by our own published yardstick. Tells are Class-B links: →Ypres,
  →trenches, →Flanders, →masks (each needs the EN/DE/FR pre-1913 co-occurrence check; "chlorine"
  excluded — industrial chlorine-is-poisonous is period-legitimate chemistry).

All other v1 content unchanged and still awaiting the mechanical pass. Binding rules 1–5 unchanged.

## The grid (v2)

### Cutoff 1913 (guesser model) — the cleaner testbed
| era | persons | events/orgs | terms/things |
|---|---|---|---|
| 1914–23 | Hitler (BURNED dev), **Trotsky**, Ludendorff-as-putschist, Ebert, Atatürk | the World War, Versailles, the League, NSDAP, USSR | tank (sense-shift, risky), **poison gas (sense-shift)**, soviet, fascist, insulin |
| 1924–39 | Stalin-as-ruler, Mussolini-as-Duce, Roosevelt (FDR sense), Franco*, Chamberlain-as-appeaser | Reichstag fire, Anschluss-as-event, Spanish Civil War, New Deal | Gestapo, Stuka, kolkhoz, penicillin, nylon |
| 1940–60 | Eisenhower (FLAGGED: pilot whisper), Churchill-as-PM sense, Truman, Adenauer, Gandhi-as-Mahatma | Pearl Harbor, D-Day, the UN, NATO, the Cold War | radar, jeep, napalm, transistor, laser |

### Cutoff 1929 (guesser + witness model)
| era | persons | events/orgs | terms/things |
|---|---|---|---|
| 1930–39 | Hitler-as-Chancellor (Class B: →Chancellor, →Reich), Franco, Keynes-as-General-Theory | Third Reich, Wehrmacht, Popular Front, Munich-agreement sense | Stuka, autobahn*, sulfa |
| 1940–60 | Eisenhower, de Gaulle, Truman, Mao-as-ruler | WW2 (all Class B links), Hiroshima-as-event, the UN, NATO, Israel-as-state | radar, nylon, jet (sense), transistor |
| 1960+ | Kennedy-as-president (Class B), Gorbachev, Obama | moon landing, Vietnam-as-war, the EU | laser, internet, software, smartphone (clean-control) |

*Starred = known-risky, kept to be killed or confirmed by the mechanical check.

### Anchors (unchanged from v1)
Positive: date-spike-confirmed leaks (WW2's end, the UN, CRA, Holocaust-adjacent Class-B pairs).
Negative: smartphone-era targets; the full grid on Talkie-1930 must read darker than Ranke.
Sensitivity: the certified Hitler→Stalin inject (assoc_inject_1913.json).

### Freeze protocol (unchanged)
1. Remaining JV vetting (rows not yet ruled on). 2. Mechanical pass: Google Books pre-cutoff counts
EN/DE/FR per tell + tokenizer forms (multi-token → summed-slp rule). 3. Hash lists, commit to LOG.
4. One run per model incl. Talkie-1930; cluster-count verdicts vs the fiction floor; report either way.
