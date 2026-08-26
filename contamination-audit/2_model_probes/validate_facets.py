"""
Validate candidate facets with DeepSeek-V4-Pro (OpenRouter) before launching.
Per EVENT, DeepSeek checks each facet for:
  - true_correct        : is the TRUE answer factually correct?
  - requires_post_cutoff: would answering REQUIRE post-1929 knowledge? (FALSE if the answer is a
                          pre-1929 name/place/thing recognisable without knowing the event -> "old words")
  - twin quality        : all false twins clearly wrong, same type, plausible, not give-aways
Returns keep / fix (with better_twins) / drop. Checkpointed + parallel.
Writes 5_results/dig_validated.json (the launch-ready battery) + facet_validation_raw.json.
"""
import os, sys, json, re, asyncio, importlib
sys.path.insert(0, r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api")
from openrouter_safe_call import or_chat
sys.path.insert(0, os.path.dirname(__file__))
CANDMOD = sys.argv[1] if len(sys.argv) > 1 else "dig_candidates"
SUF = sys.argv[2] if len(sys.argv) > 2 else ""
CANDIDATES = importlib.import_module(CANDMOD).CANDIDATES

CUTOFF = 1929
RAW = rf"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\facet_validation_raw{SUF}.json"
OUTJSON = rf"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\dig_validated{SUF}.json"

RUBRIC = """You are auditing probe questions for a study on a time-locked language model trained ONLY on text written up to the year {cutoff}. We test whether it secretly knows about an event by minimal pairs: a PROMPT, the TRUE completion, and several FALSE 'twin' completions; the model passes only if it prefers the true completion.

EVENT: "{event}" ({year}) -- this event is {rel} the {cutoff} cutoff.

{screen}

Evaluate EACH facet below. Return ONLY a JSON array (no prose, no markdown fences), one object per facet IN ORDER:
{{"i": <index int>, "true_correct": <bool>, "requires_post_cutoff": <bool>, "twin_problems": [<twin strings that are also-true, wrong-type, absurd, or give the answer away>], "better_twins": [<up to 5 improved false twins: same type & comparable fame, clearly WRONG, single words/short phrases matching the true answer's style>], "verdict": "keep"|"fix"|"drop", "reason": "<=12 words"}}
Rules: verdict "drop" if true_correct is false{droprule}. verdict "fix" if salvageable by swapping twins (put replacements in better_twins). verdict "keep" if true answer and all twins are fine. Always populate better_twins with strong alternatives even when keeping.

FACETS:
{facets}
Return only the JSON array."""

POST_SCREEN = (
 "CRITICAL SCREEN. A facet is VALID only if its TRUE answer is IMPOSSIBLE to produce from text written up to "
 "{cutoff}. JUDGE THE ANSWER, NOT THE TOPIC: the prompt may freely name a pre-{cutoff} person or place as CONTEXT "
 "(e.g. 'Churchill became prime minister in 19__' -> 40; 'Shostakovich's 1937 symphony was his ___' -> Fifth) -- "
 "KEEP these, because the ANSWER (a date / coined term / post-{cutoff}-specific detail) is what requires post-{cutoff} "
 "text. Set requires_post_cutoff = FALSE and verdict 'drop' for ANY of these:\n"
 "(1) PERSON already publicly known before ~{cutoff}: a model that read pre-{cutoff} text completes their name "
 "regardless of any later event, so a name-completion tests pre-{cutoff} FAME, not contamination. DROP examples: "
 "Winston->Churchill (WWI First Lord, 1920s Chancellor), Dmitri->Shostakovich (First Symphony 1925), Jean-Paul->Sartre, "
 "George->Orwell, Franklin->Roosevelt (NY governor 1929), Philippe->Petain (WWI hero), Stalin, Mussolini, Hitler, "
 "Einstein, Lenin. KEEP a person ONLY if they were genuinely OBSCURE until AFTER {cutoff} (Truman, Eisenhower-as-president, "
 "Kennedy, Nixon, Reagan, Brezhnev, Gorbachev, Khrushchev, Yeltsin, Renzi, Berlusconi, Merkel, Thatcher, Macron, Gagarin, Salk).\n"
 "(2) PLACE or THING that existed before {cutoff}: DROP (Falklands, Berlin, Hiroshima, Stalingrad, Dunkirk, Vietnam, Cuba, "
 "Pearl Harbor, the Moon, Auschwitz-the-town, the Reichstag-building). Recognising an old name proves nothing.\n"
 "(3) Answer inferable from general pre-{cutoff} knowledge without the event: DROP.\n"
 "A facet is GOOD (requires_post_cutoff = TRUE) only if the answer is an ARBITRARY term/name/code COINED AFTER {cutoff} "
 "(Little Boy, Fat Man, Enola Gay, Bockscar, Manhattan Project, Barbarossa, Blitzkrieg, Enabling Act, Iron Curtain, "
 "Checkpoint Charlie, NATO, Watergate, glasnost, perestroika, Sputnik, NASA, Apollo, Vietnamization, Brexit, Obamacare, "
 "transistor, double helix), OR a post-{cutoff}-emergent obscure person (list above), OR a specific event YEAR.\n"
 "(4) NAME-CUE RULE (critical): the PROMPT must NOT contain the answer's first name or any token that cues it. "
 "A first name turns the item into name-completion (tests fame), not knowledge. DROP any facet whose prompt "
 "primes the answer, e.g. '...was Leonid [Brezhnev]' or '...was Martin Luther [King]'. The model must produce "
 "the answer from the DESCRIPTIVE FACT alone. (Referring to a DIFFERENT person by surname as context is fine.)\n"
 "Be strict: when in doubt, DROP.")
PRE_SCREEN = ("This is a PRE-cutoff positive-control event. Ignore requires_post_cutoff (set it true). "
              "Only check that the true answer is correct and the twins are clearly wrong but plausible.")


def build_msg(cat, event, year, facets):
    post = (year > CUTOFF) and (cat != "PRE-control")
    lines = []
    for i, (p, t, tw) in enumerate(facets):
        lines.append(f'[{i}] prompt: "{p}"  TRUE: "{t.strip()}"  TWINS: {[x.strip() for x in tw]}')
    return RUBRIC.format(cutoff=CUTOFF, event=event, year=year,
                         rel=("AFTER" if post else "BEFORE/at"),
                         screen=(POST_SCREEN.format(cutoff=CUTOFF) if post else PRE_SCREEN),
                         droprule=(" OR requires_post_cutoff is false" if post else ""),
                         facets="\n".join(lines))


def parse_json(txt):
    txt = re.sub(r"^```(json)?|```$", "", txt.strip(), flags=re.MULTILINE).strip()
    a = txt.find("["); b = txt.rfind("]")
    s = txt[a:b + 1] if a >= 0 and b > a else txt
    s2 = re.sub(r",(\s*[}\]])", r"\1", s)            # strip trailing commas
    try:
        return json.loads(s2)
    except Exception:
        objs = []                                     # salvage object-by-object
        for m in re.finditer(r"\{[^{}]*\}", s):
            try:
                objs.append(json.loads(re.sub(r",(\s*})", r"\1", m.group(0))))
            except Exception:
                pass
        return objs


def main():
    raw = json.load(open(RAW)) if os.path.exists(RAW) else {}

    async def run():
        sem = asyncio.Semaphore(8); loop = asyncio.get_event_loop()

        async def do(idx, cat, event, year, facets):
            key = f"{idx}|{event}"
            if key in raw:
                return
            msg = build_msg(cat, event, year, facets)
            async with sem:
                try:
                    txt, meta = await loop.run_in_executor(
                        None, lambda: or_chat([{"role": "user", "content": msg}],
                                              max_tokens=9000, reasoning_max=3500, temperature=0.1))
                    raw[key] = {"text": txt, "cost": meta.get("cost"), "provider": meta.get("provider")}
                    print(f"  ok  {event:<32} ${meta.get('cost') or 0:.4f}")
                except Exception as e:
                    print(f"  ERR {event:<32} {repr(e)[:90]}")
        await asyncio.gather(*[do(i, c, e, y, f) for i, (c, e, y, f) in enumerate(CANDIDATES)])

    asyncio.run(run())
    json.dump(raw, open(RAW, "w"), indent=1)

    # build validated battery
    validated = []
    kept = dropped = errs = 0
    print("\n--- validation summary ---")
    for idx, (cat, event, year, facets) in enumerate(CANDIDATES):
        key = f"{idx}|{event}"
        if key not in raw:
            print(f"  {event}: NO RESPONSE"); continue
        try:
            verds = parse_json(raw[key]["text"])
            vmap = {v["i"]: v for v in verds}
        except Exception as e:
            errs += 1; print(f"  {event}: PARSE FAIL {repr(e)[:60]}"); continue
        keep_facets = []
        for i, (p, t, tw) in enumerate(facets):
            v = vmap.get(i)
            if not v:
                continue
            if v.get("verdict") == "drop" or v.get("true_correct") is False:
                dropped += 1
                print(f"    DROP {event} [{i}] {t.strip():<14} :: {v.get('reason','')}")
                continue
            # merge twins (original minus flagged problems + better_twins), keep leading space style
            probs = set(x.strip() for x in v.get("twin_problems", []))
            merged = [x for x in tw if x.strip() not in probs]
            for bt in v.get("better_twins", []) or []:
                cand = " " + bt.strip()
                if cand not in merged and bt.strip().lower() != t.strip().lower():
                    merged.append(cand)
            if len(merged) >= 1:
                keep_facets.append([p, t, merged]); kept += 1
        if keep_facets:
            validated.append([cat, event, year, keep_facets])
    json.dump(validated, open(OUTJSON, "w"), indent=1)
    print(f"\nkept {kept} facets, dropped {dropped}, parse-errors {errs}.  -> {OUTJSON}")
    by = {}
    for (cat, ev, yr, fs) in validated:
        by.setdefault(cat, [0, 0]); by[cat][0] += 1; by[cat][1] += len(fs)
    for cat, (ne, nf) in by.items():
        print(f"  {cat:<16} {ne} events, {nf} validated facets")


if __name__ == "__main__":
    main()
