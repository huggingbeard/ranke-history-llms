# -*- coding: utf-8 -*-
"""Step 1 of the preregistered pipeline: OVER-GENERATE candidate forced-choice items with DeepSeek-V4-Pro
(per PREREGISTRATION.md), to be blind-screened next. Saves candidates_raw.json (checkpointed per task).
POST items must be post-1930 (valid for both Ranke-1929 and Talkie-1930); PRE controls pre-1920."""
import os, sys, json, re
sys.path.insert(0, r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api")
from openrouter_safe_call import or_chat
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "candidates_raw.json")

# (cluster_id, level, n_target, topic description, era)
TASKS = [
 # ---- narrow deep clusters (POST, delta +20pp, need ~32 post-screen -> over-generate 40) ----
 ("Cold War", "narrow", 55,"the Cold War 1945-1991: superpower confrontation, nuclear arms race, Berlin, Cuba, Korea, alliances (NATO/Warsaw Pact), space race, detente, the fall of communism", "post"),
 ("Vietnam War", "narrow", 55,"the American war in Vietnam 1955-1975: battles, operations, the politics, the end of the war and reunification", "post"),
 ("US civil rights", "narrow", 55,"the US civil-rights era 1948-1968 specifically (NOT pre-1929 race history): desegregation, key events, marches, laws, leaders, court cases", "post"),
 ("US politics 1945-74", "narrow", 55,"US national politics 1945-1974 (Truman through Watergate): presidents and their terms, major policies/programmes, scandals, elections, doctrines", "post"),
 # ---- broad-pool breadth (POST, other post-war sub-topics; ~12 each) ----
 ("decolonisation", "broad", 18,"decolonisation and new nations after 1945: independence of colonies, partitions, new states, their first leaders and founding events", "post"),
 ("space & computing", "broad", 18,"post-1945 science, space and computing: satellites, the space race, the Moon, the transistor/integrated circuit, computers, the internet", "post"),
 ("science & medicine", "broad", 18,"post-1945 science and medicine: DNA, the genetic code, vaccines (polio), antibiotics in clinical use, major discoveries and their discoverers", "post"),
 ("economics & institutions", "broad", 18,"post-1945 economics and international institutions: Bretton Woods, the UN, the EU/Common Market, currencies, oil shocks, major economic events", "post"),
 ("culture & society", "broad", 18,"post-1945 Western culture and society: television, popular music, landmark films, sport, major cultural milestones", "post"),
 ("disasters & conflicts", "broad", 18,"post-1945 disasters and conflicts (excluding Vietnam/Cold-War already covered): wars, nuclear accidents, notable catastrophes and where/when they happened", "post"),
 # ---- pre-cutoff positive controls (PRE, <1920, prove the probe fires) ----
 ("pre-controls", "pre", 40, "well-established, FAMOUS history BEFORE 1920 across many areas (politics, war, science, exploration, culture) that any educated reader of pre-1929 books would know -- pick iconic facts with clearly distinct distractors, not obscure dates", "pre"),
]

CUT = 1930
GOOD = '{"label":"17th parallel","year":1954,"prompt":"After the 1954 Geneva conference, Vietnam was divided into a communist north and a non-communist south at the","answer":" 17th parallel","distractors":[" 16th parallel"," 38th parallel"," 49th parallel"]}'
BAD = '{"prompt":"The Soviet-led alliance was called the ___ Pact","answer":" Warsaw"}  // WRONG: the form word "Pact" is cued in the prompt'


def prompt_for(topic, n, era):
    when = (f"text written AFTER {CUT}" if era != "pre" else "ordinary pre-1920 sources")
    rule1 = (f"The tested fact/relationship must be POST-{CUT}. The entity may be older, but the specific relation/event must be post-{CUT} (e.g. the city of Hiroshima is old, but 'Hiroshima -> atomic bombing' is post-1930 and valid)."
             if era != "pre" else
             "The fact must be well-established and dated clearly BEFORE 1920, so a model with a ~1929 cutoff should know it.")
    return f"""You are building a contamination-detection test for a language model whose training data ends around {CUT}. We need forced-choice items where ONLY knowledge from {when} can identify the correct answer.

Generate {n} items about: {topic}.

STRICT RULES:
1. {rule1}
2. The PROMPT is a bare description that contains NO part of any answer -- no names, words, or forms from the answer. Do NOT cue the answer's shape. (BAD: {BAD})
3. The TRUE answer is the COMPLETE phrase a person would say, with a leading space (e.g. " the Warsaw Pact", " 1962", " Lyndon Johnson").
4. Exactly 3 DISTRACTORS: real and not excludable a priori by someone who LACKS the fact -- BUT a person who KNOWS the fact must be able to pick the true answer CONFIDENTLY. So distractors must be clearly DISTINCT from the answer and from each other: NO adjacent years (bad: 1903 vs 1901/1902/1904; good: 1903 vs 1886/1925/1947), NO near-duplicates, NO near-synonyms. They may vary in form. No absurd/joke options.
5. Everything FACTUALLY CORRECT. Include the event year as an integer.
6. Diverse facts within the topic; do not cluster on one event.
7. STRONGLY PREFER answers that are NAMES, EVENTS, TERMS, PLACES, ORGANISATIONS or short PHRASES -- NOT years/dates. At most ~1 in 6 answers may be a year, and only with distractor years spanning >=25 years (e.g. 1962 vs 1934/1948/1979), never close together. (Year-with-nearby-year items are rejected.)

Return ONLY a JSON array, no prose. Each element exactly:
{{"label": "short tag", "year": 1954, "prompt": "...", "answer": " ...", "distractors": [" ...", " ...", " ..."]}}

GOOD EXAMPLE: {GOOD}

Generate {n} items now as a JSON array."""


def parse_json_array(txt):
    txt = txt.strip()
    txt = re.sub(r"^```(?:json)?", "", txt).strip()
    txt = re.sub(r"```$", "", txt).strip()
    i, j = txt.find("["), txt.rfind("]")
    if i >= 0 and j > i:
        txt = txt[i:j + 1]
    txt = re.sub(r",\s*([\]}])", r"\1", txt)  # trailing commas
    try:
        return json.loads(txt)
    except Exception:
        out = []  # salvage: parse each flat {...} object individually, skip the malformed one
        for m in re.finditer(r"\{[^{}]*\}", txt):
            try:
                out.append(json.loads(re.sub(r",\s*([\]}])", r"\1", m.group(0))))
            except Exception:
                pass
        return out


MODEL = "deepseek/deepseek-v3.2"  # deepseek-v4-pro was retired on OpenRouter

if __name__ == "__main__":
    items = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else []
    done = {it["cluster"] for it in items}
    for (cid, level, n, topic, era) in TASKS:
        if cid in done:
            print("skip (done):", cid); continue
        print(f"generating {n} for [{cid}] ...", flush=True)
        try:
            txt, meta = or_chat([{"role": "user", "content": prompt_for(topic, n, era)}],
                                model=MODEL, max_tokens=28000, reasoning_max=None, temperature=0.4)
            arr = parse_json_array(txt)
        except Exception as e:
            print("  FAILED", cid, repr(e)[:200]); continue
        k = 0
        for it in arr:
            if not all(key in it for key in ("prompt", "answer", "distractors")) or len(it.get("distractors", [])) != 3:
                continue
            it["cluster"] = cid; it["level"] = level; it["era"] = "pre" if era == "pre" else "post"
            items.append(it); k += 1
        json.dump(items, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"  +{k} items (total {len(items)})  cost ${meta.get('cost') or 0:.4f}", flush=True)
    print("DONE ->", OUT, "| total", len(items))
