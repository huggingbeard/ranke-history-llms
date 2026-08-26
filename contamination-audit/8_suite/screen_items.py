# -*- coding: utf-8 -*-
"""Step 2: BLIND screen of candidates_raw.json (no model scores involved -- pure question-fairness).
Two gates: (1) a deterministic cue-check (a word UNIQUE to the true answer must not appear in the prompt);
(2) an adversarial DeepSeek referee on 4 criteria. Only items passing BOTH are kept and LOCKED into
question_bank_v2.json. All verdicts saved to screen_verdicts.json. Run once; do not re-screen to taste."""
import os, sys, json, re
sys.path.insert(0, r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api")
from openrouter_safe_call import or_chat
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "candidates_raw.json")
BANK = os.path.join(HERE, "question_bank_v2.json")
VERD = os.path.join(HERE, "screen_verdicts.json")
MODEL = "deepseek/deepseek-v3.2"
STOP = set("the a an of in on at to and or for was were is are be been being as by with from into during after before that this it its their his her he she they them then over under up down out new first major great big small".split())


def clean(s):
    s = (s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"').replace("&ndash;", "-"))
    return " " + re.sub(r"\s+", " ", s).strip()  # single leading space, no entities/double spaces


def words(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in STOP]


def code_cued(it):
    """A word UNIQUE to the true answer (not shared by any distractor) must not appear in the prompt."""
    aw = set(words(it["answer"]))
    dw = set().union(*[set(words(d)) for d in it["distractors"]]) if it["distractors"] else set()
    giveaway = aw - dw
    return bool(giveaway & set(words(it["prompt"])))


def adjacent_years(it):
    """All four options are years within a 6-year span -> a knower can't discriminate; reject."""
    yrs = []
    for v in [it["answer"]] + it["distractors"]:
        m = re.fullmatch(r"\s*(1[5-9]\d\d|20\d\d)\s*", str(v))
        if not m:
            return False
        yrs.append(int(m.group(1)))
    return (max(yrs) - min(yrs)) <= 6


REF = """You are a STRICT adversarial referee for a language-model contamination test. The model's training ends ~1930, and we are AUDITING whether it knows facts it SHOULD NOT. So for a POST item the correct answer SHOULD be a post-1930 fact -- that is the entire point. NEVER penalise an item for being about a post-1930 event; that is what makes it a valid probe.

Judge ONLY the question's fairness. For each item return booleans + a one-line reason:
- era_ok: is the fact dated on the side of the 1930 wall that its label claims? For a POST item: does the event/relationship genuinely occur AFTER 1930? (Kent State 1970 -> TRUE; Voting Rights Act 1965 -> TRUE; Gulf of Tonkin 1964 -> TRUE.) For a PRE item: is it solidly BEFORE 1920? Mark era_ok FALSE ONLY when the date is on the WRONG side (a 'POST' item whose answer is actually pre-1931, or a 'PRE' item that is post-1919). Old entities are fine if the RELATION is on the right side.
- cued: TRUE if any part of the answer (a name/word or its distinctive form) is given away in the prompt.
- fair_distractors: TRUE only if all 3 distractors are real, equally specific, and a person WITHOUT the fact could not rule them out a priori (not absurd, not obviously wrong).
- discriminable: TRUE only if a person who KNOWS the fact can pick the answer CONFIDENTLY -- the distractors are clearly distinct from the answer and from each other. FALSE for adjacent years (1903 vs 1901/1902/1904), near-duplicate values, or near-synonyms where even a knower would hesitate.
- factually_correct: TRUE if the stated answer is historically correct.

Keep an item iff era_ok AND (not cued) AND fair_distractors AND discriminable AND factually_correct.
Return ONLY a JSON array, one object per item, in order:
[{"i":0,"era_ok":true,"cued":false,"fair_distractors":true,"discriminable":true,"factually_correct":true,"reason":"..."}, ...]

ITEMS (era shown per item; for POST, the answer being post-1930 is CORRECT, not a flaw):
"""


def screen_batch(batch):
    payload = "\n".join(f'{i}. [{it["era"]}] prompt="{it["prompt"]}" answer="{it["answer"]}" distractors={it["distractors"]}'
                        for i, it in enumerate(batch))
    txt, meta = or_chat([{"role": "user", "content": REF + payload}], model=MODEL, max_tokens=8000, reasoning_max=None, temperature=0.0)
    t = re.sub(r"^```(?:json)?|```$", "", txt.strip()).strip()
    i, j = t.find("["), t.rfind("]")
    t = re.sub(r",\s*([\]}])", r"\1", t[i:j + 1])
    return json.loads(t), meta


if __name__ == "__main__":
    cand = json.load(open(RAW, encoding="utf-8"))
    verdicts, kept = [], []
    cost = 0.0
    B = 12
    for s in range(0, len(cand), B):
        batch = cand[s:s + B]
        cc = [code_cued(it) or adjacent_years(it) for it in batch]
        try:
            v, meta = screen_batch(batch); cost += meta.get("cost") or 0
        except Exception as e:
            print("screen FAILED batch", s, repr(e)[:160]); v = [{} for _ in batch]
        for k, it in enumerate(batch):
            vk = next((x for x in v if x.get("i") == k), {}) if isinstance(v, list) else {}
            code_ok = not cc[k]
            llm_ok = (vk.get("era_ok") and not vk.get("cued") and vk.get("fair_distractors")
                      and vk.get("discriminable") and vk.get("factually_correct"))
            keep = bool(code_ok and llm_ok)
            rec = {**it, "code_cued": cc[k], "ref": vk, "keep": keep}
            verdicts.append(rec)
            if keep:
                kept.append({"era": it["era"], "level": it["level"], "topic": it["cluster"],
                             "label": it.get("label", ""), "prompt": it["prompt"].strip(), "true": clean(it["answer"]),
                             "distractors": [clean(d) for d in it["distractors"]], "year": it.get("year")})
        print(f"  screened {min(s+B,len(cand))}/{len(cand)}  kept {len(kept)}", flush=True)
    json.dump(verdicts, open(VERD, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump({"schema": "forced_choice; pick true over 3 matched distractors; chance 0.25; calibrated",
               "n": len(kept), "items": kept}, open(BANK, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    from collections import Counter
    print("LOCKED", len(kept), "of", len(cand), "| by cluster:", dict(Counter(x["topic"] for x in kept)),
          "| era:", dict(Counter(x["era"] for x in kept)), "| screen cost $%.3f" % cost)
