#!/usr/bin/env python3
"""Night run 2026-09-29 (JV 18:54: "use encyclopedias to create question and answer pairs that it should know").
Deterministic question TEMPLATES from Encyclopaedia Britannica 11th ed. (1910-11) entry headwords; no model writes
the questions. Talkie-13B later answers each question directly (without the entry), like the book questions; the
entry's opening text is kept with each question so the answers can be checked mechanically against it.

Entries used (paragraphs that open with an upper-case EB11 headword):
  person  "SURNAME, GIVEN NAMES[, title] (YYYY-YYYY), descriptor"  with a historical descriptor (king, statesman ...)
           -> "Who was <Given Surname>, and for what is he or she chiefly remembered?"
  event   headwords naming a treaty / peace / battle / war / siege / congress / council / diet / edict / revolution
           -> "What was the <event>, and what were its consequences?"
Filters: death year <= 1913; JV event rule (EVENT_TERMS + years 1914-1939, qa_v2_restore.py); no word-set Jaccard
> 0.5 with any of the 84 eval prompts; one question per headword. Output: runs/night_20260929/data/
encyclopedia_questions.jsonl (candidate_id, question, template, headword, entry_text first 1,500 chars, source)."""
import json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
sys.path.insert(0, str(ROOT / "scripts" / "v2"))
from ranke_judge import wset  # noqa: E402
# JV 16:16 event rule, copied verbatim from sources_v3/qa/qa_v2_restore.py (importing that file re-runs it and
# rewrites qa_pairs_v2_uncorrected.jsonl, so it is not imported).
from ranke_judge import LATE_YEAR, SPELLED_YEARS  # noqa: E402
EVENT_TERMS = ["russo-japanese", "russo japanese", "port arthur", "tsushima", "mukden", "liaoyang", "shaho",
               "treaty of portsmouth", "norman angell", "great illusion"]


def blocked(text):
    low = text.lower()
    ev = [t for t in EVENT_TERMS if t in low]
    yrs = [m.group(0) for m in LATE_YEAR.finditer(text)] +           [w for w in SPELLED_YEARS if w in low.replace("nineteen hundred and ", "nineteen ")]
    return ev + yrs


SRC = ROOT / "sources_v3" / "cpt" / "cpt_train_v2.jsonl"
OUT = ROOT / "runs" / "night_20260929" / "data"
ev = [wset(json.loads(l)["prompt"]) for l in open(ROOT / "eval" / "frozen_eval_84.jsonl", encoding="utf-8")]

HIST = re.compile(r"\b(king|queen|emperor|empress|statesman|general|marshal|admiral|pope|prince|princess|duke|"
                  r"minister|politician|chancellor|president|historian|reformer|revolutionary|soldier|elector|"
                  r"archbishop|cardinal|sultan|tsar|czar|khan|conqueror|diplomatist|diplomat|regent|dictator|"
                  r"patriot|leader|chief|founder|lawgiver|tyrant|consul|senator|tribune|agitator)\b", re.I)
PERSON = re.compile(r"^([A-ZÀ-Þ][A-ZÀ-Þ'\-]+(?: [A-ZÀ-Þ][A-ZÀ-Þ'\-]+)?), ((?:[A-ZÀ-Þ][A-Za-zÀ-þ'\-\.]*\s?){1,6})"
                    r"(?:,[^(]{0,60})?\((?:c\. )?(\d{3,4})[-–](\d{3,4})\),? ([^.;]{4,140})")
EVENT = re.compile(r"^([A-ZÀ-Þ][A-ZÀ-Þ'\- ]{1,40}?), (TREATY|PEACE|BATTLE|SIEGE|CONGRESS|COUNCIL|DIET|EDICT|CONFERENCE|"
                   r"CONVENTION|WAR|WARS) OF( THE)?((?: [A-ZÀ-Þ][A-ZÀ-Þ'\-]*){0,4})(?=[,( ])")
LOWER = {"von", "van", "de", "der", "den", "la", "le", "du", "di", "da", "of", "and", "zu", "d'"}


def tc(s):
    out = []
    for w in s.strip().split():
        lw = w.lower()
        out.append(lw if lw in LOWER else "-".join(p[:1].upper() + p[1:].lower() for p in w.split("-")))
    return " ".join(out)


def event_name(h):
    h = h.strip(" ,")
    m = re.match(r"(.+?),\s*(TREATY|PEACE|BATTLE|SIEGE|CONGRESS|COUNCIL|DIET|EDICT|CONFERENCE|LEAGUE|CONVENTION)"
                 r" OF\b(.*)$", h)
    if m:                                   # "UTRECHT, PEACE OF" -> "Peace of Utrecht"
        return f"{tc(m.group(2))} of {tc(m.group(1))}{tc(m.group(3))}".strip()
    return tc(h)


rows, seen, why = [], set(), Counter()
for line in open(SRC, encoding="utf-8"):
    r = json.loads(line)
    if not r["source_id"].startswith("eb11"):
        continue
    for para in r["text"].split("\n\n"):
        para = para.strip()
        if not para[:1].isupper() or not para[:3].isupper():
            continue
        m = PERSON.match(para)
        if m:
            sur, given, b, d, desc = m.groups()
            desc = re.split(r",? (?:was|who|is|born|son|daughter|the son|the daughter)", desc)[0]
            if int(d) > 1913 or not HIST.search(desc):
                why["person_not_historical_or_late"] += 1
                continue
            name = f"{tc(given)} {tc(sur)}"
            she = re.search(r"(queen|empress|princess|duchess|abbess|electress|regent and queen)", desc, re.I)
            q = f"Who was {name}, and for what is {'she' if she else 'he'} chiefly remembered?"
            kind, key = "person", f"P:{sur}:{given}:{b}"
        else:
            m = EVENT.match(para)
            if not m:
                continue
            place, kw, the, rest = m.groups()
            rest = rest.strip()
            if rest and not rest.endswith(" OF"):
                why["event_heading_unclear"] += 1
                continue
            name = f"{tc(kw)} of {'the ' if the else ''}{(tc(rest) + ' ') if rest else ''}{tc(place)}"
            if len(name.split()) > 8:
                why["event_name_too_long"] += 1
                continue
            q = f"What was the {name}, and what were its consequences?"
            kind, key = "event", "E:" + name.lower()
        if key in seen:
            why["duplicate_headword"] += 1
            continue
        seen.add(key)
        entry = para[:1500]
        if blocked(q + " " + entry):
            why["jv_event_or_1914_1939_rule"] += 1
            continue
        qs = wset(q)
        if max(len(qs & e) / len(qs | e) for e in ev) > 0.5:
            why["eval_overlap"] += 1
            continue
        rows.append({"candidate_id": f"eb11::{kind}::{len(rows):05d}", "question": q, "template": kind,
                     "headword": para[:120].split(",")[0], "entry_text": entry, "source_id": r["source_id"],
                     "title": r["title"], "year": 1911, "category": "history"})
f = OUT / "encyclopedia_questions.jsonl"
f.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
print(json.dumps({"kept": len(rows), "by_template": Counter(x["template"] for x in rows), "dropped": why}, indent=1))
