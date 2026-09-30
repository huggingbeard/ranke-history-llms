#!/usr/bin/env python3
"""Rewrite trial 2026-09-29 (JV 22:37 "rewrite on cluster. sure"): mechanical fidelity check of Talkie's rewrites of
EB11 passages (PRE-REGISTERED in LOG.md before the job ran; no model judges anything).
A rewrite PASSES if all hold:
  1. Talkie stopped cleanly (termination == "eos") and the rewrite has >= 40 words;
  2. every number in the rewrite (years, dates, counts; digits) also appears in the source passage;
  3. >= 80% of the capitalised words in the rewrite (not sentence-initial, not in a small list of common words,
     not "Question"/"Answer"/"Q"/"A") also appear in the source passage;
  4. length ratio rewrite/source: plain 0.5-1.6, summary 0.15-0.9, catechism 0.5-2.5, causes 0.3-2.0.
Pass bar for scaling up (pre-registered): >= 90% of rewrites pass, AND in a manual read of 20 random passing
rewrites against their source at most 2 contain an invented fact.
Output: rewrite_check.jsonl (one row per rewrite: checks + reasons), rewrite_check_report.json, and
rewrite_read_sheet.md (20 random passing rewrites next to their source, for the manual read)."""
import json, random, re, sys
from collections import Counter
from pathlib import Path

T = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"G:\detritus\ranke1929_talkie_sft_20260928\runs\night_20260929\rewrite_trial")
src = {json.loads(l)["passage_id"]: json.loads(l) for l in open(T / "rewrite_trial_sources.jsonl", encoding="utf-8")}
rows = [json.loads(l) for l in open(T / "rewrite_trial_answers.jsonl", encoding="utf-8") if l.strip()]
NUM = re.compile(r"\d+")
COMMON = {"The", "This", "That", "These", "Those", "He", "She", "It", "They", "His", "Her", "Its", "Their", "In", "On",
          "At", "By", "For", "From", "With", "After", "Before", "During", "When", "Where", "Who", "What", "Which", "Why",
          "How", "A", "An", "And", "But", "Or", "As", "Being", "Having", "Question", "Answer", "Q", "A.", "Q.", "Mr",
          "Sir", "Lord", "King", "Queen", "Emperor", "Prince", "Duke", "Count", "General", "Consequences", "Causes",
          "Summary", "Yes", "No", "There", "Then", "Thus", "Of", "To", "Is", "Was", "Were", "Did", "Do", "I", "We", "You"}
RATIO = {"plain": (0.5, 1.6), "summary": (0.15, 0.9), "catechism": (0.5, 2.5), "causes": (0.3, 2.0)}


def caps(text):
    out = []
    for sent in re.split(r"(?<=[.!?:;])\s+|\n+", text):
        w = re.findall(r"[A-Za-zÀ-ÿ'’\-]+", sent)
        out += [x for x in w[1:] if x[:1].isupper() and x not in COMMON]
    return out


res = []
for r in rows:
    pid, style = r["candidate_id"].rsplit("::", 1)
    s = src[pid]["passage"]
    a = r["answer"]
    why = []
    if r.get("termination") != "eos":
        why.append("not_stopped_cleanly")
    if len(a.split()) < 40:
        why.append("under_40_words")
    new_nums = sorted(set(NUM.findall(a)) - set(NUM.findall(s)))
    if new_nums:
        why.append("number_not_in_source:" + ",".join(new_nums[:5]))
    c = caps(a)
    src_words = set(re.findall(r"[A-Za-zÀ-ÿ'’\-]+", s))
    share = (sum(1 for x in c if x in src_words) / len(c)) if c else 1.0
    if share < 0.8:
        why.append(f"names_not_in_source:{share:.2f}:" + ",".join(sorted({x for x in c if x not in src_words})[:6]))
    ratio = len(a.split()) / max(len(s.split()), 1)
    lo, hi = RATIO[style]
    if not lo <= ratio <= hi:
        why.append(f"length_ratio:{ratio:.2f}")
    res.append({"candidate_id": r["candidate_id"], "passage_id": pid, "style": style, "pass": not why, "reasons": why,
                "name_share": round(share, 3), "length_ratio": round(ratio, 3), "rewrite": a})
(T / "rewrite_check.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in res), encoding="utf-8")
rep = {"rewrites": len(res), "passed": sum(x["pass"] for x in res),
       "pass_rate": round(sum(x["pass"] for x in res) / max(len(res), 1), 3),
       "pass_rate_by_style": {st: round(sum(x["pass"] for x in res if x["style"] == st) /
                                        max(sum(1 for x in res if x["style"] == st), 1), 3) for st in RATIO},
       "reasons": Counter(w.split(":")[0] for x in res for w in x["reasons"])}
(T / "rewrite_check_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
random.seed(7)
passing = [x for x in res if x["pass"]]
sheet = ["# Rewrite trial - manual read sheet (20 random passing rewrites)", ""]
for k, x in enumerate(random.sample(passing, min(20, len(passing))), 1):
    sheet += [f"## {k}. {src[x['passage_id']]['headword']} - style: {x['style']}", "", "**Source (EB11):**", "",
              src[x["passage_id"]]["passage"], "", "**Talkie rewrite:**", "", x["rewrite"], "",
              "Invented fact? [ ] no  [ ] yes: ...", ""]
(T / "rewrite_read_sheet.md").write_text("\n".join(sheet), encoding="utf-8")
print(json.dumps(rep, indent=1))
