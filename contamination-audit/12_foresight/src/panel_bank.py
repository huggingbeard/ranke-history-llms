# -*- coding: utf-8 -*-
"""Screen and FREEZE the panel bank (see PREREGISTRATION_FORESIGHT_v2.md).

Static screen (CPU):
  * structure: 12 items/event, 6 T/F pairs/event, 6 gate probes/event, all in en/de/fr;
  * leakage: no content word of the target appears in the subject (stopword-filtered);
  * length parity: per item, max/min character length across languages <= 2.0;
  * gates: correct_idx in range, options pairwise distinct.
Period-vocabulary screen (GPU, one 1913 load):
  * per-language robust outliers (median + 2.5*MAD of per-token stem NLL under the 1913 model) are
    FLAGGED for review in the screen report (not auto-dropped): a stem the 1913 mind itself finds
    bizarre is suspect of anachronism.
Freeze:
  * writes results/question_bank_panel.json with a SHA-256 over the canonical dump; scorers refuse
    to run if the hash of the live bank differs.

Run: python src/panel_bank.py [--static-only]
"""
import os, sys, json, hashlib, re, statistics as st
import common
from panel_events import EVENTS_META, ITEMS, TIER_B, TIER_C, LANGS
from panel_checks import TRUEFALSE, GATES

STOP = set("""the a an of in to for and or will would be is are was were it its his her their on at
by with from into upon when should shall one day next great der die das den dem des ein eine einer
eines einem einen und oder wird werden wurde würde ist sind war waren im am an auf zu für von mit
aus bei le la les un une des du de au aux et ou sera seront est sont était étaient dans sur pour
par avec en qui que quand si se sa son ses leur leurs nicht pas ne plus tout toute tous toutes
alle jeder jede jedes""".split())


def content_words(s):
    return [w for w in re.findall(r"[a-zA-ZäöüßÄÖÜéèêàâçîïôûùœ']+", s.lower()) if w not in STOP and len(w) > 3]


problems, flags = [], []


def check_structure():
    for ev, meta in EVENTS_META.items():
        its = ITEMS.get(ev, [])
        if len(its) != 12:
            problems.append(f"{ev}: {len(its)} items (want 12)")
        if len(TRUEFALSE.get(ev, [])) != 6:
            problems.append(f"{ev}: {len(TRUEFALSE.get(ev, []))} T/F pairs (want 6)")
        if len(GATES.get(ev, [])) != 6:
            problems.append(f"{ev}: {len(GATES.get(ev, []))} gate probes (want 6)")
        for i, it in enumerate(its):
            for lg in LANGS:
                if lg not in it or len(it[lg]) != 3:
                    problems.append(f"{ev}#{i}: bad {lg} tuple")
            # leakage: target content words in subject
            for lg in LANGS:
                subj, tgt, neu = it[lg]
                leak = set(content_words(tgt)) & set(content_words(subj))
                if leak:
                    flags.append(f"LEAK? {ev}#{i}/{lg}: {sorted(leak)}")
            # length parity across langs
            lens = [len(it[lg][0] + it[lg][1]) for lg in LANGS]
            if max(lens) / max(1, min(lens)) > 2.0:
                problems.append(f"{ev}#{i}: length disparity {lens}")
        for i, tf in enumerate(TRUEFALSE.get(ev, [])):
            for lg in LANGS:
                if lg not in tf or len(tf[lg]) != 4:
                    problems.append(f"TF {ev}#{i}: bad {lg} tuple")
                elif tf[lg][1] == tf[lg][2]:
                    problems.append(f"TF {ev}#{i}/{lg}: true==false")
        for i, g in enumerate(GATES.get(ev, [])):
            for lg in LANGS:
                if lg not in g or len(g[lg]) != 4:
                    problems.append(f"GATE {ev}#{i}: bad {lg} tuple")
                else:
                    stem, opts, ci, neu = g[lg]
                    if not (0 <= ci < len(opts)) or len(set(opts)) != len(opts):
                        problems.append(f"GATE {ev}#{i}/{lg}: bad options/correct_idx")
    for name, ctrl in [("tierB", TIER_B), ("tierC", TIER_C)]:
        for i, it in enumerate(ctrl):
            for lg in LANGS:
                if lg not in it or len(it[lg]) != 3:
                    problems.append(f"{name}#{i}: bad {lg} tuple")


def period_vocab_screen():
    """Per-token NLL of every stem under the 1913 model; robust outliers flagged per language."""
    from foresight_bank import RANKE
    tok, model = common.load(RANKE["1913"])
    stems = []  # (key, lang, text)
    for ev, its in ITEMS.items():
        for i, it in enumerate(its):
            for lg in LANGS:
                stems.append((f"{ev}#{i}", lg, it[lg][0]))
    for ev, tfs in TRUEFALSE.items():
        for i, tf in enumerate(tfs):
            for lg in LANGS:
                stems.append((f"TF:{ev}#{i}", lg, tf[lg][0]))
    out = []
    for key, lg, text in stems:
        n = common.toklen(tok, text)
        nll = -common.slp(model, tok, "\n", text) / max(1, n)
        out.append((key, lg, round(nll, 3)))
    common.free(model)
    by_lang = {}
    for key, lg, v in out:
        by_lang.setdefault(lg, []).append(v)
    for lg, vals in by_lang.items():
        med = st.median(vals)
        mad = st.median([abs(v - med) for v in vals]) or 0.3
        thr = med + 2.5 * mad
        for key, l2, v in out:
            if l2 == lg and v > thr:
                flags.append(f"PERIOD? {key}/{lg}: stem NLL/token {v} > {round(thr,2)} (median {round(med,2)})")
    return {f"{k}/{lg}": v for k, lg, v in out}


def canonical_bank():
    return {
        "version": "panel-v2", "frozen": None, "langs": list(LANGS),
        "events_meta": EVENTS_META, "items": ITEMS,
        "tierB": TIER_B, "tierC": TIER_C, "truefalse": TRUEFALSE, "gates": GATES,
        "counts": {"event_items": sum(len(v) for v in ITEMS.values()),
                   "tf_pairs": sum(len(v) for v in TRUEFALSE.values()),
                   "gate_probes": sum(len(v) for v in GATES.values()),
                   "tierB": len(TIER_B), "tierC": len(TIER_C)},
    }


def bank_hash(bank):
    d = {k: v for k, v in bank.items() if k != "frozen"}
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_frozen():
    p = os.path.join(common.RESULTS, "question_bank_panel.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def assert_frozen_matches():
    """Called by every scorer: the live bank must hash-match the frozen bank."""
    frozen = load_frozen()
    if not frozen:
        raise RuntimeError("panel bank not frozen — run: python src/panel_bank.py")
    live = bank_hash(canonical_bank())
    if frozen.get("sha256") != live:
        raise RuntimeError("live panel bank differs from the frozen bank — re-freeze deliberately first")
    return frozen


if __name__ == "__main__":
    import datetime
    check_structure()
    print(f"structure: {len(problems)} problems, {len(flags)} flags")
    for p in problems:
        print("  PROBLEM:", p)
    stem_nll = {}
    if "--static-only" not in sys.argv and not problems:
        print("period-vocabulary screen under the 1913 model ...", flush=True)
        stem_nll = period_vocab_screen()
    for f in flags:
        print("  FLAG:", f)
    report = {"problems": problems, "flags": flags, "stem_nll_per_token": stem_nll}
    json.dump(report, open(os.path.join(common.RESULTS, "panel_screen_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if problems:
        print("NOT FROZEN — fix problems first.")
        sys.exit(1)
    bank = canonical_bank()
    bank["sha256"] = bank_hash(bank)
    bank["frozen"] = datetime.datetime.now().isoformat(timespec="seconds")
    json.dump(bank, open(os.path.join(common.RESULTS, "question_bank_panel.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"FROZEN: results/question_bank_panel.json  sha256={bank['sha256'][:16]}...  "
          f"({bank['counts']})")
