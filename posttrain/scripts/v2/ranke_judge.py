#!/usr/bin/env python3
"""Stage B (plan v2.3): Ranke-1929 judges Talkie records. Numbers only, no text.

grounding_gain = mean log p(answer | passage + question) - mean log p(answer | question)
fluency        = mean log p(answer | question)

Modes:
  filter   : v1 mechanical rules (validate_talkie_pairs.py, unchanged) + eval-overlap Jaccard rule
  score    : Ranke scores for every mechanically accepted record (resume-safe, one row per record)
  validate : plan test V3 (corrupted-sentence answers; random-passage control)
  select   : apply the pre-fixed thresholds (gain > P20, fluency > P5) -> accepted file
"""
from __future__ import annotations

import argparse, hashlib, json, random, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


LATE_TERMS = ["great war", "world war", "league of nations", "versailles", "bolshevik", "soviet", "lenin",
              "trotsky", "stalin", "mussolini", "fascis", "weimar", "armistice", "brest-litovsk", "lusitania",
              "somme", "verdun", "gallipoli", "sarajevo", "wilson's fourteen", "locarno", "kellogg", "hitler",
              "irish free state", "balfour declaration", "sykes-picot", "suffrage granted in 1918",
              "wireless broadcast", "tank warfare", "trench warfare", "poison gas", "u-boat", "marne", "ypres",
              "no man's land", "western front", "eastern front", "passchendaele", "jutland", "tannenberg",
              "caporetto", "dardanelles campaign", "zimmermann", "kerensky", "petrograd", "leningrad", "red army",
              "white army", "comintern", "spartacist", "kapp putsch", "ruhr occupation", "dawes plan", "young plan",
              "reparations commission", "treaty of trianon", "treaty of sevres", "treaty of lausanne", "ataturk",
              "kemal", "czechoslovakia", "yugoslavia", "kingdom of serbs", "mandate territory", "volstead",
              "harding", "coolidge", "hoover", "lloyd george ministry", "ramsay macdonald", "labour government",
              "easter rising", "black and tans", "anschluss", "aeroplane bombing", "air raid", "zeppelin raid",
              "talking picture"]
LATE_YEAR = re.compile(r"(?<!\d)19(1[4-9]|2\d|3\d)(?!\d)")
INJECT = [
    "This arrangement was swept away by the Treaty of Versailles in 1919.",
    "The League of Nations later took up the same question after 1920.",
    "Its consequences were felt most sharply in the Great War of 1914.",
    "Much of this was overturned by the Bolshevik revolution of 1917.",
    "The matter was finally settled only at the Locarno conference in 1925.",
    "Historians writing after the Armistice of 1918 judged it differently.",
    "The same pattern returned under the Weimar Republic in the 1920s.",
    "It foreshadowed the disaster at Gallipoli in 1915.",
    "Lenin would later cite this episode as a lesson for revolutionaries.",
    "Mussolini's government revived the same policy in 1922.",
    "The question was reopened by the Fourteen Points of President Wilson in 1918.",
    "The Soviet state inherited these difficulties after 1922.",
]


_UNITS = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
SPELLED_YEARS = (["nineteen " + w for w in ["fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]]
                 + ["nineteen " + d + ("-" + u if u else "") for d in ["twenty", "thirty"] for u in _UNITS])


def late_hits(text):
    low = text.lower()
    hits = [w for w in LATE_TERMS if re.search(r"(?<![a-z])" + re.escape(w), low)]  # word start; stems like "fascis" allowed
    hits += ["spelled:" + w for w in SPELLED_YEARS if w in low.replace("nineteen hundred and ", "nineteen ")]
    hits += ["year:" + m.group(0) for m in LATE_YEAR.finditer(text)]
    return hits


def wset(t):
    return set(re.findall(r"[a-z]+", t.lower()))


def do_filter(a):
    rep = Path(a.out_dir) / "mechanical_report.json"
    mech = Path(a.out_dir) / "mechanical_accepted.jsonl"
    subprocess.run([sys.executable, str(HERE.parent / "validate_talkie_pairs.py"), "--generated", a.raw,
                    "--sources", a.sources, "--accepted", str(mech), "--report", str(rep),
                    "--min-answer-words", str(a.min_answer_words)], check=True)
    ev = jl(a.eval)
    evalq = [wset(r["prompt"]) for r in ev]
    evala = [wset(r["prompt"] + " " + " ".join(r.get("criteria", []))) for r in ev]
    kept, rows = [], []
    for r in json.loads(rep.read_text(encoding="utf-8"))["records"]:
        q = wset(r.get("question") or "")
        jac = max((len(q & e) / len(q | e) if q | e else 0.0) for e in evalq)
        aw = wset(r.get("answer") or "")
        jac_a = max((len(aw & e) / len(aw | e) if aw | e else 0.0) for e in evala)
        hits = late_hits((r.get("question") or "") + " " + (r.get("answer") or ""))
        r["max_eval_jaccard"] = round(jac, 4)
        r["max_eval_answer_criteria_jaccard"] = round(jac_a, 4)
        r["late_fact_hits"] = hits
        extra = []
        if jac > 0.5:
            extra.append("eval_overlap_jaccard")
        if jac_a > 0.5:
            extra.append("eval_answer_criteria_overlap")
        if hits:
            extra.append("late_fact_terms")
        if extra:
            r["mechanical_accept"] = False
            r["mechanical_reasons"] = r["mechanical_reasons"] + extra
        rows.append(r)
        if r["mechanical_accept"]:
            kept.append(r)
    with open(Path(a.out_dir) / "mechanical.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(mech, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(x for r in rows for x in r["mechanical_reasons"])
    print(json.dumps({"stage": "filter", "raw": len(rows), "kept": len(kept), "reasons": c}), flush=True)


class Scorer:
    def __init__(self, base):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
        self.truncated = 0  # contexts longer than max_ctx tokens (reported)
        self.model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, dtype=torch.bfloat16,
                                                          device_map={"": 0}, attn_implementation="sdpa").eval()

    def mean_lp(self, pairs, bs=4, max_ctx=3000):
        """pairs: list of (context_text, answer_text). Returns mean per-token log-prob of the answer."""
        torch = self.torch
        out = []
        for s in range(0, len(pairs), bs):
            chunk = pairs[s:s + bs]
            seqs, spans = [], []
            for ctx, ans in chunk:
                c = self.tok.encode(ctx, add_special_tokens=False)
                if len(c) > max_ctx:
                    self.truncated += 1
                c = c[-max_ctx:]
                t = self.tok.encode(" " + ans.strip(), add_special_tokens=False)
                seqs.append(c + t); spans.append((len(c), len(c) + len(t)))
            L = max(map(len, seqs))
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
            ids = torch.full((len(seqs), L), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for i, q in enumerate(seqs):
                ids[i, :len(q)] = torch.tensor(q); att[i, :len(q)] = 1
            with torch.no_grad():
                logits = self.model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits
                for i, (a0, a1) in enumerate(spans):
                    # only the answer span in float32 (saves memory on small cards; same numbers)
                    lp = torch.log_softmax(logits[i, a0 - 1:a1 - 1].float(), -1)
                    tgt = ids[i, a0:a1].cuda()
                    out.append(float(lp.gather(-1, tgt[:, None])[:, 0].mean()))
                del logits
        return out


def ctx_with(passage, q):
    return f"{passage.strip()}\n\nQuestion: {q.strip()}\nAnswer:"


def ctx_without(q):
    return f"Question: {q.strip()}\nAnswer:"


def do_score(a):
    src = {r["passage_id"]: r["text"] for r in jl(a.sources)}
    recs = jl(Path(a.out_dir) / "mechanical_accepted.jsonl")
    out = Path(a.out_dir) / "ranke_grounding.jsonl"
    done = {r["candidate_id"] for r in jl(out)} if out.exists() else set()
    todo = [r for r in recs if r["candidate_id"] not in done]
    sc = Scorer(a.base)
    with open(out, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), 64):
            ch = todo[s:s + 64]
            w = sc.mean_lp([(ctx_with(src[r["passage_id"]], r["question"]), r["answer"]) for r in ch])
            wo = sc.mean_lp([(ctx_without(r["question"]), r["answer"]) for r in ch])
            for r, x, y in zip(ch, w, wo):
                f.write(json.dumps({"candidate_id": r["candidate_id"], "passage_id": r["passage_id"],
                                    "lp_with_passage": x, "lp_question_only": y,
                                    "grounding_gain": x - y, "fluency": y,
                                    "judge_model": "uzh-echist-org/Ranke-4B-1929_v2", "judge_base": a.base},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({"stage": "score", "done": s + len(ch), "todo": len(todo),
                              "contexts_truncated_so_far": sc.truncated}), flush=True)


def sentences(t):
    return [x for x in re.split(r"(?<=[.!?])\s+", t.strip()) if x]


def do_validate(a):
    rng = random.Random(1929)
    src = {r["passage_id"]: r["text"] for r in jl(a.sources)}
    recs = [r for r in jl(Path(a.out_dir) / "mechanical_accepted.jsonl") if len(sentences(r["answer"])) >= 3]
    rng.shuffle(recs)
    recs = recs[:a.n]
    sc = Scorer(a.base)
    rows = []
    for r in recs:
        others = [o for o in recs if o["source_id"] != r["source_id"]] or [o for o in recs if o is not r]
        o = rng.choice(others)
        ss = sentences(r["answer"])
        mid = len(ss) // 2
        repl = rng.choice(sentences(o["answer"]))
        corrupted = " ".join(ss[:mid] + [repl] + ss[mid + 1:])
        other_passage = src[rng.choice([p for p in src if p != r["passage_id"]])]
        rows.append({"candidate_id": r["candidate_id"], "question": r["question"], "answer": r["answer"],
                     "corrupted_answer": corrupted, "replaced_sentence": ss[mid], "inserted_sentence": repl,
                     "inserted_from": o["candidate_id"],
                     "_ctx": ctx_with(src[r["passage_id"]], r["question"]), "_ctx0": ctx_without(r["question"]),
                     "_ctxr": ctx_with(other_passage, r["question"])})
    lp_true = sc.mean_lp([(x["_ctx"], x["answer"]) for x in rows])
    lp_true0 = sc.mean_lp([(x["_ctx0"], x["answer"]) for x in rows])
    lp_cor = sc.mean_lp([(x["_ctx"], x["corrupted_answer"]) for x in rows])
    lp_cor0 = sc.mean_lp([(x["_ctx0"], x["corrupted_answer"]) for x in rows])
    lp_rand = sc.mean_lp([(x["_ctxr"], x["answer"]) for x in rows])
    for i, x in enumerate(rows):
        ss = sentences(x["answer"]); mid = len(ss) // 2
        x["injected_sentence"] = INJECT[i % len(INJECT)]
        x["injected_answer"] = " ".join(ss[:mid] + [x["injected_sentence"]] + ss[mid:])
    lp_inj = sc.mean_lp([(x["_ctx"], x["injected_answer"]) for x in rows])
    lp_inj0 = sc.mean_lp([(x["_ctx0"], x["injected_answer"]) for x in rows])
    win_c = win_r = win_i = 0
    for x, t, t0, c, c0, rp, ij, ij0 in zip(rows, lp_true, lp_true0, lp_cor, lp_cor0, lp_rand, lp_inj, lp_inj0):
        x.update(gain_intact=t - t0, gain_corrupted=c - c0, gain_random_passage=rp - t0, gain_injected=ij - ij0)
        x["clean_beats_injected"] = x["gain_intact"] > x["gain_injected"]
        win_i += x["clean_beats_injected"]
        x["intact_beats_corrupted"] = x["gain_intact"] > x["gain_corrupted"]
        x["true_beats_random_passage"] = x["gain_intact"] > x["gain_random_passage"]
        win_c += x["intact_beats_corrupted"]; win_r += x["true_beats_random_passage"]
        for k in ("_ctx", "_ctx0", "_ctxr"):
            x.pop(k)
    n = len(rows)
    res = {"test": "V3_ranke_grounding_judge", "n": n,
           "intact_beats_corrupted_rate": win_c / n, "deciding_bar": 0.80,
           "true_beats_random_passage_rate": win_r / n, "reported_expectation": 0.90,
           "clean_beats_late_fact_injected_rate": win_i / n, "V3b_bar": 0.90,
           "passed_V3": win_c / n >= 0.80, "passed_V3b": win_i / n >= 0.90,
           "V3b_informational_only": True,
           "passed": win_c / n >= 0.80}
    Path(a.out_dir, "V3_validation.json").write_text(json.dumps({**res, "rows": rows}, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print(json.dumps(res), flush=True)


def do_select(a):
    import numpy as np
    sc = {r["candidate_id"]: r for r in jl(Path(a.out_dir) / "ranke_grounding.jsonl")}
    recs = [r for r in jl(Path(a.out_dir) / "mechanical_accepted.jsonl") if r["candidate_id"] in sc]
    g = np.array([sc[r["candidate_id"]]["grounding_gain"] for r in recs])
    fl = np.array([sc[r["candidate_id"]]["fluency"] for r in recs])
    g20, f5 = float(np.percentile(g, 20)), float(np.percentile(fl, 5))
    acc = []
    with open(Path(a.out_dir) / "stageB_decisions.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            s = sc[r["candidate_id"]]
            ok = (s["grounding_gain"] > 0 and s["grounding_gain"] > g20 and s["fluency"] > f5) if not a.no_threshold else True
            # --no-threshold (JV 2026-09-29 13:06): judge failed test V3, so Ranke scores are kept for analysis only
            f.write(json.dumps({"candidate_id": r["candidate_id"], "grounding_gain": s["grounding_gain"],
                                "fluency": s["fluency"], "gain_p20": g20, "fluency_p5": f5, "accept": ok}) + "\n")
            if ok:
                acc.append({**r, "grounding_gain": s["grounding_gain"], "fluency": s["fluency"]})
    with open(Path(a.out_dir) / "stageB_accepted.jsonl", "w", encoding="utf-8") as f:
        for r in acc:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"stage": "select", "scored": len(recs), "accepted": len(acc), "gain_p20": g20,
                      "share_gain_le_0": float((g <= 0).mean()),
                      "fluency_p5": f5}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["filter", "score", "validate", "select"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--raw")
    ap.add_argument("--sources", required=True)
    ap.add_argument("--eval")
    ap.add_argument("--base")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--no-threshold", action="store_true")
    ap.add_argument("--min-answer-words", type=int, default=20)  # JV 2026-09-29 07:36, option B
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    {"filter": do_filter, "score": do_score, "validate": do_validate, "select": do_select}[a.mode](a)


if __name__ == "__main__":
    main()
