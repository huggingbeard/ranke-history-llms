"""Combination frontier: Qwen3.6-35B (BF16, A100 full slice) + the best of the other models.

v1 (2026-07-23), PI question: "if we combined qwen 35 with the best the other models have to
offer where can we end up in terms of recall and fp?"

Search space (all computed from stored per-passage verdicts, zero GPU):
  members: 35B + subsets of {gpt-oss-20b, Qwen2.5-14B-AWQ, Qwen2.5-7B, Phi-3.5-mini}
  book-level rules: each member flags via its own k-of-8 (35B k in 1..6, others k in 1..3);
                    combine by AND (all flag), OR (any flags), MAJORITY (>= ceil(M/2) flag)
  passage-level rules (pairs/triples incl. 35B): passage = AFTER if >= m of M members agree,
                    then book k-of-8 (k in 1..4)
  axes: L in {1000, 2000} x strategy in {random, grep}, N = 8 (the frontier config family)
Overfitting guard: books split 50/50 (seed 1929); combos SELECTED on half A, headline numbers
REPORTED from held-out half B. Full-sample Pareto also saved for the record.
Output: outputs\\combo35_results.csv + console: held-out best at FP<=5% / <=10% / recall-max.
"""
import os, glob, json
from itertools import combinations
import random as _r
import pandas as pd
import numpy as np

DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
OUT = os.path.join(DROPBOX, "outputs")
REPO = os.path.dirname(os.path.abspath(__file__))
MEMBERS = {
    "q35": ["judg_Qwen36-35B-A3B-BF16-A100.parquet"],
    "gptoss": ["judg_openai__gpt-oss-20b.parquet", "judg_openai__gpt-oss-20b.pos3046.parquet"],
    "q14b": ["judg_Qwen__Qwen2.5-14B-Instruct-AWQ.parquet", "judg_Qwen__Qwen2.5-14B-Instruct-AWQ.pos3046.parquet"],
    "q7b": ["judg_Qwen__Qwen2.5-7B-Instruct.parquet", "judg_Qwen__Qwen2.5-7B-Instruct.pos3046.parquet"],
    "phi": ["judg_microsoft__Phi-3.5-mini-instruct.parquet", "judg_microsoft__Phi-3.5-mini-instruct.pos3046.parquet"],
}
K35 = range(1, 7); KOTH = range(1, 4)

def strata():
    neg = set(pd.read_csv(os.path.join(REPO, "manifest_negatives.csv"), dtype={"id": str}).id)
    mc = pd.read_csv(os.path.join(REPO, "manifest_contamC.csv"), dtype={"id": str})
    cmod = set(mc[mc.cls == "contam_modern"].id)
    b = set(pd.read_csv(os.path.join(REPO, "manifest_pos3046.csv"), dtype={"id": str}).id)
    return neg, cmod, b

def load():
    mats = {}
    for tag, files in MEMBERS.items():
        d = pd.concat([pd.read_parquet(os.path.join(OUT, f)) for f in files], ignore_index=True)
        d = d.drop_duplicates(["id", "length", "strategy", "idx"])
        d = d[(d.length.isin([1000, 2000])) & (d.idx < 8)]
        mats[tag] = d.set_index(["id", "length", "strategy", "idx"]).pred
    base = pd.concat(mats, axis=1, join="inner").reset_index()
    base.columns = ["id", "length", "strategy", "idx"] + list(MEMBERS)
    print(f"aligned matrix: {len(base)} passages, {base.id.nunique()} books", flush=True)
    return base

def eval_flags(flag_by_id, neg, cmod, b, ids):
    f = flag_by_id.loc[flag_by_id.index.isin(ids)]
    def rate(s):
        idx = f.index.isin(s)
        return float(f[idx].mean()) if idx.sum() else np.nan
    return rate(cmod), rate(b), rate(neg)

def main():
    neg, cmod, b = strata()
    base = load()
    ids = sorted(base.id.unique())
    rng = _r.Random(1929); rng.shuffle(ids)
    half_a, half_b = set(ids[: len(ids) // 2]), set(ids[len(ids) // 2:])
    rows = []
    for L in (1000, 2000):
        for strat in ("random", "grep"):
            sub = base[(base.length == L) & (base.strategy == strat)]
            votes = sub.groupby("id")[list(MEMBERS)].sum()          # per-book AFTER counts (of 8)
            # book-level combos
            for r in range(0, 4):
                for others in combinations([m for m in MEMBERS if m != "q35"], r):
                    mem = ("q35",) + others
                    kgrids = [K35] + [KOTH] * len(others)
                    def rec(idx, ks):
                        if idx == len(mem):
                            flags = pd.DataFrame({m: (votes[m] >= k).astype(int) for m, k in zip(mem, ks)})
                            for rule, fl in (("AND", flags.min(axis=1)), ("OR", flags.max(axis=1)),
                                             ("MAJ", (flags.sum(axis=1) >= -(-len(mem) // 2)).astype(int))):
                                if rule == "MAJ" and len(mem) < 3: continue
                                for split, name in ((half_a, "A"), (half_b, "B"), (None, "full")):
                                    sids = split if split is not None else set(ids)
                                    rc, rb, fp = eval_flags(fl, neg, cmod, b, sids)
                                    rows.append(dict(level="book", members="+".join(mem),
                                                     ks="/".join(map(str, ks)), rule=rule, length=L,
                                                     strategy=strat, split=name, recall_C=rc,
                                                     recall_B=rb, fp=fp))
                            return
                        for k in kgrids[idx]: rec(idx + 1, ks + [k])
                    rec(0, [])
            # passage-level fusion (pairs + triples incl. 35B), m-of-M then k-of-8
            for r in (1, 2):
                for others in combinations(["gptoss", "q14b", "q7b"], r):
                    mem = ("q35",) + others; M = len(mem)
                    pv = sub[list(mem)].ge(1).astype(int) if False else (sub[list(mem)] == 1).astype(int)
                    for m_req in range(1, M + 1):
                        pf = (pv.sum(axis=1) >= m_req).astype(int)
                        bv = pf.groupby(sub.id).sum()
                        for k in range(1, 5):
                            fl = (bv >= k).astype(int)
                            for split, name in ((half_a, "A"), (half_b, "B"), (None, "full")):
                                sids = split if split is not None else set(ids)
                                rc, rb, fp = eval_flags(fl, neg, cmod, b, sids)
                                rows.append(dict(level="passage", members="+".join(mem),
                                                 ks=f"m{m_req}ofM{M}_k{k}", rule=f"{m_req}of{M}",
                                                 length=L, strategy=strat, split=name,
                                                 recall_C=rc, recall_B=rb, fp=fp))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "combo35_results.csv"), index=False)
    n_combos = len(res[res.split == "full"])
    print(f"{n_combos} combinations evaluated -> combo35_results.csv", flush=True)

    # honest selection: pick on half A, report held-out half B
    a = res[res.split == "A"].reset_index(drop=True)
    bb = res[res.split == "B"].set_index(["level", "members", "ks", "rule", "length", "strategy"])
    key = ["level", "members", "ks", "rule", "length", "strategy"]
    a = a.assign(worst=a[["recall_C", "recall_B"]].min(axis=1))
    for fpmax in (0.05, 0.10, 0.15):
        top = a[a.fp <= fpmax].sort_values("worst", ascending=False).head(120)
        printed = 0
        print(f"\n=== selected on half A at FP<={fpmax:.0%}, REPORTED on held-out half B:")
        seen = set()
        for _, r in top.iterrows():
            sig = (r.level, r.members, r.rule)
            if sig in seen: continue
            seen.add(sig)
            try:
                h = bb.loc[tuple(r[k] for k in key)]
            except KeyError:
                continue
            print(f"  {r.members:22s} {r.rule:5s} ks={r.ks:12s} L={int(r.length)} {r.strategy:6s} | "
                  f"held-out: C={h.recall_C:.3f} B={h.recall_B:.3f} FP={h.fp:.3f}")
            printed += 1
            if printed >= 6: break

if __name__ == "__main__":
    main()
