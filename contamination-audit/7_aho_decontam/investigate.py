"""
investigate.py — three follow-ups on the keyword-list diagnostics:
  (1) POSITIVE CONTROL: do the kept markers actually fire on real post-1945 books? (recall)
  (2) READ THE TEXT: why do holocaust / cia / nagasaki / nasa fire in genuinely-old books? (snippets)
  (3) CONTEXT SEPARATION: would a +-window context test separate the OLD sense (reject) from the
      POST-1945 sense (keep)? For each term we measure the share of OLD fires vs MODERN fires that
      have the post-1945 context nearby. Clean separation = a context guard salvages/fixes the term.
"""
import os, re, sys
from collections import defaultdict
from pathlib import Path
import ahocorasick, aho_scan as A

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
N_OLD, N_MOD, CAP, WIN = 15000, 300, 20000, 70

# term -> (what the post-1945 sense is, context regex that marks it)
INTEREST = {
 "nasa":      ("the space agency",   r"space|rocket|launch|astronaut|apollo|orbit|satellite|aeronaut|spacecraft"),
 "nagasaki":  ("the 1945 A-bombing", r"atomic|\bbomb|nuclear|1945|hiroshima|surrender|war"),
 "hiroshima": ("the 1945 A-bombing", r"atomic|\bbomb|nuclear|1945|nagasaki|surrender|war"),
 "holocaust": ("the Nazi genocide",  r"jew|nazi|genocid|hitler|six million|exterminat|auschwitz|concentration camp"),
 "cia":       ("the US spy agency",  r"intelligence|\bspy|espionage|soviet|langley|\bagent|covert|central intelligence"),
 "detente":   ("the Cold War thaw",  r"cold war|soviet|superpower|moscow|nixon|\barms|east-west|nuclear"),
}
AUTO = ahocorasick.Automaton()
for t in INTEREST: AUTO.add_word(t, t)
AUTO.make_automaton()
CTX = {t: re.compile(rx, re.I) for t, (_, rx) in INTEREST.items()}


def fires(norm):
    out = []
    for end, term in AUTO.iter(norm):
        s = end - len(term) + 1
        b = norm[s-1] if s > 0 else " "; a = norm[end+1] if end+1 < len(norm) else " "
        if not (b.isalnum() or a.isalnum()):
            has_ctx = bool(CTX[term].search(norm[max(0, s-WIN):end+WIN]))
            out.append((term, s, has_ctx))
    return out


def main():
    tok = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read_text().strip()
    os.environ["HF_HUB_DISABLE_SYMLINKS"]="1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
    os.environ["HF_TOKEN"]=tok
    from datasets import load_dataset
    kept, contexts = A.load_markers(); autoK = A.build_automaton(kept)

    ds = load_dataset("storytracer/LoC-PD-Books", split="train", streaming=True, token=tok)
    old_ctx = {t: [0, 0] for t in INTEREST}   # [without_ctx, with_ctx]
    mod_ctx = {t: [0, 0] for t in INTEREST}
    snips = defaultdict(list)
    n_old = n_mod = mod_flagged = 0
    kept_fire_mod = defaultdict(int)
    for i, rec in enumerate(ds):
        if i >= CAP or (n_old >= N_OLD and n_mod >= N_MOD): break
        yr, txt = rec.get("year"), A.doc_text(rec)
        if not isinstance(yr, int) or not txt: continue
        norm = A.normalize(txt)
        if yr <= 1929 and n_old < N_OLD:
            n_old += 1
            for term, s, hc in fires(norm):
                old_ctx[term][1 if hc else 0] += 1
                if not hc and len(snips[term]) < 4:
                    snips[term].append(norm[max(0, s-45):s+45].strip())
        elif yr >= 1946 and n_mod < N_MOD:
            n_mod += 1
            if A.scan_text(norm, autoK, contexts): mod_flagged += 1
            for m in {mm for mm, _ in A.scan_text(norm, autoK, contexts)}: kept_fire_mod[m] += 1
            for term, s, hc in fires(norm):
                mod_ctx[term][1 if hc else 0] += 1
        if i % 4000 == 0: print(f"  streamed {i}: old={n_old} mod={n_mod}", file=sys.stderr, flush=True)

    print(f"\n{'='*72}\n(1) POSITIVE CONTROL — do the kept markers catch real post-1945 books?\n{'='*72}")
    print(f"  {mod_flagged}/{n_mod} modern books flagged by the vocabulary list = recall {100*mod_flagged/max(n_mod,1):.0f}%")
    top = sorted(kept_fire_mod.items(), key=lambda x: -x[1])[:12]
    print("  markers firing most on modern books:", ", ".join(f"{m}({c})" for m, c in top))

    print(f"\n{'='*72}\n(2) WHY THEY FIRE IN OLD BOOKS — real snippets (no post-1945 context)\n{'='*72}")
    for t in INTEREST:
        for sn in snips[t][:3]:
            print(f"  [{t}] ...{sn}...")

    print(f"\n{'='*72}\n(3) CONTEXT SEPARATION — share of fires WITH the post-1945 context nearby\n{'='*72}")
    print(f"{'term':<12}{'post-1945 sense':<20}{'OLD fires (n)':>14}{'% w/ ctx':>10}{'MODERN fires (n)':>18}{'% w/ ctx':>10}")
    for t, (sense, _) in INTEREST.items():
        o0, o1 = old_ctx[t]; m0, m1 = mod_ctx[t]
        oP = 100*o1/max(o0+o1, 1); mP = 100*m1/max(m0+m1, 1)
        print(f"{t:<12}{sense:<20}{o0+o1:>14}{oP:>9.0f}%{m0+m1:>18}{mP:>9.0f}%")
    print("\nreading: if OLD %-with-context is LOW and MODERN %-with-context is HIGH, a context guard "
          "cleanly separates the two — so we can guard the leaky kept term (e.g. nasa) and/or re-admit "
          "the rejected term (e.g. nagasaki) as a guarded post-1945 marker.")


if __name__ == "__main__":
    main()
