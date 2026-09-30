#!/usr/bin/env python3
"""Night watcher 2026-09-29/30 (JV 19:38: "watch the cluster job ... we don't want to waste the night run").
Runs every 5 minutes (Windows task "Ranke-Night-Watch"); every step is idempotent, state in orchestrator_state.json,
timestamped log in orchestrator.log (runs/night_20260929/). All compute goes through the router.
  A. Continued-pretraining model copied home and checksum OK -> queue arm 2 (book Q&A on that model + its raw answers).
  B. Talkie cluster job (router 70ad1a1eb3 / Slurm 6697045): log every state change; 10+ min after start log the
     answer rate and projected end.
  C. Talkie job ended -> fetch its answers, run build_talkie_direct_set.py (pre-registered mechanical filters),
     queue arms 3, 4, 5 on the 4090 (on the continued-pretraining model).
Never cancels or resubmits anything; a failure is logged for the morning report."""
import json, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
N = ROOT / "runs" / "night_20260929"
STATE, LOG = N / "orchestrator_state.json", N / "orchestrator.log"
CH = [r"C:\Program Files\Python313\python.exe", r"C:\Users\hvoth\cluster\router\ch.py"]
VENV = r"G:\detritus\ranke_lora_smoke_20260927\.venv\Scripts\python.exe"
ARM = str(ROOT / "scripts" / "v3" / "night_local_arm.py")
TALKIE_ROUTER_ID = "70ad1a1eb3"
REMOTE_ANS = "/scratch/hvoth/ranke1929-talkie-sft/v2/run/out_night/talkie_direct_answers.jsonl"
N_QUESTIONS = 12072
CPT_DIR = N / "cpt_full_20260929" / "final"
CPT_REV = "cpt_full_20260929"
CPT_SHA = "721ad80000b27da18f07b52f77e34e85f194213f93573924a6e8a86e5da815dc"
ENV = {"CH_AGENT": "claude-night", "MSYS_NO_PATHCONV": "1", "PYTHONIOENCODING": "utf-8"}


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def sh(cmd, timeout=600):
    import os
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env={**os.environ, **ENV},
                       encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def submit(name, desc, est_min, rows, args):
    cmd = f'"{VENV}" "{ARM}" ' + " ".join(args)
    rc, out = sh(CH + ["submit", "--name", name, "--desc", desc, "--needs-gpu", "--gpu-mem", "16", "--est-minutes",
                       str(est_min), "--est-rows", str(rows), "--no-wait", "--cmd", cmd])
    m = re.search(r"queued as (\w{10})|(\w{10}) started at once", out)
    log(f"submit {name}: rc={rc} router_id={m and (m.group(1) or m.group(2))} :: {out.strip().splitlines()[-2:] if out.strip() else ''}")
    return rc == 0


def cpt_ok(st):
    if st.get("cpt_verified"):
        return True
    f = CPT_DIR / "model.safetensors"
    if not f.exists() or f.stat().st_size != 8044982080:
        return False
    import hashlib
    h = hashlib.sha256()
    with f.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    st["cpt_verified"] = h.hexdigest() == CPT_SHA
    log(f"continued-pretraining model checksum {'OK' if st['cpt_verified'] else 'MISMATCH ' + h.hexdigest()}")
    return st["cpt_verified"]


def main():
    N.mkdir(parents=True, exist_ok=True)
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    if datetime.now() > datetime(2026, 9, 30, 12, 0):
        return
    # A. arm 2 on the continued-pretraining model
    if not st.get("arm2_submitted") and cpt_ok(st):
        if submit("ranke-cpt-bookqa", "Ranke night arm 2: book Q&A (11,737) LoRA r64 on the continued-pretraining model + raw answers",
                  75, 35211, ["--arm", "cpt_bookqa_all_r64", "--base", str(CPT_DIR), "--base-revision", CPT_REV,
                              "--data", str(N / "data" / "bookqa_all.jsonl"), "--epochs", "3", "--eval-epochs", "1,2,3",
                              "--raw-eval", "cpt_raw"]):
            st["arm2_submitted"] = datetime.now().isoformat(timespec="seconds")
    # B. watch the Talkie job
    if not st.get("talkie_fetched"):
        rc, out = sh(CH + ["status", "--id", TALKIE_ROUTER_ID], timeout=120)
        m = re.search(r"\]: (\w+)", out)
        state = m.group(1) if m else "UNKNOWN"
        if state != st.get("talkie_state"):
            log(f"Talkie cluster job state {st.get('talkie_state')} -> {state} :: {out.strip().splitlines()[0][:200] if out.strip() else ''}")
            st["talkie_state"] = state
            if state == "RUNNING":
                em = re.search(r"RUNNING\s+(?:(\d+)-)?(\d+):(\d+):(\d+)", out)   # elapsed as Slurm reports it
                el = (int(em.group(1) or 0) * 86400 + int(em.group(2)) * 3600 + int(em.group(3)) * 60 + int(em.group(4))) if em else 0
                st["talkie_started_seen"] = time.time() - el
        if state == "RUNNING":
            rc2, n = sh(["ssh", "-o", "BatchMode=yes", "hvoth@cluster.s3it.uzh.ch", f"wc -l < {REMOTE_ANS} 2>/dev/null || echo 0"], 60)
            try:
                rows = int(n.strip().splitlines()[-1])
            except (ValueError, IndexError):
                rows = -1
            mins = (time.time() - st["talkie_started_seen"]) / 60
            if mins >= 10 and not st.get("talkie_rate_logged"):
                rate = rows / max(mins - 5, 1)      # ~5 min model load
                left = (N_QUESTIONS - rows) / rate if rate > 0 else float("inf")
                log(f"Talkie speed check: {rows} answers after {mins:.0f} min (~{rate:.0f}/min), ~{left:.0f} min to go")
                st["talkie_rate_logged"] = True
            st["talkie_rows"] = rows
        if state in ("COMPLETED", "TIMEOUT", "FAILED", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "DONE"):
            dest = N / "talkie_direct"
            dest.mkdir(exist_ok=True)
            rc, out = sh(CH + ["fetch", "--remote", REMOTE_ANS, "--local", str(dest), "--overwrite"], 900)
            f = dest / "talkie_direct_answers.jsonl"
            rows = sum(1 for _ in f.open(encoding="utf-8")) if f.exists() else 0
            log(f"Talkie job ended ({state}); fetched {rows} of {N_QUESTIONS} answers (fetch rc={rc})")
            if rows < 1000:
                log("TOO FEW ANSWERS - not building training sets; needs a look in the morning")
                st["talkie_fetched"] = "too_few"
            else:
                st["talkie_fetched"] = rows
    # C. build sets and queue arms 3-5
    if isinstance(st.get("talkie_fetched"), int) and not st.get("arms345_submitted"):
        rc, out = sh([sys.executable, str(ROOT / "scripts" / "v3" / "build_talkie_direct_set.py"),
                      str(N / "talkie_direct" / "talkie_direct_answers.jsonl")])
        log(f"filters rc={rc}: " + " ".join(out.split())[:600])
        if rc == 0:
            rep = json.loads((N / "data" / "talkie_direct_report.json").read_text(encoding="utf-8"))
            base = ["--base", str(CPT_DIR), "--base-revision", CPT_REV] if cpt_ok(st) else []
            tag = "cpt_" if base else "base_"
            f = rep["files"]
            for arm, key, what in (("talkie_all", "arm3", "Talkie answers (book + encyclopaedia)"),
                                   ("everything", "arm4", "book Q&A + all Talkie answers"),
                                   ("no_encyclopedia", "arm5", "book Q&A + Talkie answers to book questions")):
                n = f[key]["rows"]
                submit(f"ranke-{arm}", f"Ranke night: {what}, {n} pairs, LoRA r64 3 epochs",
                       int(n * 3 / 12 / 60 * 1.3) + 10, n * 3,
                       ["--arm", tag + arm + "_r64", "--data", f[key]["file"], "--epochs", "3", "--eval-epochs", "1,2,3"] + base)
            st["arms345_submitted"] = datetime.now().isoformat(timespec="seconds")
    # D. second Talkie job (47 new books; router eff38f7a6c / Slurm 6698787)
    if not st.get("talkie2_fetched"):
        rc, out = sh(CH + ["status", "--id", "eff38f7a6c"], timeout=120)
        m = re.search(r"\]: (\w+)", out)
        state = m.group(1) if m else "UNKNOWN"
        if state != st.get("talkie2_state"):
            log(f"Talkie job 2 state {st.get('talkie2_state')} -> {state} :: {out.strip().splitlines()[0][:200] if out.strip() else ''}")
            st["talkie2_state"] = state
        if state in ("COMPLETED", "TIMEOUT", "FAILED", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "DONE"):
            dest = N / "talkie_direct_v3"
            dest.mkdir(exist_ok=True)
            rc, out = sh(CH + ["fetch", "--remote", REMOTE_ANS.replace(".jsonl", "_v3.jsonl"), "--local", str(dest),
                               "--overwrite"], 900)
            f = dest / "talkie_direct_answers_v3.jsonl"
            rows = sum(1 for _ in f.open(encoding="utf-8")) if f.exists() else 0
            log(f"Talkie job 2 ended ({state}); fetched {rows} of 9288 answers (fetch rc={rc})")
            st["talkie2_fetched"] = rows if rows >= 500 else "too_few"
    # E. arm 6 = everything plus (needs arms 3-5 data built and job 2 home)
    if st.get("arms345_submitted") and isinstance(st.get("talkie2_fetched"), int) and not st.get("arm6_submitted"):
        rc, out = sh([sys.executable, str(ROOT / "scripts" / "v3" / "build_everything_plus.py")])
        log(f"everything-plus build rc={rc}: " + " ".join(out.split())[:700])
        if rc == 0:
            rep = json.loads((N / "data" / "everything_plus_report.json").read_text(encoding="utf-8"))
            n = rep["files"]["train"]["rows"]
            base = ["--base", str(CPT_DIR), "--base-revision", CPT_REV] if cpt_ok(st) else []
            if submit("ranke-everything-plus", f"Ranke night BEST CANDIDATE: all book pairs (58 books) + all Talkie answers, {n} pairs",
                      int(n * 3 / 12 / 60 * 1.3) + 10, n * 3,
                      ["--arm", ("cpt_" if base else "base_") + "everything_plus_r64", "--data", rep["files"]["train"]["file"],
                       "--epochs", "3", "--eval-epochs", "1,2,3"] + base):
                st["arm6_submitted"] = datetime.now().isoformat(timespec="seconds")
    STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")


if __name__ == "__main__":
    LOCK = N / "orchestrator.lock"          # a tick can take minutes (fetch, build); never run two at once
    if LOCK.exists() and time.time() - LOCK.stat().st_mtime < 1800:
        sys.exit(0)
    LOCK.write_text(str(datetime.now()), encoding="utf-8")
    try:
        main()
    except Exception as ex:          # never die silently
        log(f"ERROR {type(ex).__name__}: {ex}")
        raise
    finally:
        LOCK.unlink(missing_ok=True)   # lock file only (contains a timestamp), not data
