"""Send PLAN_v2 to DeepSeek-V4-Pro for adversarial review. Saves each round verbatim."""
import os, sys, json, datetime
from pathlib import Path
API = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api")
os.environ["OPENROUTER_API_KEY"] = (API / "openrouter-secundo.txt").read_text().strip()
sys.path.insert(0, str(API))
from openrouter_safe_call import or_chat
root = Path(__file__).resolve().parents[1]
rnd = sys.argv[1]
plan = (root / "PLAN_v2_mutual_judging.md").read_text(encoding="utf-8")
extra = (root / "review" / f"context_round{rnd}.md")
extra = extra.read_text(encoding="utf-8") if extra.exists() else ""
prompt = f"""You are a hard-nosed ML-methods and research-design reviewer. Attack this overnight experiment plan.
Goal: post-train a small period-locked language model (Ranke-4B, trained only on text up to 1929) using question-answer data written by another period model (Talkie-13B-IT, cutoff 1930), where the two period models judge each other and NO modern model touches training text; modern models only grade final eval answers.

Attack: validity of each judge (log-prob grounding gain by a 4B base model; next-token A/B preference by Talkie), leakage of post-1913 knowledge, eval contamination, confounds (length/format), the dose design, DPO settings, pass/fail bars, statistical power with 84 eval prompts, compute feasibility in ~7 hours on one H100 80GB, anything missing or wrong, and cheaper/better alternatives that respect the hard constraint.
Be concrete. Number each point. Separate MUST-FIX (would make the result invalid or the run fail) from SHOULD-FIX.
End with exactly one line: `VERDICT: SHIP` or `VERDICT: REVISE` followed by numbered must-fix items.

{extra}

=== PLAN ===
{plan}
"""
txt, meta = or_chat([{"role": "user", "content": prompt}], model="deepseek/deepseek-v4-pro",
                    max_tokens=24000, reasoning_max=8000)
out = root / "review" / f"deepseek_v4pro_round{rnd}.md"
out.write_text(f"# DeepSeek-V4-Pro plan review, round {rnd}\n{datetime.datetime.now().isoformat()}\nmeta: {json.dumps(meta, default=str)}\n\n{txt}\n", encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8"); print(txt); print(meta)
