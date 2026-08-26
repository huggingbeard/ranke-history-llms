"""
Review chain for the time-locked-LLM contamination question.

Step 1 (already done by Claude): 01_claude_analysis.md  -- the advice.
Step 2 (this script): send it to an OpenAI *codex* model via OpenRouter for a
        tough critique  -> 02_codex_feedback.md
Step 3 (this script): hand BOTH (advice + critique) to openrouter/fusion and ask
        for a short, plain-English practical synopsis -> 03_fusion_synopsis.md

Uses the user's own safe-call helper (empty-response guard, usage/cost tracking).
Every step is saved to disk immediately (checkpoint) before the next step runs.
"""
import sys, pathlib

API_DIR = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api"
sys.path.insert(0, API_DIR)
from openrouter_safe_call import or_chat  # noqa: E402

HERE = pathlib.Path(r"C:\Users\hvoth\timelock-contamination")
analysis = (HERE / "01_claude_analysis.md").read_text(encoding="utf-8")
critic_prompt = (HERE / "prompt_for_codex.txt").read_text(encoding="utf-8")

CODEX_MODEL = "openai/gpt-5.3-codex"   # newest 'codex' model available on OpenRouter
FUSION_MODEL = "openrouter/fusion"

# ----------------------------------------------------------------------------
# STEP 2 -- the codex-style critique
# ----------------------------------------------------------------------------
# The critic prompt references a file path; replace that with the actual text
# so the API model (which can't read the disk) sees the document inline.
critic_instructions = critic_prompt.replace(
    "Read the advice document at this path:\n  C:\\Users\\hvoth\\timelock-contamination\\01_claude_analysis.md",
    "Here is the advice document, between the markers:")

critic_msg = [
    {"role": "system",
     "content": "You are a sharp, senior machine-learning researcher acting as an "
                "honest peer reviewer. You do not flatter. You write in plain English "
                "for a non-computer-scientist reader."},
    {"role": "user",
     "content": critic_instructions
                + "\n\n===== BEGIN ADVICE DOCUMENT =====\n"
                + analysis
                + "\n===== END ADVICE DOCUMENT =====\n"},
]

print(f"[step2] calling critic model {CODEX_MODEL} ...", flush=True)
critique, m2 = or_chat(
    critic_msg,
    model=CODEX_MODEL,
    max_tokens=48000,      # room for reasoning + a ~1000-word answer + margin
    reasoning_max=16000,   # cap thinking so it can't eat the whole budget
    cheapest=False,        # route straight to the model, no provider games
    temperature=1.0,       # reasoning models prefer temp=1
    retries=2,
    timeout=900,
)
(HERE / "02_codex_feedback.md").write_text(critique, encoding="utf-8")
print(f"[step2] DONE  provider={m2['provider']}  finish={m2['finish_reason']}  "
      f"cost=${(m2['cost'] or 0):.4f}  chars={len(critique)}", flush=True)

# ----------------------------------------------------------------------------
# STEP 3 -- fusion reads the whole discussion and writes a practical synopsis
# ----------------------------------------------------------------------------
discussion = (
    "Below is a discussion between two AI assistants helping an economic historian.\n"
    "The historian trained a language model only on text up to the year 1913, and wants\n"
    "to find out whether it secretly learned about events AFTER 1913 (this is called\n"
    "'training-data contamination'). Assistant A wrote advice. Assistant B (a reviewer)\n"
    "critiqued it.\n\n"
    "===== ASSISTANT A: THE ADVICE =====\n" + analysis +
    "\n\n===== ASSISTANT B: THE CRITIQUE =====\n" + critique + "\n"
)

fusion_task = (
    "You are a third expert. Read the whole discussion above and write a PRACTICAL "
    "SYNOPSIS for the historian. Rules:\n"
    "- Plain English, no jargon, explain like to a smart 15-year-old. The reader is NOT "
    "a computer scientist and hates buzzwords and noun-stacking.\n"
    "- Be decisive. Tell him what to actually DO, in what order, this week vs later.\n"
    "- Where A and B disagree, say who is right (or what the truth in the middle is) and "
    "why, in one or two plain sentences.\n"
    "- Call out the single biggest risk that could still make him fool himself.\n"
    "- Give him a short 'how would I know it worked / how would I know it failed' check "
    "for the main method.\n"
    "- End with a tight numbered action list (max 7 items) he can hand to his research "
    "assistant tomorrow morning.\n"
    "- Around 500-800 words. Do not just repeat A and B; resolve them into one clear path."
)

fusion_msg = [
    {"role": "system",
     "content": "You are a careful, decisive research advisor who writes in warm, "
                "concrete, plain English for non-technical readers."},
    {"role": "user", "content": discussion + "\n\n===== YOUR TASK =====\n" + fusion_task},
]

print(f"[step3] calling {FUSION_MODEL} for the synopsis ...", flush=True)
synopsis, m3 = or_chat(
    fusion_msg,
    model=FUSION_MODEL,
    max_tokens=30000,
    reasoning_max=None,    # unknown if it's a reasoning model; don't send the param
    cheapest=False,
    temperature=1.0,
    retries=2,
    timeout=900,
)
(HERE / "03_fusion_synopsis.md").write_text(synopsis, encoding="utf-8")
print(f"[step3] DONE  provider={m3['provider']}  finish={m3['finish_reason']}  "
      f"cost=${(m3['cost'] or 0):.4f}  chars={len(synopsis)}", flush=True)

print("\n=========== ALL STEPS COMPLETE ===========")
