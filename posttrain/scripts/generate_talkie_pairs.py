#!/usr/bin/env python3
"""Generate grounded history question-answer records with Talkie-1930 IT.

The script preserves every raw completion. It never repairs model text.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import re
import sys
import time
from pathlib import Path


MODEL_REVISION = "8033675be6360ae0127fa75f941c12d52064f1dc"
EXPECTED_REPO_ID = "talkie-lm/talkie-1930-13b-it"
ASSISTANT_PREFIX = "<question>"
FORMS = (
    (
        "cause_consequence",
        "Ask what caused an important development in the passage and what followed from it. "
        "The answer must explain the chain, not merely list facts.",
    ),
    (
        "comparison",
        "Ask for a comparison or contrast that the passage supports. The answer must identify "
        "the main difference and explain why it mattered.",
    ),
    (
        "chronology_mechanism",
        "Ask how a change occurred through time. The answer must connect the stages and mechanism.",
    ),
    (
        "motive_constraint",
        "Ask about an actor's purpose and the limits under which the actor worked. The answer must "
        "weigh both motive and constraint.",
    ),
    (
        "argument_judgment",
        "Ask for the passage's main historical argument and a measured judgment of it. The answer "
        "must state the argument, its evidence, and any qualification found in the passage.",
    ),
    (
        "institution",
        "Ask how an institution or constitutional rule worked in practice. The answer must explain "
        "the mechanism and its political effect.",
    ),
)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                yield line_no, json.loads(line)


def form_for(passage_id: str, variant: int) -> tuple[str, str]:
    digest = hashlib.sha256(passage_id.encode("utf-8")).digest()
    start = int.from_bytes(digest[:2], "big") % len(FORMS)
    # Three places apart gives two clearly different jobs when variants=2.
    return FORMS[(start + 3 * variant) % len(FORMS)]


def build_prompt(row: dict, form_instruction: str) -> str:
    return f"""You are preparing a question and answer for an educated reader in 1929.

Use only the historical passage below. Do not add a name, date, event, cause, or judgment that the passage does not support. Do not refer to "the passage", "the text", or "the author" in the question or answer. Do not copy whole sentences when you can explain them plainly.

Task: {form_instruction}

Write one self-contained question. Then write a direct answer of 90 to 220 words. The answer must make a clear claim, connect evidence by reasoning, and finish cleanly. Use prose natural to a serious historical work of the period. Do not use headings inside the answer.

Return exactly this form:
<question>one question</question>
<answer>one answer</answer>

Source: {row['author']}, {row['title']} ({row['publication_year']})

{row['text']}
"""


def parse_completion(raw: str) -> tuple[str | None, str | None, str]:
    q = re.search(r"<question>\s*(.*?)\s*</question>", raw, flags=re.I | re.S)
    a = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.I | re.S)
    if not q or not a:
        return None, None, "missing_tags"
    question = re.sub(r"\s+", " ", q.group(1)).strip()
    answer = re.sub(r"\s+", " ", a.group(1)).strip()
    if not question or not answer:
        return None, None, "empty_field"
    return question, answer, "parsed"


def generate_structured_completion(
    model, prompt: str, raw_path: Path | None = None, **generation_kwargs
) -> tuple[str, int, bool]:
    """Start the IT assistant turn with a fixed, non-semantic delimiter.

    The first gate showed that an empty assistant turn can make this small IT
    model ignore the requested XML and emit a free-standing passage.  This
    keeps the user prompt unchanged and leaves every question and answer word
    to Talkie.
    """
    if model.spec.style != "it":
        raise ValueError("Structured generation requires Talkie's IT checkpoint")
    from talkie.chat import format_prompt

    formatted = format_prompt(prompt) + ASSISTANT_PREFIX
    chunks = [ASSISTANT_PREFIX]
    chunk_count = 0
    import torch

    if raw_path is None:
        with torch.inference_mode():
            for token in model._stream_raw(formatted, **generation_kwargs):
                chunks.append(token)
                chunk_count += 1
        return "".join(chunks), chunk_count, chunk_count >= generation_kwargs["max_tokens"]

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = raw_path.with_name(raw_path.name + ".partial")
    with partial_path.open("w", encoding="utf-8") as raw_handle:
        raw_handle.write(ASSISTANT_PREFIX)
        raw_handle.flush()
        with torch.inference_mode():
            for token in model._stream_raw(formatted, **generation_kwargs):
                chunks.append(token)
                chunk_count += 1
                raw_handle.write(token)
                # Preserve text already produced if generation later raises or the
                # scheduler stops the process.
                raw_handle.flush()
    partial_path.replace(raw_path)
    return "".join(chunks), chunk_count, chunk_count >= generation_kwargs["max_tokens"]


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def capture_stem(candidate_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id).strip("_")
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:80]}_{digest}"


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for _, row in iter_jsonl(path):
        if row.get("candidate_id"):
            done.add(row["candidate_id"])
    return done


def memory_report() -> dict:
    """Return small Linux memory diagnostics for the run receipt."""
    report = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                report[key.lower() + "_kb"] = value.strip()
    except OSError:
        pass
    for name, path in (
        ("cgroup_current_bytes", "/sys/fs/cgroup/memory.current"),
        ("cgroup_peak_bytes", "/sys/fs/cgroup/memory.peak"),
        ("cgroup_limit_bytes", "/sys/fs/cgroup/memory.max"),
    ):
        try:
            report[name] = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return report


def load_checkpoint_low_memory(
    checkpoint_path: Path, device: str, target_vocab_size: int | None
):
    """Load the exact Talkie weights without a full float32 CPU copy."""
    import torch
    from talkie.model import GPTConfig, TalkieModel, resize_model_embeddings

    print(
        "checkpoint_load="
        + json.dumps(
            {
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "mode": "mmap_meta_assign",
                "memory_before": memory_report(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    ckpt = torch.load(
        str(checkpoint_path), map_location="cpu", mmap=True, weights_only=True
    )
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }

    checkpoint_vocab_size = state_dict["embed.weight"].shape[0]
    config = GPTConfig(vocab_size=checkpoint_vocab_size)
    with torch.device("meta"):
        model = TalkieModel(config, torch.device("meta"))
    model.load_state_dict(state_dict, strict=True, assign=True)

    # These small non-persistent buffers are absent from the checkpoint.
    cos, sin = model._precompute_rotary_embeddings(4096, config.head_dim)
    model.cos = cos
    model.sin = sin

    target_device = torch.device(device)
    model = model.to(device=target_device, dtype=torch.bfloat16)
    del ckpt, state_dict
    if target_vocab_size is not None and checkpoint_vocab_size < target_vocab_size:
        model = resize_model_embeddings(model, target_vocab_size, target_device)
    model.device = target_device
    model.eval()
    print(
        "checkpoint_loaded="
        + json.dumps(
            {
                "dtype": str(model.embed.weight.dtype),
                "device": str(model.embed.weight.device),
                "vocab_size": model.config.vocab_size,
                "memory_after": memory_report(),
                "cuda_allocated_bytes": torch.cuda.memory_allocated(target_device)
                if target_device.type == "cuda"
                else 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return model


def load_pinned_talkie(model_name: str, revision: str, device: str, cache_dir: str | None):
    """Load Talkie from an exact Hub revision.

    Talkie's public loader follows the moving ``main`` branch and does not
    accept a revision. Resolve both files ourselves, verify that Hugging Face
    returned the requested snapshot, and replace only the loader hook used by
    Talkie's constructor.
    """
    from huggingface_hub import hf_hub_download
    from talkie.config import MODELS
    import talkie.generate as talkie_generate

    if model_name not in MODELS:
        raise ValueError(f"Unknown Talkie model: {model_name}")
    spec = MODELS[model_name]
    if spec.repo_id != EXPECTED_REPO_ID:
        raise RuntimeError(
            f"Wrong Talkie repository: expected {EXPECTED_REPO_ID}, got {spec.repo_id}"
        )
    kwargs = {
        "repo_id": spec.repo_id,
        "revision": revision,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    # Keep the logical snapshot paths returned by huggingface_hub. Calling
    # Path.resolve() here follows the cache symlinks into ``blobs/`` and hides
    # the pinned commit directory, which makes a valid exact-revision download
    # look unpinned.
    checkpoint = Path(hf_hub_download(filename=spec.checkpoint_filename, **kwargs))
    vocab = Path(hf_hub_download(filename=spec.vocab_filename, **kwargs))

    expected_marker = f"snapshots{os.sep}{revision}{os.sep}"
    for resolved in (checkpoint, vocab):
        if expected_marker not in str(resolved):
            raise RuntimeError(
                f"Revision lock failed: expected {revision} in resolved path {resolved}"
            )
        if not resolved.exists():
            raise RuntimeError(f"Pinned model file does not exist: {resolved}")

    def pinned_files(requested_name: str, cache_dir: str | None = None):
        if requested_name != model_name:
            raise ValueError(f"Pinned loader expected {model_name}, got {requested_name}")
        return checkpoint, vocab

    talkie_generate.get_model_files = pinned_files
    talkie_generate.load_checkpoint = lambda path, target_device, target_vocab_size=None: (
        load_checkpoint_low_memory(Path(path), str(target_device), target_vocab_size)
    )
    model = talkie_generate.Talkie(model_name, device=device, cache_dir=cache_dir)
    receipt = {
        "model_name": model_name,
        "repo_id": spec.repo_id,
        "revision": revision,
        "checkpoint_filename": spec.checkpoint_filename,
        "vocab_filename": spec.vocab_filename,
        "snapshot_dir": str(checkpoint.parent),
        "checkpoint_blob": str(checkpoint.resolve()),
        "vocab_blob": str(vocab.resolve()),
    }
    return model, receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="talkie-1930-13b-it")
    ap.add_argument("--model-revision", default=MODEL_REVISION)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variants", type=int, default=1, choices=(1, 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1929)
    ap.add_argument("--temperature", type=float, default=0.55)
    ap.add_argument("--top-p", type=float, default=0.90)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--hf-home")
    ap.add_argument("--cache-dir")
    ap.add_argument("--code-archive-sha256", required=True)
    ap.add_argument(
        "--capture-dir",
        help="Directory for independent request, streamed raw text, and record files",
    )
    args = ap.parse_args()

    if args.num_shards < 1:
        ap.error("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        ap.error("--shard-index must be between 0 and num-shards - 1")

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture_dir = (
        Path(args.capture_dir)
        if args.capture_dir
        else output_path.with_name(output_path.name + ".captures")
    )
    capture_dir.mkdir(parents=True, exist_ok=True)

    all_jobs = []
    for _, row in iter_jsonl(input_path):
        for variant in range(args.variants):
            form_name, instruction = form_for(row["passage_id"], variant)
            all_jobs.append((row, variant, form_name, instruction))
    jobs = [
        job for global_index, job in enumerate(all_jobs)
        if global_index % args.num_shards == args.shard_index
    ]
    if args.limit:
        jobs = jobs[: args.limit]

    done = completed_ids(output_path)
    jobs = [j for j in jobs if f"{j[0]['passage_id']}::v{j[1]}" not in done]
    print(f"jobs={len(jobs)} already_done={len(done)}", flush=True)
    if not jobs:
        return 0

    run_request_path = output_path.with_name(output_path.name + ".request.json")
    run_request = {
        "input": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output": str(output_path),
        "candidate_ids": [f"{j[0]['passage_id']}::v{j[1]}" for j in jobs],
        "model": args.model,
        "expected_repo_id": EXPECTED_REPO_ID,
        "model_revision": args.model_revision,
        "device": args.device,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "hf_home": args.hf_home,
        "cache_dir": args.cache_dir,
        "code_archive_sha256": args.code_archive_sha256,
    }
    # Written before model loading so even an early startup failure leaves the
    # exact request and input hash.
    write_json_atomic(run_request_path, run_request)

    import torch
    started = time.time()
    print(
        f"loading model={args.model} revision={args.model_revision} device={args.device}",
        flush=True,
    )
    model, model_receipt = load_pinned_talkie(
        args.model, args.model_revision, args.device, args.cache_dir
    )
    stream_signature = inspect.signature(model._stream_raw)
    stream_parameters = stream_signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in stream_parameters.values()
    )
    required_sampling_args = {"temperature", "top_p", "top_k", "max_tokens"}
    missing_sampling_args = (
        set() if accepts_kwargs else required_sampling_args - set(stream_parameters)
    )
    if missing_sampling_args:
        raise RuntimeError(
            "Talkie _stream_raw cannot accept required sampling arguments: "
            + ", ".join(sorted(missing_sampling_args))
        )
    model_receipt["stream_raw_signature"] = str(stream_signature)
    print("model_lock_verified=" + json.dumps(model_receipt, sort_keys=True), flush=True)
    print(f"model_loaded elapsed_seconds={time.time() - started:.1f}", flush=True)

    generation_ok = 0
    generation_errors = 0
    parsed_count = 0
    missing_tags_count = 0
    with output_path.open("a", encoding="utf-8") as out:
        for index, (source, variant, form_name, instruction) in enumerate(jobs):
            candidate_id = f"{source['passage_id']}::v{variant}"
            # Keep each candidate's seed unchanged when the run is split into shards.
            seed_offset = int(hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:8], 16)
            item_seed = args.seed + seed_offset
            random.seed(item_seed)
            torch.manual_seed(item_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(item_seed)
            prompt = build_prompt(source, instruction)
            stem = capture_stem(candidate_id)
            request_path = capture_dir / f"{stem}.request.json"
            raw_path = capture_dir / f"{stem}.completion.txt"
            record_path = capture_dir / f"{stem}.record.json"
            request_receipt = {
                "candidate_id": candidate_id,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "assistant_prefix": ASSISTANT_PREFIX,
                "code_archive_sha256": args.code_archive_sha256,
                "model": args.model,
                "model_revision": args.model_revision,
                "seed": item_seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_tokens": args.max_tokens,
            }
            # This exists before this generation call, so a generation failure
            # still leaves the exact candidate request.
            write_json_atomic(request_path, request_receipt)
            record = {
                "candidate_id": candidate_id,
                "passage_id": source["passage_id"],
                "source_id": source["source_id"],
                "source_title": source["title"],
                "source_author": source["author"],
                "source_year": source["publication_year"],
                "source_text_sha256": source["text_sha256"],
                "form": form_name,
                "variant": variant,
                "seed": item_seed,
                "model": args.model,
                "model_revision": args.model_revision,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_tokens": args.max_tokens,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "assistant_prefix": ASSISTANT_PREFIX,
                "code_archive_sha256": args.code_archive_sha256,
                "request_capture": str(request_path),
                "raw_capture": str(raw_path),
                "record_capture": str(record_path),
            }
            try:
                raw, generated_chunks, hit_max_tokens = generate_structured_completion(
                    model,
                    prompt,
                    raw_path=raw_path,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens,
                )
                question, answer, parse_status = parse_completion(raw)
                record.update(
                    {
                        "generation_status": "ok",
                        "parse_status": parse_status,
                        "question": question,
                        "answer": answer,
                        "raw_completion": raw,
                        "generated_stream_chunks": generated_chunks,
                        "hit_max_tokens": hit_max_tokens,
                    }
                )
                generation_ok += 1
                if parse_status == "parsed":
                    parsed_count += 1
                if parse_status == "missing_tags":
                    missing_tags_count += 1
            except Exception as exc:  # preserve failure evidence and continue
                record.update(
                    {
                        "generation_status": "error",
                        "parse_status": "not_parsed",
                        "question": None,
                        "answer": None,
                        "raw_completion": None,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                partial_path = raw_path.with_name(raw_path.name + ".partial")
                if partial_path.exists():
                    record["partial_raw_capture"] = str(partial_path)
                    record["partial_raw_bytes"] = partial_path.stat().st_size
                generation_errors += 1
            write_json_atomic(record_path, record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"progress={index + 1}/{len(jobs)} id={candidate_id} "
                f"status={record['generation_status']} parse={record['parse_status']} "
                f"elapsed_seconds={time.time() - started:.1f}",
                flush=True,
            )
    receipt_path = output_path.with_name(output_path.name + ".receipt.json")
    receipt = {
        "status": "complete" if generation_errors == 0 else "completed_with_errors",
        "new_rows": len(jobs),
        "generation_ok": generation_ok,
        "generation_errors": generation_errors,
        "parsed": parsed_count,
        "missing_tags": missing_tags_count,
        "output": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "capture_dir": str(capture_dir),
        "request_captures": len(list(capture_dir.glob("*.request.json"))),
        "complete_raw_captures": len(list(capture_dir.glob("*.completion.txt"))),
        "partial_raw_captures": len(list(capture_dir.glob("*.completion.txt.partial"))),
        "record_captures": len(list(capture_dir.glob("*.record.json"))),
        "model_lock": model_receipt,
        "code_archive_sha256": args.code_archive_sha256,
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(receipt_path, receipt)
    print("run_receipt=" + json.dumps(receipt, sort_keys=True), flush=True)
    print(f"complete rows={len(jobs)} elapsed_seconds={time.time() - started:.1f}", flush=True)
    return 0 if generation_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
