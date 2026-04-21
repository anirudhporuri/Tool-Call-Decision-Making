#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence

from cai_stage_utils import load_selected_source_rows, ordered_stage_rows
from cai_utils import (
    build_revision_prompt,
    ensure_dir,
    evaluate_candidate_response,
    generate_responses,
    get_constitution,
    load_generation_model,
    load_jsonl,
    progress,
    save_json,
    unload_model,
    write_jsonl,
)


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.getenv(name)
    return int(value) if value is not None else default


def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name[2:].replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", default=default, help=help_text)
    parser.add_argument(f"--no-{name[2:]}", dest=dest, action="store_false", help=argparse.SUPPRESS)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CAI revisions from initial outputs and critiques.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("revision_model_name_or_path")
    parser.add_argument("revision_family", choices=["llama", "gemma", "qwen", "gpt-oss"])
    parser.add_argument("output_dir")
    parser.add_argument("source_file")
    parser.add_argument("initial_outputs_file")
    parser.add_argument("critiques_file")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "bfloat16"))
    parser.add_argument("--attn-implementation", default=os.getenv("ATTN_IMPL"))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=env_int("MAX_EXAMPLES", None))
    parser.add_argument("--max-new-tokens", type=int, default=env_int("REVISION_MAX_NEW_TOKENS", 256))
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", 2))
    parser.add_argument("--max-prompt-length", type=int, default=env_int("MAX_PROMPT_LENGTH", 1024))
    parser.add_argument("--dry-run", action="store_true", default=env_flag("DRY_RUN", False))
    parser.add_argument("--smoke-run", action="store_true", default=env_flag("SMOKE_RUN", False))
    parser.add_argument("--dry-run-max-examples", type=int, default=env_int("DRY_RUN_MAX_EXAMPLES", 8))
    parser.add_argument("--smoke-run-max-examples", type=int, default=env_int("SMOKE_RUN_MAX_EXAMPLES", 6))
    add_bool_flag(parser, "--load-in-4bit", env_flag("LOAD_IN_4BIT", True), "Load model in 4-bit.")
    add_bool_flag(parser, "--trust-remote-code", env_flag("TRUST_REMOTE_CODE", False), "Allow custom model code.")
    args = parser.parse_args(argv)
    if args.dry_run and args.smoke_run:
        parser.error("--dry-run and --smoke-run are mutually exclusive.")
    return args


def sanitized_args_dict(args: argparse.Namespace) -> Dict[str, Any]:
    payload = vars(args).copy()
    if payload.get("hf_token"):
        payload["hf_token"] = "[REDACTED]"
    return payload


def resolve_max_examples(args: argparse.Namespace) -> Optional[int]:
    if args.max_examples is not None:
        return args.max_examples
    if args.dry_run:
        return args.dry_run_max_examples
    if args.smoke_run:
        return args.smoke_run_max_examples
    return None


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    class_counts: Dict[str, int] = {}
    structural_kind_counts: Dict[str, int] = {}
    used_original_without_generation = 0
    invalid_reason_counts: Dict[str, int] = {}
    for record in records:
        revision_class = record["revised_output_class"]
        structural_kind = record["revised_output_structural_kind"]
        class_counts[revision_class] = class_counts.get(revision_class, 0) + 1
        structural_kind_counts[structural_kind] = structural_kind_counts.get(structural_kind, 0) + 1
        if record["revision_used_original_without_generation"]:
            used_original_without_generation += 1
        if not record["revised_output_valid"] and record["revised_output_validation_reason"]:
            reason = record["revised_output_validation_reason"]
            invalid_reason_counts[reason] = invalid_reason_counts.get(reason, 0) + 1
    return {
        "revision_class_counts": class_counts,
        "structural_kind_counts": structural_kind_counts,
        "used_original_without_generation_rows": used_original_without_generation,
        "invalid_reason_counts": invalid_reason_counts,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    out_dir = ensure_dir(args.output_dir)
    save_json(out_dir / "run_config.json", sanitized_args_dict(args))

    max_examples = resolve_max_examples(args)
    source_rows = load_selected_source_rows(
        source_file=args.source_file,
        start_index=args.start_index,
        max_examples=max_examples,
        smoke_run=args.smoke_run and args.max_examples is None,
    )
    initial_rows = ordered_stage_rows(source_rows, load_jsonl(args.initial_outputs_file), "initial_outputs")
    critique_rows = ordered_stage_rows(source_rows, load_jsonl(args.critiques_file), "critiques")
    constitution = get_constitution()

    preview_rows = []
    for idx, (source_row, initial_row, critique_row) in enumerate(
        zip(
            source_rows[: min(3, len(source_rows))],
            initial_rows[: min(3, len(initial_rows))],
            critique_rows[: min(3, len(critique_rows))],
        )
    ):
        preview_rows.append(
            {
                "index": idx,
                "example_id": source_row["example_id"],
                "revision_prompt": build_revision_prompt(
                    model_family=args.revision_family,
                    constitution=constitution,
                    user_request=source_row["user_request"],
                    tools=source_row["tools"],
                    assistant_response=initial_row.get("initial_output_raw") or initial_row["initial_output"],
                    critique_text=critique_row["critique_effective_text"],
                ),
            }
        )
    write_jsonl(out_dir / "prompt_previews.jsonl", preview_rows)

    if args.dry_run:
        save_json(
            out_dir / "dry_run_summary.json",
            {
                "mode": "dry_run",
                "source_examples": len(source_rows),
                "saved_preview": str(out_dir / "prompt_previews.jsonl"),
            },
        )
        return

    tokenizer, model = load_generation_model(
        model_name_or_path=args.revision_model_name_or_path,
        hf_token=args.hf_token,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )

    record_slots: List[Optional[Dict[str, Any]]] = [None] * len(source_rows)
    generation_tasks: List[Dict[str, Any]] = []
    iterator = progress(
        enumerate(zip(source_rows, initial_rows, critique_rows)),
        total=len(source_rows),
        desc=f"{args.revision_family} revisions",
        leave=False,
    )
    for idx, (source_row, initial_row, critique_row) in iterator:
        original_response = initial_row.get("initial_output_raw") or initial_row["initial_output"]
        parsed_critique = critique_row["critique"]
        used_original_without_generation = bool(
            parsed_critique.get("valid") and parsed_critique.get("verdict") == "NO_ISSUES"
        )
        if used_original_without_generation:
            revision_raw = original_response
            evaluation = evaluate_candidate_response(revision_raw, source_row["tools"])
            record_slots[idx] = {
                "example_id": source_row["example_id"],
                "source_row_index": source_row["source_row_index"],
                "source_split": source_row["source_split"],
                "chosen_behavior_class": source_row["chosen_behavior_class"],
                "revision_used_original_without_generation": used_original_without_generation,
                "revised_output_raw": revision_raw,
                "revised_output": evaluation["canonical"],
                "revised_output_class": evaluation["class"],
                "revised_output_valid": evaluation["valid"],
                "revised_output_validation_reason": evaluation["reason"],
                "revised_output_structural_kind": evaluation["structural_kind"],
            }
        else:
            generation_tasks.append(
                {
                    "record_index": idx,
                    "source_row": source_row,
                    "used_original_without_generation": used_original_without_generation,
                    "prompt": build_revision_prompt(
                        model_family=args.revision_family,
                        constitution=constitution,
                        user_request=source_row["user_request"],
                        tools=source_row["tools"],
                        assistant_response=original_response,
                        critique_text=critique_row["critique_effective_text"],
                        tokenizer=tokenizer,
                    ),
                }
            )

    generation_iterator = progress(
        range(0, len(generation_tasks), args.batch_size),
        total=(len(generation_tasks) + args.batch_size - 1) // args.batch_size if generation_tasks else 0,
        desc=f"{args.revision_family} revision generations",
        leave=False,
    )
    for start_idx in generation_iterator:
        task_batch = generation_tasks[start_idx : start_idx + args.batch_size]
        prompts = [task["prompt"] for task in task_batch]
        revision_outputs = generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            model_family=args.revision_family,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            seed=args.seed + start_idx,
            max_prompt_length=args.max_prompt_length,
        )
        for task, revision_raw in zip(task_batch, revision_outputs):
            source_row = task["source_row"]
            evaluation = evaluate_candidate_response(revision_raw, source_row["tools"])
            record_slots[task["record_index"]] = {
                "example_id": source_row["example_id"],
                "source_row_index": source_row["source_row_index"],
                "source_split": source_row["source_split"],
                "chosen_behavior_class": source_row["chosen_behavior_class"],
                "revision_used_original_without_generation": task["used_original_without_generation"],
                "revised_output_raw": revision_raw,
                "revised_output": evaluation["canonical"],
                "revised_output_class": evaluation["class"],
                "revised_output_valid": evaluation["valid"],
                "revised_output_validation_reason": evaluation["reason"],
                "revised_output_structural_kind": evaluation["structural_kind"],
            }
    if any(record is None for record in record_slots):
        raise RuntimeError("Missing revision records after batched generation.")
    records = [record for record in record_slots if record is not None]

    unload_model(tokenizer, model)

    output_path = out_dir / "revisions.jsonl"
    write_jsonl(output_path, records)
    save_json(
        out_dir / "summary.json",
        {
            "revision_model_name_or_path": args.revision_model_name_or_path,
            "revision_family": args.revision_family,
            "source_file": args.source_file,
            "initial_outputs_file": args.initial_outputs_file,
            "critiques_file": args.critiques_file,
            "source_examples": len(source_rows),
            "outputs": {
                "revisions": str(output_path),
            },
            "result_summary": summarize_records(records),
        },
    )


if __name__ == "__main__":
    main()
