#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence

from cai_stage_utils import load_selected_source_rows, ordered_stage_rows
from cai_utils import (
    build_critique_prompt,
    build_fallback_critique_text,
    critique_missing_fields,
    ensure_dir,
    generate_responses,
    get_constitution,
    load_generation_model,
    load_jsonl,
    parse_critique_output,
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
        description="Generate CAI critiques for a set of initial outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("critic_model_name_or_path")
    parser.add_argument("critic_family", choices=["llama", "gemma", "qwen", "gpt-oss"])
    parser.add_argument("output_dir")
    parser.add_argument("source_file")
    parser.add_argument("initial_outputs_file")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "bfloat16"))
    parser.add_argument("--attn-implementation", default=os.getenv("ATTN_IMPL"))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=env_int("MAX_EXAMPLES", None))
    parser.add_argument("--max-new-tokens", type=int, default=env_int("CRITIQUE_MAX_NEW_TOKENS", 96))
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
    final_valid = 0
    retries = 0
    fallbacks = 0
    invalid_attempt_total = 0
    missing_field_counts: Dict[str, int] = {}
    for record in records:
        if record["critique_final_valid"]:
            final_valid += 1
        if record["critique_attempts"] > 1:
            retries += 1
        if record["critique_fallback_used"]:
            fallbacks += 1
        invalid_attempt_total += int(record["critique_invalid_attempts"])
        for missing_fields in record["critique_missing_fields_by_attempt"]:
            for field in missing_fields:
                missing_field_counts[field] = missing_field_counts.get(field, 0) + 1
    return {
        "final_valid_rows": final_valid,
        "retry_rows": retries,
        "fallback_rows": fallbacks,
        "invalid_attempt_total": invalid_attempt_total,
        "missing_field_counts": missing_field_counts,
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
    constitution = get_constitution()

    preview_rows = []
    for idx, (source_row, initial_row) in enumerate(zip(source_rows[: min(3, len(source_rows))], initial_rows[: min(3, len(initial_rows))])):
        preview_rows.append(
            {
                "index": idx,
                "example_id": source_row["example_id"],
                "critique_prompt": build_critique_prompt(
                    model_family=args.critic_family,
                    constitution=constitution,
                    user_request=source_row["user_request"],
                    tools=source_row["tools"],
                    assistant_response=initial_row.get("initial_output_raw") or initial_row["initial_output"],
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
        model_name_or_path=args.critic_model_name_or_path,
        hf_token=args.hf_token,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )

    records: List[Dict[str, Any]] = []
    iterator = progress(
        range(0, len(source_rows), args.batch_size),
        total=(len(source_rows) + args.batch_size - 1) // args.batch_size,
        desc=f"{args.critic_family} critiques",
        leave=False,
    )
    for start_idx in iterator:
        batch_source_rows = source_rows[start_idx : start_idx + args.batch_size]
        batch_initial_rows = initial_rows[start_idx : start_idx + args.batch_size]
        prompts = []
        for source_row, initial_row in zip(batch_source_rows, batch_initial_rows):
            assistant_response = initial_row.get("initial_output_raw") or initial_row["initial_output"]
            prompts.append(
                build_critique_prompt(
                    model_family=args.critic_family,
                    constitution=constitution,
                    user_request=source_row["user_request"],
                    tools=source_row["tools"],
                    assistant_response=assistant_response,
                    tokenizer=tokenizer,
                )
            )
        raw_critiques = generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            model_family=args.critic_family,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            seed=args.seed + start_idx,
            max_prompt_length=args.max_prompt_length,
        )
        for source_row, first_raw in zip(batch_source_rows, raw_critiques):
            parsed = parse_critique_output(first_raw)
            fallback_used = not parsed["valid"]
            effective_text = parsed["raw_text"] if parsed["raw_text"] else build_fallback_critique_text([first_raw], [parsed])
            records.append(
                {
                    "example_id": source_row["example_id"],
                    "source_row_index": source_row["source_row_index"],
                    "source_split": source_row["source_split"],
                    "chosen_behavior_class": source_row["chosen_behavior_class"],
                    "critique": parsed,
                    "critique_attempts": 1,
                    "critique_first_try_valid": parsed["valid"],
                    "critique_final_valid": parsed["valid"],
                    "critique_invalid_attempts": 0 if parsed["valid"] else 1,
                    "critique_missing_fields_by_attempt": [critique_missing_fields(parsed)],
                    "critique_all_attempts_raw": [first_raw],
                    "critique_fallback_used": fallback_used,
                    "critique_effective_text": effective_text,
                }
            )

    unload_model(tokenizer, model)

    output_path = out_dir / "critiques.jsonl"
    write_jsonl(output_path, records)
    save_json(
        out_dir / "summary.json",
        {
            "critic_model_name_or_path": args.critic_model_name_or_path,
            "critic_family": args.critic_family,
            "source_file": args.source_file,
            "initial_outputs_file": args.initial_outputs_file,
            "source_examples": len(source_rows),
            "outputs": {
                "critiques": str(output_path),
            },
            "result_summary": summarize_records(records),
        },
    )


if __name__ == "__main__":
    main()
