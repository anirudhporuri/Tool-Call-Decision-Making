#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence

from cai_stage_utils import load_selected_source_rows, should_enforce_strict_balance, source_balance
from cai_utils import (
    append_jsonl,
    build_policy_prompt,
    ensure_dir,
    evaluate_candidate_response,
    generate_responses,
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


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name[2:].replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", default=default, help=help_text)
    parser.add_argument(f"--no-{name[2:]}", dest=dest, action="store_false", help=argparse.SUPPRESS)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate initial CAI policy outputs for a source split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("policy_model_name_or_path")
    parser.add_argument("policy_family", choices=["llama", "gemma"])
    parser.add_argument("output_dir")
    parser.add_argument("source_file")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "bfloat16"))
    parser.add_argument("--attn-implementation", default=os.getenv("ATTN_IMPL"))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=env_int("MAX_EXAMPLES", None))
    parser.add_argument("--max-new-tokens", type=int, default=env_int("POLICY_MAX_NEW_TOKENS", 256))
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", 2))
    parser.add_argument("--max-prompt-length", type=int, default=env_int("MAX_PROMPT_LENGTH", 1024))
    parser.add_argument("--temperature", type=float, default=env_float("POLICY_TEMPERATURE", 0.7))
    parser.add_argument("--top-p", type=float, default=env_float("POLICY_TOP_P", 0.95))
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
    validation_reason_counts: Dict[str, int] = {}
    valid_count = 0
    for record in records:
        output_class = record["initial_output_class"]
        structural_kind = record["initial_output_structural_kind"]
        class_counts[output_class] = class_counts.get(output_class, 0) + 1
        structural_kind_counts[structural_kind] = structural_kind_counts.get(structural_kind, 0) + 1
        if record["initial_output_valid"]:
            valid_count += 1
        elif record["initial_output_validation_reason"]:
            reason = record["initial_output_validation_reason"]
            validation_reason_counts[reason] = validation_reason_counts.get(reason, 0) + 1
    return {
        "output_class_counts": class_counts,
        "structural_kind_counts": structural_kind_counts,
        "valid_rows": valid_count,
        "invalid_reason_counts": validation_reason_counts,
    }


def index_records_by_example_id(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        example_id = record.get("example_id")
        if not example_id:
            raise ValueError(f"Initial output record missing example_id: {record}")
        if example_id in indexed:
            raise ValueError(f"Duplicate initial output record for example_id={example_id}")
        indexed[example_id] = record
    return indexed


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    out_dir = ensure_dir(args.output_dir)
    save_json(out_dir / "run_config.json", sanitized_args_dict(args))

    max_examples = resolve_max_examples(args)
    strict_balance = should_enforce_strict_balance(
        dry_run=args.dry_run,
        smoke_run=args.smoke_run,
        start_index=args.start_index,
        max_examples=max_examples,
    )
    source_rows = load_selected_source_rows(
        source_file=args.source_file,
        start_index=args.start_index,
        max_examples=max_examples,
        smoke_run=args.smoke_run and args.max_examples is None,
    )
    selected_balance = source_balance(source_rows, strict_balance)

    preview_rows = [
        {
            "index": idx,
            "example_id": row["example_id"],
            "chosen_behavior_class": row["chosen_behavior_class"],
            "policy_prompt": build_policy_prompt(args.policy_family, row),
        }
        for idx, row in enumerate(source_rows[: min(3, len(source_rows))])
    ]
    write_jsonl(out_dir / "prompt_previews.jsonl", preview_rows)

    if args.dry_run:
        save_json(
            out_dir / "dry_run_summary.json",
            {
                "mode": "dry_run",
                "source_examples": len(source_rows),
                "source_balance": selected_balance,
                "saved_preview": str(out_dir / "prompt_previews.jsonl"),
            },
        )
        return

    output_path = out_dir / "initial_outputs.jsonl"
    existing_records: List[Dict[str, Any]] = []
    if output_path.exists():
        existing_records = load_jsonl(output_path, allow_partial_last_line=True)
        if output_path.stat().st_size > 0 and not existing_records:
            raise RuntimeError(
                f"Existing initial outputs file {output_path} is non-empty but no rows could be recovered. "
                "Refusing to overwrite it automatically."
            )
        print(f"Found existing initial outputs file at {output_path} with {len(existing_records)} parsed rows.")
    existing_by_id = index_records_by_example_id(existing_records)
    source_example_ids = {row["example_id"] for row in source_rows}
    extra_ids = sorted(set(existing_by_id) - source_example_ids)
    if extra_ids:
        raise ValueError(
            f"Existing initial outputs contain rows not present in the selected source set: {extra_ids[:3]}"
        )
    remaining_rows = [row for row in source_rows if row["example_id"] not in existing_by_id]

    if existing_records:
        print(
            f"Resuming initial outputs from {output_path} with "
            f"{len(existing_records)} completed rows and {len(remaining_rows)} remaining."
        )

    tokenizer = None
    model = None
    if remaining_rows:
        tokenizer, model = load_generation_model(
            model_name_or_path=args.policy_model_name_or_path,
            hf_token=args.hf_token,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            load_in_4bit=args.load_in_4bit,
            trust_remote_code=args.trust_remote_code,
        )

        iterator = progress(
            range(0, len(remaining_rows), args.batch_size),
            total=(len(remaining_rows) + args.batch_size - 1) // args.batch_size,
            desc=f"{args.policy_family} initial outputs",
            leave=False,
        )
        for batch_offset in iterator:
            batch_rows = remaining_rows[batch_offset : batch_offset + args.batch_size]
            prompts = [build_policy_prompt(args.policy_family, row) for row in batch_rows]
            initial_outputs_raw = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                model_family=args.policy_family,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + batch_rows[0]["source_row_index"],
                max_prompt_length=args.max_prompt_length,
            )
            batch_records: List[Dict[str, Any]] = []
            for row, initial_output_raw in zip(batch_rows, initial_outputs_raw):
                evaluation = evaluate_candidate_response(initial_output_raw, row["tools"])
                record = {
                    "example_id": row["example_id"],
                    "source_row_index": row["source_row_index"],
                    "source_split": row["source_split"],
                    "chosen_behavior_class": row["chosen_behavior_class"],
                    "tools": row["tools"],
                    "messages": row["messages"],
                    "user_request": row["user_request"],
                    "initial_output_raw": initial_output_raw,
                    "initial_output": evaluation["canonical"],
                    "initial_output_class": evaluation["class"],
                    "initial_output_valid": evaluation["valid"],
                    "initial_output_validation_reason": evaluation["reason"],
                    "initial_output_structural_kind": evaluation["structural_kind"],
                }
                existing_by_id[row["example_id"]] = record
                batch_records.append(record)
            append_jsonl(output_path, batch_records)

        unload_model(tokenizer, model)

    ordered_records = [existing_by_id[row["example_id"]] for row in source_rows]
    write_jsonl(output_path, ordered_records)
    print(f"Wrote complete initial outputs to {output_path} ({len(ordered_records)} rows).")
    save_json(
        out_dir / "summary.json",
        {
            "policy_model_name_or_path": args.policy_model_name_or_path,
            "policy_family": args.policy_family,
            "source_file": args.source_file,
            "source_examples": len(source_rows),
            "source_balance": selected_balance,
            "strict_balance_enforced": strict_balance,
            "resumed_existing_rows": len(existing_records),
            "outputs": {
                "initial_outputs": str(output_path),
            },
            "result_summary": summarize_records(ordered_records),
        },
    )
    print(f"Wrote initial output summary to {out_dir / 'summary.json'}.")


if __name__ == "__main__":
    main()
