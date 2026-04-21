#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence

from cai_stage_utils import load_selected_source_rows, should_enforce_strict_balance, source_balance
from cai_utils import (
    build_policy_prompt,
    ensure_dir,
    evaluate_candidate_response,
    generate_responses,
    load_generation_model,
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
        description="Generate CAI DPO response pairs for a source split.",
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
    parser.add_argument("--temperature", type=float, default=env_float("POLICY_TEMPERATURE", 0.9))
    parser.add_argument("--top-p", type=float, default=env_float("POLICY_TOP_P", 0.95))
    parser.add_argument(
        "--candidate-attempts",
        type=int,
        default=env_int("PAIR_CANDIDATE_ATTEMPTS", 6),
        help="Maximum number of sampled attempts per prompt while searching for two distinct canonical responses.",
    )
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
    class_counts_a: Dict[str, int] = {}
    class_counts_b: Dict[str, int] = {}
    duplicate_rows = 0
    resampled_rows = 0
    max_sampling_attempts = 0
    for record in records:
        class_counts_a[record["response_a_class"]] = class_counts_a.get(record["response_a_class"], 0) + 1
        class_counts_b[record["response_b_class"]] = class_counts_b.get(record["response_b_class"], 0) + 1
        if record["response_a"] == record["response_b"]:
            duplicate_rows += 1
        if record.get("sampling_attempts", 0) > 2:
            resampled_rows += 1
        max_sampling_attempts = max(max_sampling_attempts, int(record.get("sampling_attempts", 0)))
    return {
        "response_a_class_counts": class_counts_a,
        "response_b_class_counts": class_counts_b,
        "duplicate_rows": duplicate_rows,
        "resampled_rows": resampled_rows,
        "max_sampling_attempts": max_sampling_attempts,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.candidate_attempts < 2:
        raise ValueError("--candidate-attempts must be at least 2.")
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

    tokenizer, model = load_generation_model(
        model_name_or_path=args.policy_model_name_or_path,
        hf_token=args.hf_token,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )

    states: List[Dict[str, Any]] = []
    for idx, row in enumerate(source_rows):
        states.append(
            {
                "row_index": idx,
                "row": row,
                "prompt": build_policy_prompt(args.policy_family, row),
                "sampled_candidates": [],
                "distinct_candidates": [],
                "seen_canonicals": set(),
            }
        )

    attempt_iterator = progress(
        range(args.candidate_attempts),
        total=args.candidate_attempts,
        desc=f"{args.policy_family} response pair attempts",
        leave=False,
    )
    for attempt_idx in attempt_iterator:
        active_state_indices = [
            idx for idx, state in enumerate(states) if len(state["distinct_candidates"]) < 2
        ]
        if not active_state_indices:
            break
        batch_iterator = progress(
            range(0, len(active_state_indices), args.batch_size),
            total=(len(active_state_indices) + args.batch_size - 1) // args.batch_size,
            desc=f"{args.policy_family} response pair batches",
            leave=False,
        )
        for batch_start in batch_iterator:
            state_index_batch = active_state_indices[batch_start : batch_start + args.batch_size]
            prompt_batch = [states[state_idx]["prompt"] for state_idx in state_index_batch]
            raw_outputs = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompt_batch,
                model_family=args.policy_family,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + (attempt_idx * 100_000) + batch_start,
                max_prompt_length=args.max_prompt_length,
            )
            for state_idx, response_raw in zip(state_index_batch, raw_outputs):
                state = states[state_idx]
                evaluation = evaluate_candidate_response(response_raw, state["row"]["tools"])
                candidate = {
                    "raw": response_raw,
                    "canonical": evaluation["canonical"],
                    "class": evaluation["class"],
                    "valid": evaluation["valid"],
                    "validation_reason": evaluation["reason"],
                    "structural_kind": evaluation["structural_kind"],
                }
                state["sampled_candidates"].append(candidate)
                canonical_key = candidate["canonical"]
                if canonical_key not in state["seen_canonicals"]:
                    state["seen_canonicals"].add(canonical_key)
                    state["distinct_candidates"].append(candidate)

    records: List[Dict[str, Any]] = []
    for state in states:
        row = state["row"]
        sampled_candidates = state["sampled_candidates"]
        distinct_candidates = state["distinct_candidates"]
        if not sampled_candidates:
            raise RuntimeError(f"No sampled candidates were generated for example_id={row['example_id']}.")
        response_a = distinct_candidates[0] if distinct_candidates else sampled_candidates[0]
        response_b = distinct_candidates[1] if len(distinct_candidates) >= 2 else sampled_candidates[-1]
        records.append(
            {
                "example_id": row["example_id"],
                "source_row_index": row["source_row_index"],
                "source_split": row["source_split"],
                "chosen_behavior_class": row["chosen_behavior_class"],
                "tools": row["tools"],
                "messages": row["messages"],
                "user_request": row["user_request"],
                "sampling_attempts": len(sampled_candidates),
                "distinct_candidates_found": len(distinct_candidates),
                "response_a_raw": response_a["raw"],
                "response_a": response_a["canonical"],
                "response_a_class": response_a["class"],
                "response_a_valid": response_a["valid"],
                "response_a_validation_reason": response_a["validation_reason"],
                "response_a_structural_kind": response_a["structural_kind"],
                "response_b_raw": response_b["raw"],
                "response_b": response_b["canonical"],
                "response_b_class": response_b["class"],
                "response_b_valid": response_b["valid"],
                "response_b_validation_reason": response_b["validation_reason"],
                "response_b_structural_kind": response_b["structural_kind"],
            }
        )

    unload_model(tokenizer, model)

    output_path = out_dir / "response_pairs.jsonl"
    write_jsonl(output_path, records)
    save_json(
        out_dir / "summary.json",
        {
            "policy_model_name_or_path": args.policy_model_name_or_path,
            "policy_family": args.policy_family,
            "source_file": args.source_file,
            "source_examples": len(source_rows),
            "source_balance": selected_balance,
            "strict_balance_enforced": strict_balance,
            "outputs": {
                "response_pairs": str(output_path),
            },
            "result_summary": summarize_records(records),
        },
    )


if __name__ == "__main__":
    main()
