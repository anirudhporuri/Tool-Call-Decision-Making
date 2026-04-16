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
    generate_response,
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
    parser.add_argument("--temperature", type=float, default=env_float("POLICY_TEMPERATURE", 0.8))
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
    class_counts_a: Dict[str, int] = {}
    class_counts_b: Dict[str, int] = {}
    duplicate_rows = 0
    for record in records:
        class_counts_a[record["response_a_class"]] = class_counts_a.get(record["response_a_class"], 0) + 1
        class_counts_b[record["response_b_class"]] = class_counts_b.get(record["response_b_class"], 0) + 1
        if record["response_a"] == record["response_b"]:
            duplicate_rows += 1
    return {
        "response_a_class_counts": class_counts_a,
        "response_b_class_counts": class_counts_b,
        "duplicate_rows": duplicate_rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
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

    records: List[Dict[str, Any]] = []
    iterator = progress(
        enumerate(source_rows),
        total=len(source_rows),
        desc=f"{args.policy_family} response pairs",
        leave=False,
    )
    for idx, row in iterator:
        policy_prompt = build_policy_prompt(args.policy_family, row)
        response_a_raw = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=policy_prompt,
            model_family=args.policy_family,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed + (idx * 2),
        )
        response_b_raw = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=policy_prompt,
            model_family=args.policy_family,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed + (idx * 2) + 1,
        )
        eval_a = evaluate_candidate_response(response_a_raw, row["tools"])
        eval_b = evaluate_candidate_response(response_b_raw, row["tools"])
        records.append(
            {
                "example_id": row["example_id"],
                "source_row_index": row["source_row_index"],
                "source_split": row["source_split"],
                "chosen_behavior_class": row["chosen_behavior_class"],
                "tools": row["tools"],
                "messages": row["messages"],
                "user_request": row["user_request"],
                "response_a_raw": response_a_raw,
                "response_a": eval_a["canonical"],
                "response_a_class": eval_a["class"],
                "response_a_valid": eval_a["valid"],
                "response_a_validation_reason": eval_a["reason"],
                "response_a_structural_kind": eval_a["structural_kind"],
                "response_b_raw": response_b_raw,
                "response_b": eval_b["canonical"],
                "response_b_class": eval_b["class"],
                "response_b_valid": eval_b["valid"],
                "response_b_validation_reason": eval_b["reason"],
                "response_b_structural_kind": eval_b["structural_kind"],
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
