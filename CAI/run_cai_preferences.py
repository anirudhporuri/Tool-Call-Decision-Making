#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence

from cai_stage_utils import (
    load_selected_source_rows,
    ordered_stage_rows,
    should_enforce_strict_balance,
    source_balance,
)
from cai_utils import (
    build_preference_prompt,
    count_label_values,
    ensure_dir,
    generate_response,
    get_constitution,
    load_generation_model,
    load_jsonl,
    parse_preference_output,
    progress,
    save_json,
    unload_model,
    validate_balanced_counts,
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
        description="Judge CAI response pairs and write the final DPO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("judge_model_name_or_path")
    parser.add_argument("judge_family", choices=["llama", "gemma", "qwen", "gpt-oss"])
    parser.add_argument("output_dir")
    parser.add_argument("source_file")
    parser.add_argument("response_pairs_file")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "bfloat16"))
    parser.add_argument("--attn-implementation", default=os.getenv("ATTN_IMPL"))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=env_int("MAX_EXAMPLES", None))
    parser.add_argument("--max-new-tokens", type=int, default=env_int("JUDGE_MAX_NEW_TOKENS", 64))
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
    valid_counts: Dict[str, int] = {}
    chosen_class_counts: Dict[str, int] = {}
    failure_counts: Dict[str, int] = {}
    for record in records:
        label = record["chosen_behavior_class"]
        if record.get("valid"):
            valid_counts[label] = valid_counts.get(label, 0) + 1
        chosen_class = record.get("chosen_class")
        if chosen_class:
            chosen_class_counts[chosen_class] = chosen_class_counts.get(chosen_class, 0) + 1
        reason = record.get("failure_reason")
        if reason:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
    return {
        "valid_counts": valid_counts,
        "chosen_class_counts": chosen_class_counts,
        "failure_counts": failure_counts,
    }


def export_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        if not record.get("valid"):
            continue
        rows.append(
            {
                "tools": record["tools"],
                "messages": record["messages"],
                "chosen_response": {"role": "assistant", "content": record["preferred_response"]},
                "rejected_response": {"role": "assistant", "content": record["rejected_response"]},
                "source_split": "cai_dpo",
                "behavior_class": record["chosen_behavior_class"],
            }
        )
    return rows


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
    pair_rows = ordered_stage_rows(source_rows, load_jsonl(args.response_pairs_file), "response_pairs")
    selected_balance = source_balance(source_rows, strict_balance)
    constitution = get_constitution()

    preview_rows = []
    for idx, (source_row, pair_row) in enumerate(zip(source_rows[: min(3, len(source_rows))], pair_rows[: min(3, len(pair_rows))])):
        preview_rows.append(
            {
                "index": idx,
                "example_id": source_row["example_id"],
                "preference_prompt": build_preference_prompt(
                    model_family=args.judge_family,
                    constitution=constitution,
                    user_request=source_row["user_request"],
                    tools=source_row["tools"],
                    response_a=pair_row["response_a"],
                    response_b=pair_row["response_b"],
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
                "source_balance": selected_balance,
                "saved_preview": str(out_dir / "prompt_previews.jsonl"),
            },
        )
        return

    tokenizer, model = load_generation_model(
        model_name_or_path=args.judge_model_name_or_path,
        hf_token=args.hf_token,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )

    records: List[Dict[str, Any]] = []
    iterator = progress(
        enumerate(zip(source_rows, pair_rows)),
        total=len(source_rows),
        desc=f"{args.judge_family} preferences",
        leave=False,
    )
    for idx, (source_row, pair_row) in iterator:
        response_a = pair_row["response_a"]
        response_b = pair_row["response_b"]
        failure_reason = None
        preferred_response = None
        rejected_response = None
        chosen_class = None

        if not response_a or not response_b:
            parsed = {"valid": False, "winner": None, "reason": None, "raw_text": ""}
            failure_reason = "empty_candidate"
            valid = False
        elif response_a == response_b:
            parsed = {"valid": False, "winner": None, "reason": None, "raw_text": ""}
            failure_reason = "duplicate_candidates"
            valid = False
        else:
            preference_prompt = build_preference_prompt(
                model_family=args.judge_family,
                constitution=constitution,
                user_request=source_row["user_request"],
                tools=source_row["tools"],
                response_a=response_a,
                response_b=response_b,
                tokenizer=tokenizer,
            )
            first_raw = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=preference_prompt,
                model_family=args.judge_family,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                seed=args.seed + idx,
            )
            parsed = parse_preference_output(first_raw)
            if not parsed["valid"]:
                retry_raw = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=preference_prompt,
                    model_family=args.judge_family,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    seed=args.seed + 1_000_000 + idx,
                )
                parsed = parse_preference_output(retry_raw)
            if not parsed["valid"]:
                failure_reason = "preference_parse_failed"
                valid = False
            else:
                winner = parsed["winner"]
                preferred_response = response_a if winner == "A" else response_b
                rejected_response = response_b if winner == "A" else response_a
                chosen_class = pair_row["response_a_class"] if winner == "A" else pair_row["response_b_class"]
                valid = True

        records.append(
            {
                "example_id": source_row["example_id"],
                "source_row_index": source_row["source_row_index"],
                "source_split": source_row["source_split"],
                "chosen_behavior_class": source_row["chosen_behavior_class"],
                "tools": source_row["tools"],
                "messages": source_row["messages"],
                "response_a": response_a,
                "response_b": response_b,
                "preference": parsed,
                "preferred_response": preferred_response,
                "rejected_response": rejected_response,
                "chosen_class": chosen_class,
                "valid": valid,
                "failure_reason": failure_reason,
            }
        )

    unload_model(tokenizer, model)

    master_path = out_dir / "master_records.jsonl"
    write_jsonl(master_path, records)
    export_dataset = export_rows(records)
    export_path = out_dir / "cai_dpo_dataset.jsonl"
    write_jsonl(export_path, export_dataset)

    save_json(
        out_dir / "summary.json",
        {
            "judge_model_name_or_path": args.judge_model_name_or_path,
            "judge_family": args.judge_family,
            "source_file": args.source_file,
            "response_pairs_file": args.response_pairs_file,
            "source_examples": len(source_rows),
            "source_balance": selected_balance,
            "strict_balance_enforced": strict_balance,
            "outputs": {
                "master_records": str(master_path),
                "export_dataset": str(export_path),
            },
            "result_summary": summarize_records(records),
        },
    )

    export_counts = (
        validate_balanced_counts(export_dataset, "behavior_class")
        if strict_balance and export_dataset
        else count_label_values(export_dataset, "behavior_class") if export_dataset else {}
    )
    if strict_balance and export_counts != selected_balance:
        raise RuntimeError(
            f"Generated CAI DPO dataset is not fully balanced/valid. "
            f"source={selected_balance}, export={export_counts}"
        )


if __name__ == "__main__":
    main()
