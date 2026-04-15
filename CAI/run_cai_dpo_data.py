#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from cai_utils import (
    DEFAULT_CROSS_MODELS,
    DEFAULT_SELF_MODELS,
    build_policy_prompt,
    build_preference_prompt,
    canonicalize_assistant_response,
    count_label_values,
    ensure_dir,
    format_conversation,
    generate_response,
    get_constitution,
    heuristic_class,
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


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name[2:].replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", default=default, help=help_text)
    parser.add_argument(f"--no-{name[2:]}", dest=dest, action="store_false", help=argparse.SUPPRESS)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    if len(raw_argv) >= 4 and raw_argv[2] not in {"self", "cross"} and not raw_argv[2].startswith("-"):
        raw_argv = raw_argv[:2] + ["self"] + raw_argv[2:]

    parser = argparse.ArgumentParser(
        description="Generate CAI DPO training data from AI preferences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("policy_model_name_or_path")
    parser.add_argument("student_family", choices=["llama", "gemma"])
    parser.add_argument("judge_mode", nargs="?", choices=["self", "cross"], default="self")
    parser.add_argument("output_dir")
    parser.add_argument("source_file")
    parser.add_argument("--judge-model-name-or-path", default=None)
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "bfloat16"))
    parser.add_argument("--attn-implementation", default=os.getenv("ATTN_IMPL"))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=env_int("MAX_EXAMPLES", None))
    parser.add_argument("--policy-max-new-tokens", type=int, default=env_int("POLICY_MAX_NEW_TOKENS", 256))
    parser.add_argument("--policy-temperature", type=float, default=env_float("POLICY_TEMPERATURE", 0.8))
    parser.add_argument("--policy-top-p", type=float, default=env_float("POLICY_TOP_P", 0.95))
    parser.add_argument("--judge-max-new-tokens", type=int, default=env_int("JUDGE_MAX_NEW_TOKENS", 64))
    parser.add_argument("--dry-run", action="store_true", default=env_flag("DRY_RUN", False))
    parser.add_argument("--smoke-run", action="store_true", default=env_flag("SMOKE_RUN", False))
    parser.add_argument("--dry-run-max-examples", type=int, default=env_int("DRY_RUN_MAX_EXAMPLES", 8))
    parser.add_argument("--smoke-run-max-examples", type=int, default=env_int("SMOKE_RUN_MAX_EXAMPLES", 6))
    add_bool_flag(parser, "--load-in-4bit", env_flag("LOAD_IN_4BIT", True), "Load models in 4-bit.")
    add_bool_flag(parser, "--trust-remote-code", env_flag("TRUST_REMOTE_CODE", False), "Allow custom model code.")
    args = parser.parse_args(raw_argv)
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


def select_rows(rows: List[Dict[str, Any]], start_index: int, max_examples: Optional[int]) -> List[Dict[str, Any]]:
    selected = rows[start_index:]
    if max_examples is not None:
        selected = selected[:max_examples]
    return selected


def select_smoke_rows(rows: List[Dict[str, Any]], start_index: int, per_class: int = 2) -> List[Dict[str, Any]]:
    selected = rows[start_index:]
    ordered_labels = ["tool_call", "request_for_info", "cannot_answer"]
    buckets: Dict[str, List[Dict[str, Any]]] = {label: [] for label in ordered_labels}
    for row in selected:
        label = row["chosen_behavior_class"]
        if label in buckets and len(buckets[label]) < per_class:
            buckets[label].append(row)

    missing = [label for label, bucket in buckets.items() if len(bucket) < per_class]
    if missing:
        raise ValueError(
            f"Smoke run needs {per_class} examples per class, but could not satisfy: {missing}"
        )

    result: List[Dict[str, Any]] = []
    for label in ordered_labels:
        result.extend(buckets[label])
    return result


def source_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        label = row["chosen_behavior_class"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def should_enforce_strict_balance(args: argparse.Namespace, max_examples: Optional[int]) -> bool:
    return (
        not args.dry_run
        and not args.smoke_run
        and args.start_index == 0
        and max_examples is None
    )


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_counts: Dict[str, int] = {}
    chosen_counts: Dict[str, int] = {}
    failure_counts: Dict[str, int] = {}
    for record in records:
        label = record["chosen_behavior_class"]
        if record.get("valid"):
            valid_counts[label] = valid_counts.get(label, 0) + 1
        chosen_class = record.get("chosen_class")
        if chosen_class:
            chosen_counts[chosen_class] = chosen_counts.get(chosen_class, 0) + 1
        reason = record.get("failure_reason")
        if reason:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
    return {
        "valid_counts": valid_counts,
        "chosen_class_counts": chosen_counts,
        "failure_counts": failure_counts,
    }


def export_dpo_rows(records: List[Dict[str, Any]], student_family: str, judge_mode: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        if not record.get("valid"):
            continue
        rows.append(
            {
                "tools": record["tools"],
                "messages": record["messages"],
                "chosen_response": {"role": "assistant", "content": record["chosen_text"]},
                "rejected_response": {"role": "assistant", "content": record["rejected_text"]},
                "cai_student_family": student_family,
                "cai_branch": judge_mode,
                "behavior_class": record["chosen_behavior_class"],
                "source_row_index": record["source_row_index"],
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = ensure_dir(args.output_dir)
    save_json(out_dir / "run_config.json", sanitized_args_dict(args))

    constitution = get_constitution()
    max_examples = resolve_max_examples(args)
    strict_balance = should_enforce_strict_balance(args, max_examples)
    all_rows = load_jsonl(args.source_file)
    if args.smoke_run and args.max_examples is None:
        source_rows = select_smoke_rows(all_rows, args.start_index, per_class=2)
    else:
        source_rows = select_rows(all_rows, args.start_index, max_examples)
    source_balance = (
        validate_balanced_counts(source_rows, "chosen_behavior_class")
        if strict_balance
        else source_counts(source_rows)
    )

    default_judge = (
        DEFAULT_SELF_MODELS[args.student_family]
        if args.judge_mode == "self"
        else DEFAULT_CROSS_MODELS[args.student_family]
    )
    judge_model_name = args.judge_model_name_or_path or default_judge
    judge_family = args.student_family if args.judge_mode == "self" else ("gemma" if args.student_family == "llama" else "llama")

    preview_rows = []
    for i, row in enumerate(source_rows[: min(3, len(source_rows))]):
        policy_prompt = build_policy_prompt(args.student_family, row)
        preference_prompt = build_preference_prompt(
            model_family=judge_family,
            constitution=constitution,
            user_request=format_conversation(row["messages"]),
            tools=row["tools"],
            response_a="<sample response A>",
            response_b="<sample response B>",
        )
        preview_rows.append(
            {
                "index": i,
                "chosen_behavior_class": row["chosen_behavior_class"],
                "policy_prompt": policy_prompt,
                "preference_prompt": preference_prompt,
            }
        )
    write_jsonl(out_dir / "prompt_previews.jsonl", preview_rows)
    if args.dry_run:
        summary = {
            "mode": "dry_run",
            "source_examples": len(source_rows),
            "source_balance": source_balance,
            "judge_mode": args.judge_mode,
            "judge_model_name_or_path": judge_model_name,
            "saved_preview": str(out_dir / "prompt_previews.jsonl"),
        }
        save_json(out_dir / "dry_run_summary.json", summary)
        return

    records: List[Dict[str, Any]] = []
    policy_tokenizer, policy_model = load_generation_model(
        model_name_or_path=args.policy_model_name_or_path,
        hf_token=args.hf_token,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )
    iterator = progress(
        enumerate(source_rows),
        total=len(source_rows),
        desc=f"{args.student_family} policy samples",
        leave=False,
    )
    for idx, row in iterator:
        policy_prompt = build_policy_prompt(args.student_family, row)
        response_a_raw = generate_response(
            model=policy_model,
            tokenizer=policy_tokenizer,
            prompt=policy_prompt,
            model_family=args.student_family,
            max_new_tokens=args.policy_max_new_tokens,
            do_sample=True,
            temperature=args.policy_temperature,
            top_p=args.policy_top_p,
            seed=args.seed + (idx * 2),
        )
        response_b_raw = generate_response(
            model=policy_model,
            tokenizer=policy_tokenizer,
            prompt=policy_prompt,
            model_family=args.student_family,
            max_new_tokens=args.policy_max_new_tokens,
            do_sample=True,
            temperature=args.policy_temperature,
            top_p=args.policy_top_p,
            seed=args.seed + (idx * 2) + 1,
        )
        response_a = canonicalize_assistant_response(response_a_raw)
        response_b = canonicalize_assistant_response(response_b_raw)
        records.append(
            {
                "source_row_index": idx + args.start_index,
                "tools": row["tools"],
                "messages": row["messages"],
                "chosen_behavior_class": row["chosen_behavior_class"],
                "user_request": format_conversation(row["messages"]),
                "response_a_raw": response_a_raw,
                "response_b_raw": response_b_raw,
                "response_a": response_a,
                "response_b": response_b,
            }
        )
    unload_model(policy_tokenizer, policy_model)

    judge_tokenizer, judge_model = load_generation_model(
        model_name_or_path=judge_model_name,
        hf_token=args.hf_token,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )
    iterator = progress(
        enumerate(records),
        total=len(records),
        desc=f"{args.student_family} {args.judge_mode} preferences",
        leave=False,
    )
    for idx, record in iterator:
        if record["response_a"] == record["response_b"]:
            record["preference"] = {"valid": False, "winner": None, "reason": None, "raw_text": ""}
            record["valid"] = False
            record["failure_reason"] = "duplicate_candidates"
            continue

        preference_prompt = build_preference_prompt(
            model_family=judge_family,
            constitution=constitution,
            user_request=record["user_request"],
            tools=record["tools"],
            response_a=record["response_a"],
            response_b=record["response_b"],
        )
        raw_pref = generate_response(
            model=judge_model,
            tokenizer=judge_tokenizer,
            prompt=preference_prompt,
            model_family=judge_family,
            max_new_tokens=args.judge_max_new_tokens,
            do_sample=False,
            seed=args.seed + 100_000 + idx,
        )
        parsed = parse_preference_output(raw_pref)
        record["preference"] = parsed
        if not parsed["valid"]:
            record["valid"] = False
            record["failure_reason"] = "preference_parse_failed"
            continue

        winner = parsed["winner"]
        chosen_text = record["response_a"] if winner == "A" else record["response_b"]
        rejected_text = record["response_b"] if winner == "A" else record["response_a"]
        chosen_class = heuristic_class(chosen_text)
        record["chosen_text"] = chosen_text
        record["rejected_text"] = rejected_text
        record["chosen_class"] = chosen_class
        record["valid"] = chosen_class == record["chosen_behavior_class"]
        if not record["valid"]:
            record["failure_reason"] = f"class_mismatch:{chosen_class}"
        else:
            record["failure_reason"] = None
    unload_model(judge_tokenizer, judge_model)

    master_path = out_dir / "master_records.jsonl"
    write_jsonl(master_path, records)

    export_rows = export_dpo_rows(records, args.student_family, args.judge_mode)
    export_path = out_dir / f"cai_dpo_{args.student_family}_{args.judge_mode}.jsonl"
    write_jsonl(export_path, export_rows)

    summary = {
        "student_family": args.student_family,
        "policy_model_name_or_path": args.policy_model_name_or_path,
        "judge_mode": args.judge_mode,
        "judge_model_name_or_path": judge_model_name,
        "source_examples": len(source_rows),
        "source_balance": source_balance,
        "strict_balance_enforced": strict_balance,
        "outputs": {
            "master_records": str(master_path),
            "export_dataset": str(export_path),
        },
        "result_summary": summarize_records(records),
    }
    save_json(out_dir / "summary.json", summary)

    if strict_balance:
        export_counts = validate_balanced_counts(export_rows, "behavior_class") if export_rows else {}
    else:
        export_counts = count_label_values(export_rows, "behavior_class") if export_rows else {}
    if strict_balance and export_counts != source_balance:
        raise RuntimeError(
            f"Generated CAI DPO dataset is not fully balanced/valid. "
            f"source={source_balance}, export={export_counts}"
        )


if __name__ == "__main__":
    main()
