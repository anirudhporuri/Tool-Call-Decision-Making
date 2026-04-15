#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from cai_utils import (
    DEFAULT_CROSS_MODELS,
    build_critique_prompt,
    build_policy_prompt,
    build_revision_prompt,
    canonicalize_assistant_response,
    ensure_dir,
    format_conversation,
    generate_response,
    get_constitution,
    heuristic_class,
    load_generation_model,
    load_jsonl,
    parse_critique_output,
    save_json,
    student_display_name,
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
    parser = argparse.ArgumentParser(
        description="Generate CAI SFT training data from constitution-guided revisions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("student_model_name_or_path")
    parser.add_argument("student_family", choices=["llama", "gemma"])
    parser.add_argument("output_dir")
    parser.add_argument("source_file")
    parser.add_argument("--cross-model-name-or-path", default=None)
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "bfloat16"))
    parser.add_argument("--attn-implementation", default=os.getenv("ATTN_IMPL"))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=env_int("MAX_EXAMPLES", None))
    parser.add_argument("--policy-max-new-tokens", type=int, default=env_int("POLICY_MAX_NEW_TOKENS", 256))
    parser.add_argument("--policy-temperature", type=float, default=env_float("POLICY_TEMPERATURE", 0.7))
    parser.add_argument("--policy-top-p", type=float, default=env_float("POLICY_TOP_P", 0.95))
    parser.add_argument("--critique-max-new-tokens", type=int, default=env_int("CRITIQUE_MAX_NEW_TOKENS", 96))
    parser.add_argument("--revision-max-new-tokens", type=int, default=env_int("REVISION_MAX_NEW_TOKENS", 256))
    parser.add_argument("--dry-run", action="store_true", default=env_flag("DRY_RUN", False))
    parser.add_argument("--smoke-run", action="store_true", default=env_flag("SMOKE_RUN", False))
    parser.add_argument("--dry-run-max-examples", type=int, default=env_int("DRY_RUN_MAX_EXAMPLES", 8))
    parser.add_argument("--smoke-run-max-examples", type=int, default=env_int("SMOKE_RUN_MAX_EXAMPLES", 8))
    add_bool_flag(parser, "--load-in-4bit", env_flag("LOAD_IN_4BIT", True), "Load models in 4-bit.")
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


def select_rows(rows: List[Dict[str, Any]], start_index: int, max_examples: Optional[int]) -> List[Dict[str, Any]]:
    selected = rows[start_index:]
    if max_examples is not None:
        selected = selected[:max_examples]
    return selected


def source_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        label = row["chosen_behavior_class"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def build_preview_rows(rows: List[Dict[str, Any]], args: argparse.Namespace, constitution: str) -> List[Dict[str, Any]]:
    preview_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows[: min(3, len(rows))]):
        user_request = format_conversation(row["messages"])
        original_response = "<sample original response>"
        critique_prompt = build_critique_prompt(
            model_family=args.student_family,
            constitution=constitution,
            user_request=user_request,
            tools=row["tools"],
            assistant_response=original_response,
        )
        revision_prompt = build_revision_prompt(
            model_family=args.student_family,
            constitution=constitution,
            user_request=user_request,
            tools=row["tools"],
            assistant_response=original_response,
            critique_text="Verdict: ISSUES\nPrimary Issue: wrong_tool\nCritique: The response used the wrong action.",
        )
        preview_rows.append(
            {
                "index": i,
                "chosen_behavior_class": row["chosen_behavior_class"],
                "policy_prompt": build_policy_prompt(args.student_family, row),
                "critique_prompt": critique_prompt,
                "revision_prompt": revision_prompt,
            }
        )
    return preview_rows


def process_revision_branch(
    *,
    records: List[Dict[str, Any]],
    branch_name: str,
    judge_model_family: str,
    tokenizer: Any,
    model: Any,
    constitution: str,
    max_new_tokens_critique: int,
    max_new_tokens_revision: int,
    seed: int,
) -> None:
    for idx, record in enumerate(records):
        user_request = record["user_request"]
        original_response = record["original_response"]

        critique_prompt = build_critique_prompt(
            model_family=judge_model_family,
            constitution=constitution,
            user_request=user_request,
            tools=record["tools"],
            assistant_response=original_response,
        )
        critique_raw = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=critique_prompt,
            model_family=judge_model_family,
            max_new_tokens=max_new_tokens_critique,
            do_sample=False,
            seed=seed + idx,
        )
        parsed = parse_critique_output(critique_raw)
        revision_raw = ""
        if parsed["valid"] and parsed["verdict"] == "NO_ISSUES":
            revision_raw = original_response
        elif parsed["valid"]:
            revision_prompt = build_revision_prompt(
                model_family=judge_model_family,
                constitution=constitution,
                user_request=user_request,
                tools=record["tools"],
                assistant_response=original_response,
                critique_text=parsed["raw_text"],
            )
            revision_raw = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=revision_prompt,
                model_family=judge_model_family,
                max_new_tokens=max_new_tokens_revision,
                do_sample=False,
                seed=seed + 10_000 + idx,
            )

        revision = canonicalize_assistant_response(revision_raw)
        revision_class = heuristic_class(revision)
        valid = bool(parsed["valid"]) and bool(revision) and revision_class == record["chosen_behavior_class"]
        failure_reason = None
        if not parsed["valid"]:
            failure_reason = "critique_parse_failed"
        elif not revision:
            failure_reason = "empty_revision"
        elif revision_class != record["chosen_behavior_class"]:
            failure_reason = f"class_mismatch:{revision_class}"

        record[f"critique_{branch_name}"] = parsed
        record[f"revision_{branch_name}_raw"] = revision_raw
        record[f"revision_{branch_name}"] = revision
        record[f"revision_{branch_name}_class"] = revision_class
        record[f"revision_{branch_name}_valid"] = valid
        record[f"revision_{branch_name}_failure_reason"] = failure_reason


def export_branch_rows(records: List[Dict[str, Any]], branch_name: str, student_family: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        if not record.get(f"revision_{branch_name}_valid"):
            continue
        rows.append(
            {
                "tools": record["tools"],
                "messages": list(record["messages"])
                + [{"role": "assistant", "content": record[f"revision_{branch_name}"]}],
                "source_split": "cai_sft",
                "behavior_class": record["chosen_behavior_class"],
                "cai_student_family": student_family,
                "cai_branch": branch_name,
                "source_row_index": record["source_row_index"],
            }
        )
    return rows


def summarize_branch(records: List[Dict[str, Any]], branch_name: str) -> Dict[str, Any]:
    valid_counts: Dict[str, int] = {}
    heuristic_counts: Dict[str, int] = {}
    failure_counts: Dict[str, int] = {}
    for record in records:
        label = record["chosen_behavior_class"]
        if record.get(f"revision_{branch_name}_valid"):
            valid_counts[label] = valid_counts.get(label, 0) + 1
        cls = record.get(f"revision_{branch_name}_class")
        if cls:
            heuristic_counts[cls] = heuristic_counts.get(cls, 0) + 1
        reason = record.get(f"revision_{branch_name}_failure_reason")
        if reason:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
    return {
        "valid_counts": valid_counts,
        "heuristic_counts": heuristic_counts,
        "failure_counts": failure_counts,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = ensure_dir(args.output_dir)
    save_json(out_dir / "run_config.json", sanitized_args_dict(args))

    constitution = get_constitution()
    source_rows = select_rows(load_jsonl(args.source_file), args.start_index, resolve_max_examples(args))
    source_balance = validate_balanced_counts(source_rows, "chosen_behavior_class")
    cross_model_name = args.cross_model_name_or_path or DEFAULT_CROSS_MODELS[args.student_family]

    preview_rows = build_preview_rows(source_rows, args, constitution)
    write_jsonl(out_dir / "prompt_previews.jsonl", preview_rows)
    if args.dry_run:
        summary = {
            "mode": "dry_run",
            "source_examples": len(source_rows),
            "source_balance": source_balance,
            "saved_preview": str(out_dir / "prompt_previews.jsonl"),
            "cross_model_name_or_path": cross_model_name,
        }
        save_json(out_dir / "dry_run_summary.json", summary)
        return

    records: List[Dict[str, Any]] = []
    student_display = student_display_name(args.student_family)

    student_tokenizer, student_model = load_generation_model(
        model_name_or_path=args.student_model_name_or_path,
        hf_token=args.hf_token,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )
    for idx, row in enumerate(source_rows):
        policy_prompt = build_policy_prompt(args.student_family, row)
        original_response = generate_response(
            model=student_model,
            tokenizer=student_tokenizer,
            prompt=policy_prompt,
            model_family=args.student_family,
            max_new_tokens=args.policy_max_new_tokens,
            do_sample=True,
            temperature=args.policy_temperature,
            top_p=args.policy_top_p,
            seed=args.seed + idx,
        )
        records.append(
            {
                "source_row_index": idx + args.start_index,
                "student_family": args.student_family,
                "student_model_name_or_path": args.student_model_name_or_path,
                "cross_model_name_or_path": cross_model_name,
                "tools": row["tools"],
                "messages": row["messages"],
                "chosen_behavior_class": row["chosen_behavior_class"],
                "user_request": format_conversation(row["messages"]),
                "policy_prompt": policy_prompt,
                "original_response": original_response,
                "original_response_class": heuristic_class(original_response),
            }
        )

    process_revision_branch(
        records=records,
        branch_name="self",
        judge_model_family=args.student_family,
        tokenizer=student_tokenizer,
        model=student_model,
        constitution=constitution,
        max_new_tokens_critique=args.critique_max_new_tokens,
        max_new_tokens_revision=args.revision_max_new_tokens,
        seed=args.seed,
    )
    unload_model(student_tokenizer, student_model)

    cross_family = "gemma" if args.student_family == "llama" else "llama"
    cross_tokenizer, cross_model = load_generation_model(
        model_name_or_path=cross_model_name,
        hf_token=args.hf_token,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )
    process_revision_branch(
        records=records,
        branch_name="cross",
        judge_model_family=cross_family,
        tokenizer=cross_tokenizer,
        model=cross_model,
        constitution=constitution,
        max_new_tokens_critique=args.critique_max_new_tokens,
        max_new_tokens_revision=args.revision_max_new_tokens,
        seed=args.seed + 100_000,
    )
    unload_model(cross_tokenizer, cross_model)

    master_path = out_dir / "master_records.jsonl"
    write_jsonl(master_path, records)

    self_rows = export_branch_rows(records, "self", args.student_family)
    cross_rows = export_branch_rows(records, "cross", args.student_family)
    self_path = out_dir / f"cai_sft_{args.student_family}_self.jsonl"
    cross_path = out_dir / f"cai_sft_{args.student_family}_cross.jsonl"
    write_jsonl(self_path, self_rows)
    write_jsonl(cross_path, cross_rows)

    summary = {
        "student_family": args.student_family,
        "student_model_name_or_path": args.student_model_name_or_path,
        "cross_model_name_or_path": cross_model_name,
        "source_examples": len(source_rows),
        "source_balance": source_balance,
        "outputs": {
            "master_records": str(master_path),
            "self_dataset": str(self_path),
            "cross_dataset": str(cross_path),
        },
        "self": summarize_branch(records, "self"),
        "cross": summarize_branch(records, "cross"),
    }
    save_json(out_dir / "summary.json", summary)

    self_counts = validate_balanced_counts(self_rows, "behavior_class")
    cross_counts = validate_balanced_counts(cross_rows, "behavior_class")
    if self_counts != source_balance or cross_counts != source_balance:
        raise RuntimeError(
            f"Generated CAI SFT datasets are not fully balanced/valid. "
            f"source={source_balance}, self={self_counts}, cross={cross_counts}"
        )


if __name__ == "__main__":
    main()
