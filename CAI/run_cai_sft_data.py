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
    count_label_values,
    ensure_dir,
    format_conversation,
    generate_response,
    get_constitution,
    has_tool_call_marker,
    heuristic_class,
    load_generation_model,
    load_jsonl,
    parse_critique_output,
    progress,
    save_json,
    student_display_name,
    unload_model,
    validate_balanced_counts,
    validate_single_tool_call,
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
    parser.add_argument("--smoke-run-max-examples", type=int, default=env_int("SMOKE_RUN_MAX_EXAMPLES", 6))
    add_bool_flag(
        parser,
        "--include-cross",
        env_flag("INCLUDE_CROSS", False),
        "Also generate a cross-model critique/revision dataset.",
    )
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
    if not payload.get("include_cross"):
        payload.pop("cross_model_name_or_path", None)
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


def critique_missing_fields(parsed: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    if parsed.get("verdict") not in {"NO_ISSUES", "ISSUES"}:
        missing.append("Verdict")
    if parsed.get("primary_issue") is None:
        missing.append("Primary Issue")
    elif parsed.get("verdict") == "NO_ISSUES" and parsed.get("primary_issue") != "none":
        missing.append("Primary Issue")
    elif parsed.get("verdict") == "ISSUES" and not parsed.get("valid"):
        missing.append("Primary Issue")
    if not parsed.get("critique"):
        missing.append("Critique")
    return missing


def build_fallback_critique_text(raw_attempts: List[str], parsed_attempts: List[Dict[str, Any]]) -> str:
    non_empty_attempts = [text.strip() for text in raw_attempts if text and text.strip()]
    if not non_empty_attempts:
        return (
            "Critique parsing failed after two attempts.\n"
            "Missing or invalid fields: Verdict, Primary Issue, Critique.\n"
            "Use the constitution to rewrite the response correctly."
        )

    missing_fields = sorted(
        {
            field
            for parsed in parsed_attempts
            if not parsed.get("valid")
            for field in critique_missing_fields(parsed)
        }
    )
    fields_text = ", ".join(missing_fields) if missing_fields else "unknown"
    lines = [
        "Critique parsing failed after two attempts.",
        f"Missing or invalid fields: {fields_text}.",
        "Use the constitution and any useful critique content below to rewrite the response correctly.",
    ]
    for attempt_idx, text in enumerate(non_empty_attempts, start=1):
        lines.append(f"Attempt {attempt_idx} raw critique:")
        lines.append(text)
    return "\n".join(lines)


def evaluate_candidate_response(response_text: str, tools: Any) -> Dict[str, Any]:
    canonical = canonicalize_assistant_response(response_text)
    response_class = heuristic_class(canonical)
    tool_call_like = has_tool_call_marker(response_text) or has_tool_call_marker(canonical)
    tool_validation = validate_single_tool_call(canonical, tools)

    if not canonical:
        valid = False
        reason = "empty_response"
        score = 0
        structural_kind = "empty"
    elif tool_validation["valid"]:
        valid = True
        reason = None
        score = 2
        structural_kind = "valid_tool_call"
    elif tool_call_like:
        valid = False
        reason = tool_validation["reason"]
        score = 0
        structural_kind = "invalid_tool_call"
    else:
        valid = True
        reason = None
        score = 1
        structural_kind = "plain_text"

    return {
        "canonical": canonical,
        "class": response_class,
        "valid": valid,
        "reason": reason,
        "score": score,
        "structural_kind": structural_kind,
        "tool_validation": tool_validation,
    }


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
    desc: str,
) -> None:
    iterator = progress(
        enumerate(records),
        total=len(records),
        desc=desc,
        leave=False,
    )
    for idx, record in iterator:
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
        critique_attempts = [critique_raw]
        parsed_attempts = [parse_critique_output(critique_raw)]
        parsed = parsed_attempts[-1]
        if not parsed["valid"]:
            critique_retry_raw = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=critique_prompt,
                model_family=judge_model_family,
                max_new_tokens=max_new_tokens_critique,
                do_sample=False,
                seed=seed + 1_000_000 + idx,
            )
            critique_attempts.append(critique_retry_raw)
            parsed_attempts.append(parse_critique_output(critique_retry_raw))
            if parsed_attempts[-1]["valid"]:
                parsed = parsed_attempts[-1]

        critique_fallback_used = not parsed["valid"]
        critique_first_try_valid = parsed_attempts[0]["valid"]
        critique_final_valid = parsed["valid"]
        critique_invalid_attempts = sum(1 for attempt in parsed_attempts if not attempt["valid"])
        critique_missing_fields_all = [
            critique_missing_fields(attempt)
            for attempt in parsed_attempts
        ]
        effective_critique_text = (
            parsed["raw_text"]
            if parsed["valid"]
            else build_fallback_critique_text(critique_attempts, parsed_attempts)
        )
        revision_raw = ""
        if parsed["valid"] and parsed["verdict"] == "NO_ISSUES":
            revision_raw = original_response
        else:
            revision_prompt = build_revision_prompt(
                model_family=judge_model_family,
                constitution=constitution,
                user_request=user_request,
                tools=record["tools"],
                assistant_response=original_response,
                critique_text=effective_critique_text,
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

        original_eval = evaluate_candidate_response(original_response, record["tools"])
        revision_eval = evaluate_candidate_response(revision_raw, record["tools"])

        if original_eval["score"] > revision_eval["score"]:
            selected_response = original_eval["canonical"]
            selected_class = original_eval["class"]
            selected_valid = original_eval["valid"]
            selection_source = "original"
            selection_reason = f"preserved_original:{revision_eval['structural_kind']}:{revision_eval['reason']}"
        elif revision_eval["score"] > original_eval["score"]:
            selected_response = revision_eval["canonical"]
            selected_class = revision_eval["class"]
            selected_valid = revision_eval["valid"]
            selection_source = "revision"
            selection_reason = "used_revision" if revision_eval["valid"] else f"invalid_revision:{revision_eval['reason']}"
        else:
            selected_response = original_eval["canonical"]
            selected_class = original_eval["class"]
            selected_valid = original_eval["valid"]
            selection_source = "original"
            selection_reason = f"preserved_original:tie:{original_eval['structural_kind']}"

        failure_reason = None
        if not selected_response:
            failure_reason = "empty_revision"
        elif not selected_valid:
            failure_reason = selection_reason

        record[f"critique_{branch_name}"] = parsed
        record[f"critique_{branch_name}_attempts"] = len(critique_attempts)
        record[f"critique_{branch_name}_first_try_valid"] = critique_first_try_valid
        record[f"critique_{branch_name}_final_valid"] = critique_final_valid
        record[f"critique_{branch_name}_invalid_attempts"] = critique_invalid_attempts
        record[f"critique_{branch_name}_missing_fields_by_attempt"] = critique_missing_fields_all
        record[f"critique_{branch_name}_all_attempts_raw"] = critique_attempts
        record[f"critique_{branch_name}_fallback_used"] = critique_fallback_used
        record[f"critique_{branch_name}_effective_text"] = effective_critique_text
        record[f"revision_{branch_name}_raw"] = revision_raw
        record[f"original_{branch_name}_canonical"] = original_eval["canonical"]
        record[f"original_{branch_name}_class"] = original_eval["class"]
        record[f"original_{branch_name}_valid"] = original_eval["valid"]
        record[f"original_{branch_name}_validation_reason"] = original_eval["reason"]
        record[f"original_{branch_name}_structural_kind"] = original_eval["structural_kind"]
        record[f"revision_{branch_name}_candidate"] = revision_eval["canonical"]
        record[f"revision_{branch_name}_candidate_class"] = revision_eval["class"]
        record[f"revision_{branch_name}_candidate_valid"] = revision_eval["valid"]
        record[f"revision_{branch_name}_candidate_validation_reason"] = revision_eval["reason"]
        record[f"revision_{branch_name}_candidate_structural_kind"] = revision_eval["structural_kind"]
        record[f"revision_{branch_name}"] = selected_response
        record[f"revision_{branch_name}_class"] = selected_class
        record[f"revision_{branch_name}_valid"] = selected_valid
        record[f"revision_{branch_name}_selected_source"] = selection_source
        record[f"revision_{branch_name}_selection_reason"] = selection_reason
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
            }
        )
    return rows


def summarize_branch(records: List[Dict[str, Any]], branch_name: str) -> Dict[str, Any]:
    valid_counts: Dict[str, int] = {}
    heuristic_counts: Dict[str, int] = {}
    failure_counts: Dict[str, int] = {}
    critique_missing_field_counts: Dict[str, int] = {}
    retry_count = 0
    fallback_count = 0
    preserved_original_rows = 0
    first_try_format_fail_rows = 0
    final_format_fail_rows = 0
    invalid_attempt_total = 0
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
        if record.get(f"critique_{branch_name}_attempts", 1) > 1:
            retry_count += 1
        if record.get(f"critique_{branch_name}_fallback_used"):
            fallback_count += 1
        if record.get(f"revision_{branch_name}_selected_source") == "original":
            preserved_original_rows += 1
        if not record.get(f"critique_{branch_name}_first_try_valid", True):
            first_try_format_fail_rows += 1
        if not record.get(f"critique_{branch_name}_final_valid", True):
            final_format_fail_rows += 1
        invalid_attempt_total += int(record.get(f"critique_{branch_name}_invalid_attempts", 0))
        for missing_fields in record.get(f"critique_{branch_name}_missing_fields_by_attempt", []):
            for field in missing_fields:
                critique_missing_field_counts[field] = critique_missing_field_counts.get(field, 0) + 1
    return {
        "valid_counts": valid_counts,
        "heuristic_counts": heuristic_counts,
        "failure_counts": failure_counts,
        "critique_retry_rows": retry_count,
        "critique_fallback_rows": fallback_count,
        "preserved_original_rows": preserved_original_rows,
        "critique_first_try_format_fail_rows": first_try_format_fail_rows,
        "critique_final_format_fail_rows": final_format_fail_rows,
        "critique_invalid_attempt_total": invalid_attempt_total,
        "critique_missing_field_counts": critique_missing_field_counts,
    }


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
    cross_model_name = args.cross_model_name_or_path or DEFAULT_CROSS_MODELS[args.student_family]

    preview_rows = build_preview_rows(source_rows, args, constitution)
    write_jsonl(out_dir / "prompt_previews.jsonl", preview_rows)
    if args.dry_run:
        summary = {
            "mode": "dry_run",
            "source_examples": len(source_rows),
            "source_balance": source_balance,
            "saved_preview": str(out_dir / "prompt_previews.jsonl"),
            "include_cross": args.include_cross,
        }
        if args.include_cross:
            summary["cross_model_name_or_path"] = cross_model_name
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
    prompt_desc = f"{student_display} original responses"
    iterator = progress(
        enumerate(source_rows),
        total=len(source_rows),
        desc=prompt_desc,
        leave=False,
    )
    for idx, row in iterator:
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
                "tools": row["tools"],
                "messages": row["messages"],
                "chosen_behavior_class": row["chosen_behavior_class"],
                "user_request": format_conversation(row["messages"]),
                "policy_prompt": policy_prompt,
                "original_response": original_response,
                "original_response_class": heuristic_class(original_response),
            }
        )
        if args.include_cross:
            records[-1]["cross_model_name_or_path"] = cross_model_name

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
        desc=f"{student_display} self critiques/revisions",
    )
    unload_model(student_tokenizer, student_model)

    if args.include_cross:
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
            desc=f"{student_display} cross critiques/revisions",
        )
        unload_model(cross_tokenizer, cross_model)

    master_path = out_dir / "master_records.jsonl"
    write_jsonl(master_path, records)

    self_rows = export_branch_rows(records, "self", args.student_family)
    self_path = out_dir / f"cai_sft_{args.student_family}_self.jsonl"
    write_jsonl(self_path, self_rows)
    cross_rows: List[Dict[str, Any]] = []
    cross_path: Optional[Path] = None
    if args.include_cross:
        cross_rows = export_branch_rows(records, "cross", args.student_family)
        cross_path = out_dir / f"cai_sft_{args.student_family}_cross.jsonl"
        write_jsonl(cross_path, cross_rows)

    summary = {
        "student_family": args.student_family,
        "student_model_name_or_path": args.student_model_name_or_path,
        "include_cross": args.include_cross,
        "source_examples": len(source_rows),
        "source_balance": source_balance,
        "strict_balance_enforced": strict_balance,
        "outputs": {
            "master_records": str(master_path),
            "self_dataset": str(self_path),
        },
        "self": summarize_branch(records, "self"),
    }
    if args.include_cross:
        summary["cross_model_name_or_path"] = cross_model_name
        summary["outputs"]["cross_dataset"] = str(cross_path) if cross_path is not None else None
        summary["cross"] = summarize_branch(records, "cross")
    save_json(out_dir / "summary.json", summary)

    if strict_balance:
        self_counts = validate_balanced_counts(self_rows, "behavior_class") if self_rows else {}
        cross_counts = validate_balanced_counts(cross_rows, "behavior_class") if cross_rows else {}
    else:
        self_counts = count_label_values(self_rows, "behavior_class") if self_rows else {}
        cross_counts = count_label_values(cross_rows, "behavior_class") if cross_rows else {}
    if strict_balance and self_counts != source_balance:
        raise RuntimeError(
            f"Generated CAI SFT datasets are not fully balanced/valid. "
            f"source={source_balance}, self={self_counts}"
        )
    if strict_balance and args.include_cross and cross_counts != source_balance:
        raise RuntimeError(
            f"Generated CAI SFT cross dataset is not fully balanced/valid. "
            f"source={source_balance}, cross={cross_counts}"
        )


if __name__ == "__main__":
    main()
