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
from cai_utils import count_label_values, ensure_dir, load_jsonl, save_json, validate_balanced_counts, write_jsonl


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.getenv(name)
    return int(value) if value is not None else default


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final CAI SFT dataset from initial outputs, critiques, and revisions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("output_dir")
    parser.add_argument("source_file")
    parser.add_argument("initial_outputs_file")
    parser.add_argument("critiques_file")
    parser.add_argument("revisions_file")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=env_int("MAX_EXAMPLES", None))
    parser.add_argument("--dry-run", action="store_true", default=env_flag("DRY_RUN", False))
    parser.add_argument("--smoke-run", action="store_true", default=env_flag("SMOKE_RUN", False))
    parser.add_argument("--dry-run-max-examples", type=int, default=env_int("DRY_RUN_MAX_EXAMPLES", 8))
    parser.add_argument("--smoke-run-max-examples", type=int, default=env_int("SMOKE_RUN_MAX_EXAMPLES", 6))
    args = parser.parse_args(argv)
    if args.dry_run and args.smoke_run:
        parser.error("--dry-run and --smoke-run are mutually exclusive.")
    return args


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
    heuristic_counts: Dict[str, int] = {}
    failure_counts: Dict[str, int] = {}
    preserved_original_rows = 0
    selected_revision_rows = 0
    for record in records:
        label = record["chosen_behavior_class"]
        if record["selected_response_valid"]:
            valid_counts[label] = valid_counts.get(label, 0) + 1
        response_class = record.get("selected_response_class")
        if response_class:
            heuristic_counts[response_class] = heuristic_counts.get(response_class, 0) + 1
        reason = record.get("selection_failure_reason")
        if reason:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
        if record["selected_source"] == "original":
            preserved_original_rows += 1
        elif record["selected_source"] == "revision":
            selected_revision_rows += 1
    return {
        "valid_counts": valid_counts,
        "heuristic_counts": heuristic_counts,
        "failure_counts": failure_counts,
        "preserved_original_rows": preserved_original_rows,
        "selected_revision_rows": selected_revision_rows,
    }


def export_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        if not record["selected_response_valid"]:
            continue
        rows.append(
            {
                "tools": record["tools"],
                "messages": list(record["messages"])
                + [{"role": "assistant", "content": record["selected_response"]}],
                "source_split": "cai_sft",
                "behavior_class": record["chosen_behavior_class"],
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = ensure_dir(args.output_dir)
    save_json(out_dir / "run_config.json", vars(args))

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
    initial_rows = ordered_stage_rows(source_rows, load_jsonl(args.initial_outputs_file), "initial_outputs")
    critique_rows = ordered_stage_rows(source_rows, load_jsonl(args.critiques_file), "critiques")
    revision_rows = ordered_stage_rows(source_rows, load_jsonl(args.revisions_file), "revisions")
    selected_balance = source_balance(source_rows, strict_balance)

    if args.dry_run:
        save_json(
            out_dir / "dry_run_summary.json",
            {
                "mode": "dry_run",
                "source_examples": len(source_rows),
                "source_balance": selected_balance,
            },
        )
        return

    master_records: List[Dict[str, Any]] = []
    for source_row, initial_row, critique_row, revision_row in zip(source_rows, initial_rows, critique_rows, revision_rows):
        original_score = 2 if initial_row["initial_output_structural_kind"] == "valid_tool_call" else (1 if initial_row["initial_output_structural_kind"] == "plain_text" else 0)
        revision_score = 2 if revision_row["revised_output_structural_kind"] == "valid_tool_call" else (1 if revision_row["revised_output_structural_kind"] == "plain_text" else 0)

        if original_score > revision_score:
            selected_source = "original"
            selection_reason = f"preserved_original:{revision_row['revised_output_structural_kind']}:{revision_row['revised_output_validation_reason']}"
            selected_response = initial_row["initial_output"]
            selected_class = initial_row["initial_output_class"]
            selected_valid = initial_row["initial_output_valid"]
        elif revision_score > original_score:
            selected_source = "revision"
            selection_reason = "used_revision" if revision_row["revised_output_valid"] else f"invalid_revision:{revision_row['revised_output_validation_reason']}"
            selected_response = revision_row["revised_output"]
            selected_class = revision_row["revised_output_class"]
            selected_valid = revision_row["revised_output_valid"]
        else:
            selected_source = "original"
            selection_reason = f"preserved_original:tie:{initial_row['initial_output_structural_kind']}"
            selected_response = initial_row["initial_output"]
            selected_class = initial_row["initial_output_class"]
            selected_valid = initial_row["initial_output_valid"]

        selection_failure_reason = None
        if not selected_response:
            selection_failure_reason = "empty_selected_response"
        elif not selected_valid:
            selection_failure_reason = selection_reason

        master_records.append(
            {
                "example_id": source_row["example_id"],
                "source_row_index": source_row["source_row_index"],
                "source_split": source_row["source_split"],
                "chosen_behavior_class": source_row["chosen_behavior_class"],
                "tools": source_row["tools"],
                "messages": source_row["messages"],
                "user_request": source_row["user_request"],
                "initial_output_raw": initial_row["initial_output_raw"],
                "initial_output": initial_row["initial_output"],
                "initial_output_class": initial_row["initial_output_class"],
                "initial_output_valid": initial_row["initial_output_valid"],
                "initial_output_validation_reason": initial_row["initial_output_validation_reason"],
                "initial_output_structural_kind": initial_row["initial_output_structural_kind"],
                "critique": critique_row["critique"],
                "critique_attempts": critique_row["critique_attempts"],
                "critique_first_try_valid": critique_row["critique_first_try_valid"],
                "critique_final_valid": critique_row["critique_final_valid"],
                "critique_invalid_attempts": critique_row["critique_invalid_attempts"],
                "critique_missing_fields_by_attempt": critique_row["critique_missing_fields_by_attempt"],
                "critique_all_attempts_raw": critique_row["critique_all_attempts_raw"],
                "critique_fallback_used": critique_row["critique_fallback_used"],
                "critique_effective_text": critique_row["critique_effective_text"],
                "revised_output_raw": revision_row["revised_output_raw"],
                "revised_output": revision_row["revised_output"],
                "revised_output_class": revision_row["revised_output_class"],
                "revised_output_valid": revision_row["revised_output_valid"],
                "revised_output_validation_reason": revision_row["revised_output_validation_reason"],
                "revised_output_structural_kind": revision_row["revised_output_structural_kind"],
                "revision_used_original_without_generation": revision_row["revision_used_original_without_generation"],
                "selected_source": selected_source,
                "selection_reason": selection_reason,
                "selected_response": selected_response,
                "selected_response_class": selected_class,
                "selected_response_valid": selected_valid,
                "selection_failure_reason": selection_failure_reason,
            }
        )

    master_path = out_dir / "master_records.jsonl"
    write_jsonl(master_path, master_records)
    export_dataset = export_rows(master_records)
    export_path = out_dir / "cai_sft_dataset.jsonl"
    write_jsonl(export_path, export_dataset)

    save_json(
        out_dir / "summary.json",
        {
            "source_file": args.source_file,
            "initial_outputs_file": args.initial_outputs_file,
            "critiques_file": args.critiques_file,
            "revisions_file": args.revisions_file,
            "source_examples": len(source_rows),
            "source_balance": selected_balance,
            "strict_balance_enforced": strict_balance,
            "outputs": {
                "master_records": str(master_path),
                "export_dataset": str(export_path),
            },
            "result_summary": summarize_records(master_records),
        },
    )

    export_counts = (
        validate_balanced_counts(export_dataset, "behavior_class")
        if strict_balance and export_dataset
        else count_label_values(export_dataset, "behavior_class") if export_dataset else {}
    )
    if strict_balance and export_counts != selected_balance:
        raise RuntimeError(
            f"Generated CAI SFT dataset is not fully balanced/valid. "
            f"source={selected_balance}, export={export_counts}"
        )


if __name__ == "__main__":
    main()
