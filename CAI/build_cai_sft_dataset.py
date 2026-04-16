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
from cai_utils import count_label_values, ensure_dir, load_jsonl, save_json, write_jsonl


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
    exported_counts: Dict[str, int] = {}
    empty_revised_rows = 0
    revised_structural_kind_counts: Dict[str, int] = {}
    revised_valid_counts: Dict[str, int] = {}
    for record in records:
        label = record["chosen_behavior_class"]
        revised_kind = record.get("revised_output_structural_kind")
        if revised_kind:
            revised_structural_kind_counts[revised_kind] = revised_structural_kind_counts.get(revised_kind, 0) + 1
        revised_valid_key = "valid" if record.get("revised_output_valid") else "invalid"
        revised_valid_counts[revised_valid_key] = revised_valid_counts.get(revised_valid_key, 0) + 1
        if record.get("selected_response"):
            exported_counts[label] = exported_counts.get(label, 0) + 1
        else:
            empty_revised_rows += 1
    return {
        "exported_counts": exported_counts,
        "empty_revised_rows": empty_revised_rows,
        "revised_structural_kind_counts": revised_structural_kind_counts,
        "revised_valid_counts": revised_valid_counts,
    }


def export_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        if not record["selected_response"]:
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
        selected_source = "revision"
        selection_reason = "used_revision_directly"
        selected_response = revision_row["revised_output"]
        selected_valid = revision_row["revised_output_valid"]
        selection_failure_reason = None if selected_response else "empty_revised_output"

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

    save_json(
        out_dir / "export_counts.json",
        {
            "source_balance": selected_balance,
            "export_balance": count_label_values(export_dataset, "behavior_class") if export_dataset else {},
        },
    )


if __name__ == "__main__":
    main()
