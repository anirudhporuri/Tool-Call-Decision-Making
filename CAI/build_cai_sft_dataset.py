import argparse

from cai_stage_utils import (
    add_run_mode_args,
    count_values,
    env_int,
    ordered_stage_rows,
    parse_stage_args,
    prepare_stage_source,
)
from cai_utils import count_label_values, ensure_dir, load_jsonl, save_json, write_jsonl

def parse_args(argv=None):
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
    add_run_mode_args(parser)
    return parse_stage_args(parser, argv)

def summarize_records(records):
    return {
        "exported_counts": count_values(
            record["chosen_behavior_class"] for record in records if record.get("selected_response")
        ),
        "empty_revised_rows": sum(not record.get("selected_response") for record in records),
        "revised_structural_kind_counts": count_values(
            record["revised_output_structural_kind"]
            for record in records
            if record.get("revised_output_structural_kind")
        ),
        "revised_valid_counts": count_values(
            "valid" if record.get("revised_output_valid") else "invalid" for record in records
        ),
    }

def export_rows(records):
    rows = []
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

def main(argv=None):
    args = parse_args(argv)
    out_dir = ensure_dir(args.output_dir)
    save_json(out_dir / "run_config.json", vars(args))

    source = prepare_stage_source(args, enforce_strict_balance=True)
    source_rows = source.rows
    initial_rows = ordered_stage_rows(source_rows, load_jsonl(args.initial_outputs_file), "initial_outputs")
    critique_rows = ordered_stage_rows(source_rows, load_jsonl(args.critiques_file), "critiques")
    revision_rows = ordered_stage_rows(source_rows, load_jsonl(args.revisions_file), "revisions")

    if args.dry_run:
        save_json(
            out_dir / "dry_run_summary.json",
            {
                "mode": "dry_run",
                "source_examples": len(source_rows),
                "source_balance": source.balance,
            },
        )
        return

    master_records = []
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
            "source_balance": source.balance,
            "strict_balance_enforced": source.strict_balance,
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
            "source_balance": source.balance,
            "export_balance": count_label_values(export_dataset, "behavior_class") if export_dataset else {},
        },
    )

if __name__ == "__main__":
    main()
