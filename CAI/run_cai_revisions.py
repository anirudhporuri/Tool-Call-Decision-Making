import argparse
import os

from cai_stage_utils import (
    add_bool_flag,
    add_run_mode_args,
    count_values,
    env_flag,
    env_int,
    load_existing_stage_records,
    ordered_stage_rows,
    parse_stage_args,
    prepare_stage_source,
    sanitized_args_dict,
)
from cai_utils import (
    append_jsonl,
    build_revision_prompt,
    ensure_dir,
    evaluate_candidate_response,
    generate_responses,
    get_constitution,
    load_generation_model,
    load_jsonl,
    progress,
    save_json,
    unload_model,
    write_jsonl,
)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate CAI revisions from initial outputs and critiques.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("revision_model_name_or_path")
    parser.add_argument("revision_family", choices=["llama", "gemma", "qwen", "gpt-oss"])
    parser.add_argument("output_dir")
    parser.add_argument("source_file")
    parser.add_argument("initial_outputs_file")
    parser.add_argument("critiques_file")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "bfloat16"))
    parser.add_argument("--attn-implementation", default=os.getenv("ATTN_IMPL"))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=env_int("MAX_EXAMPLES", None))
    parser.add_argument("--max-new-tokens", type=int, default=env_int("REVISION_MAX_NEW_TOKENS", 256))
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", 2))
    parser.add_argument("--max-prompt-length", type=int, default=env_int("MAX_PROMPT_LENGTH", 1024))
    add_run_mode_args(parser)
    add_bool_flag(parser, "--load-in-4bit", env_flag("LOAD_IN_4BIT", True), "Load model in 4-bit.")
    add_bool_flag(parser, "--trust-remote-code", env_flag("TRUST_REMOTE_CODE", False), "Allow custom model code.")
    return parse_stage_args(parser, argv)

def summarize_records(records):
    return {
        "revision_class_counts": count_values(record["revised_output_class"] for record in records),
        "structural_kind_counts": count_values(record["revised_output_structural_kind"] for record in records),
        "used_original_without_generation_rows": sum(
            bool(record["revision_used_original_without_generation"]) for record in records
        ),
        "invalid_reason_counts": count_values(
            record["revised_output_validation_reason"]
            for record in records
            if not record["revised_output_valid"] and record["revised_output_validation_reason"]
        ),
    }

def main(argv=None):
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    out_dir = ensure_dir(args.output_dir)
    save_json(out_dir / "run_config.json", sanitized_args_dict(args))

    source_rows = prepare_stage_source(args).rows
    initial_rows = ordered_stage_rows(source_rows, load_jsonl(args.initial_outputs_file), "initial_outputs")
    critique_rows = ordered_stage_rows(source_rows, load_jsonl(args.critiques_file), "critiques")
    constitution = get_constitution()

    preview_rows = []
    for idx, (source_row, initial_row, critique_row) in enumerate(
        zip(
            source_rows[: min(3, len(source_rows))],
            initial_rows[: min(3, len(initial_rows))],
            critique_rows[: min(3, len(critique_rows))],
        )
    ):
        preview_rows.append(
            {
                "index": idx,
                "example_id": source_row["example_id"],
                "revision_prompt": build_revision_prompt(
                    model_family=args.revision_family,
                    constitution=constitution,
                    user_request=source_row["user_request"],
                    tools=source_row["tools"],
                    assistant_response=initial_row.get("initial_output_raw") or initial_row["initial_output"],
                    critique_text=critique_row["critique_effective_text"],
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
                "saved_preview": str(out_dir / "prompt_previews.jsonl"),
            },
        )
        return

    output_path = out_dir / "revisions.jsonl"
    existing_records, existing_by_id = load_existing_stage_records(output_path, source_rows, "revisions")
    remaining_indices = [
        idx for idx, source_row in enumerate(source_rows) if source_row["example_id"] not in existing_by_id
    ]
    if existing_records:
        print(
            f"Resuming revisions from {output_path} with "
            f"{len(existing_records)} completed rows and {len(remaining_indices)} remaining."
        )

    tokenizer = None
    model = None
    if remaining_indices:
        tokenizer, model = load_generation_model(
            model_name_or_path=args.revision_model_name_or_path,
            hf_token=args.hf_token,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            load_in_4bit=args.load_in_4bit,
            trust_remote_code=args.trust_remote_code,
        )

        generation_tasks = []
        iterator = progress(
            remaining_indices,
            total=len(remaining_indices),
            desc=f"{args.revision_family} revisions",
            leave=False,
        )
        for idx in iterator:
            source_row = source_rows[idx]
            initial_row = initial_rows[idx]
            critique_row = critique_rows[idx]
            original_response = initial_row.get("initial_output_raw") or initial_row["initial_output"]
            parsed_critique = critique_row["critique"]
            used_original_without_generation = bool(
                parsed_critique.get("valid") and parsed_critique.get("verdict") == "NO_ISSUES"
            )
            if used_original_without_generation:
                revision_raw = original_response
                evaluation = evaluate_candidate_response(revision_raw, source_row["tools"])
                record = {
                    "example_id": source_row["example_id"],
                    "source_row_index": source_row["source_row_index"],
                    "source_split": source_row["source_split"],
                    "chosen_behavior_class": source_row["chosen_behavior_class"],
                    "revision_used_original_without_generation": used_original_without_generation,
                    "revised_output_raw": revision_raw,
                    "revised_output": evaluation["canonical"],
                    "revised_output_class": evaluation["class"],
                    "revised_output_valid": evaluation["valid"],
                    "revised_output_validation_reason": evaluation["reason"],
                    "revised_output_structural_kind": evaluation["structural_kind"],
                }
                existing_by_id[source_row["example_id"]] = record
                append_jsonl(output_path, [record])
            else:
                generation_tasks.append(
                    {
                        "record_index": idx,
                        "source_row": source_row,
                        "used_original_without_generation": used_original_without_generation,
                        "prompt": build_revision_prompt(
                            model_family=args.revision_family,
                            constitution=constitution,
                            user_request=source_row["user_request"],
                            tools=source_row["tools"],
                            assistant_response=original_response,
                            critique_text=critique_row["critique_effective_text"],
                            tokenizer=tokenizer,
                        ),
                    }
                )

        generation_iterator = progress(
            range(0, len(generation_tasks), args.batch_size),
            total=(len(generation_tasks) + args.batch_size - 1) // args.batch_size if generation_tasks else 0,
            desc=f"{args.revision_family} revision generations",
            leave=False,
        )
        for start_idx in generation_iterator:
            task_batch = generation_tasks[start_idx : start_idx + args.batch_size]
            prompts = [task["prompt"] for task in task_batch]
            revision_outputs = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                model_family=args.revision_family,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                seed=args.seed + task_batch[0]["source_row"]["source_row_index"],
                max_prompt_length=args.max_prompt_length,
            )
            batch_records = []
            for task, revision_raw in zip(task_batch, revision_outputs):
                source_row = task["source_row"]
                evaluation = evaluate_candidate_response(revision_raw, source_row["tools"])
                record = {
                    "example_id": source_row["example_id"],
                    "source_row_index": source_row["source_row_index"],
                    "source_split": source_row["source_split"],
                    "chosen_behavior_class": source_row["chosen_behavior_class"],
                    "revision_used_original_without_generation": task["used_original_without_generation"],
                    "revised_output_raw": revision_raw,
                    "revised_output": evaluation["canonical"],
                    "revised_output_class": evaluation["class"],
                    "revised_output_valid": evaluation["valid"],
                    "revised_output_validation_reason": evaluation["reason"],
                    "revised_output_structural_kind": evaluation["structural_kind"],
                }
                existing_by_id[source_row["example_id"]] = record
                batch_records.append(record)
            append_jsonl(output_path, batch_records)

        unload_model(tokenizer, model)

    records = [existing_by_id[row["example_id"]] for row in source_rows]
    write_jsonl(output_path, records)
    save_json(
        out_dir / "summary.json",
        {
            "revision_model_name_or_path": args.revision_model_name_or_path,
            "revision_family": args.revision_family,
            "source_file": args.source_file,
            "initial_outputs_file": args.initial_outputs_file,
            "critiques_file": args.critiques_file,
            "source_examples": len(source_rows),
            "resumed_existing_rows": len(existing_records),
            "outputs": {
                "revisions": str(output_path),
            },
            "result_summary": summarize_records(records),
        },
    )

if __name__ == "__main__":
    main()
