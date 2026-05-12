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
    build_preference_prompt,
    count_label_values,
    ensure_dir,
    generate_responses,
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

def parse_args(argv=None):
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
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", 2))
    parser.add_argument("--max-prompt-length", type=int, default=env_int("MAX_PROMPT_LENGTH", 1024))
    add_run_mode_args(parser)
    add_bool_flag(parser, "--load-in-4bit", env_flag("LOAD_IN_4BIT", True), "Load model in 4-bit.")
    add_bool_flag(parser, "--trust-remote-code", env_flag("TRUST_REMOTE_CODE", False), "Allow custom model code.")
    return parse_stage_args(parser, argv)

def summarize_records(records):
    return {
        "valid_counts": count_values(record["chosen_behavior_class"] for record in records if record.get("valid")),
        "chosen_class_counts": count_values(record["chosen_class"] for record in records if record.get("chosen_class")),
        "failure_counts": count_values(record["failure_reason"] for record in records if record.get("failure_reason")),
    }

def export_rows(records):
    rows = []
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

def main(argv=None):
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    out_dir = ensure_dir(args.output_dir)
    save_json(out_dir / "run_config.json", sanitized_args_dict(args))

    source = prepare_stage_source(args, enforce_strict_balance=True)
    source_rows = source.rows
    pair_rows = ordered_stage_rows(source_rows, load_jsonl(args.response_pairs_file), "response_pairs")
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
                "source_balance": source.balance,
                "saved_preview": str(out_dir / "prompt_previews.jsonl"),
            },
        )
        return

    master_path = out_dir / "master_records.jsonl"
    existing_records, existing_by_id = load_existing_stage_records(master_path, source_rows, "preferences")
    remaining_indices = [
        idx for idx, source_row in enumerate(source_rows) if source_row["example_id"] not in existing_by_id
    ]
    if existing_records:
        print(
            f"Resuming preferences from {master_path} with "
            f"{len(existing_records)} completed rows and {len(remaining_indices)} remaining."
        )

    tokenizer = None
    model = None
    if remaining_indices:
        tokenizer, model = load_generation_model(
            model_name_or_path=args.judge_model_name_or_path,
            hf_token=args.hf_token,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            load_in_4bit=args.load_in_4bit,
            trust_remote_code=args.trust_remote_code,
        )

        judge_tasks = []
        iterator = progress(
            remaining_indices,
            total=len(remaining_indices),
            desc=f"{args.judge_family} preferences",
            leave=False,
        )
        for idx in iterator:
            source_row = source_rows[idx]
            pair_row = pair_rows[idx]
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
                record = {
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
                existing_by_id[source_row["example_id"]] = record
                append_jsonl(master_path, [record])
            elif response_a == response_b:
                parsed = {"valid": False, "winner": None, "reason": None, "raw_text": ""}
                failure_reason = "duplicate_candidates"
                valid = False
                record = {
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
                existing_by_id[source_row["example_id"]] = record
                append_jsonl(master_path, [record])
            else:
                judge_tasks.append(
                    {
                        "record_index": idx,
                        "source_row": source_row,
                        "pair_row": pair_row,
                        "prompt": build_preference_prompt(
                            model_family=args.judge_family,
                            constitution=constitution,
                            user_request=source_row["user_request"],
                            tools=source_row["tools"],
                            response_a=response_a,
                            response_b=response_b,
                            tokenizer=tokenizer,
                        ),
                    }
                )

        retry_tasks = []
        generation_iterator = progress(
            range(0, len(judge_tasks), args.batch_size),
            total=(len(judge_tasks) + args.batch_size - 1) // args.batch_size if judge_tasks else 0,
            desc=f"{args.judge_family} preference first pass",
            leave=False,
        )
        for start_idx in generation_iterator:
            task_batch = judge_tasks[start_idx : start_idx + args.batch_size]
            prompts = [task["prompt"] for task in task_batch]
            raw_outputs = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                model_family=args.judge_family,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                seed=args.seed + task_batch[0]["source_row"]["source_row_index"],
                max_prompt_length=args.max_prompt_length,
            )
            completed_batch_records = []
            for task, raw_output in zip(task_batch, raw_outputs):
                parsed = parse_preference_output(raw_output)
                if parsed["valid"]:
                    pair_row = task["pair_row"]
                    source_row = task["source_row"]
                    winner = parsed["winner"]
                    preferred_response = pair_row["response_a"] if winner == "A" else pair_row["response_b"]
                    rejected_response = pair_row["response_b"] if winner == "A" else pair_row["response_a"]
                    chosen_class = pair_row["response_a_class"] if winner == "A" else pair_row["response_b_class"]
                    record = {
                        "example_id": source_row["example_id"],
                        "source_row_index": source_row["source_row_index"],
                        "source_split": source_row["source_split"],
                        "chosen_behavior_class": source_row["chosen_behavior_class"],
                        "tools": source_row["tools"],
                        "messages": source_row["messages"],
                        "response_a": pair_row["response_a"],
                        "response_b": pair_row["response_b"],
                        "preference": parsed,
                        "preferred_response": preferred_response,
                        "rejected_response": rejected_response,
                        "chosen_class": chosen_class,
                        "valid": True,
                        "failure_reason": None,
                    }
                    existing_by_id[source_row["example_id"]] = record
                    completed_batch_records.append(record)
                else:
                    retry_tasks.append(task)
            append_jsonl(master_path, completed_batch_records)

        retry_iterator = progress(
            range(0, len(retry_tasks), args.batch_size),
            total=(len(retry_tasks) + args.batch_size - 1) // args.batch_size if retry_tasks else 0,
            desc=f"{args.judge_family} preference retries",
            leave=False,
        )
        for start_idx in retry_iterator:
            task_batch = retry_tasks[start_idx : start_idx + args.batch_size]
            prompts = [task["prompt"] for task in task_batch]
            raw_outputs = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                model_family=args.judge_family,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                seed=args.seed + 1_000_000 + task_batch[0]["source_row"]["source_row_index"],
                max_prompt_length=args.max_prompt_length,
            )
            batch_records = []
            for task, raw_output in zip(task_batch, raw_outputs):
                parsed = parse_preference_output(raw_output)
                pair_row = task["pair_row"]
                source_row = task["source_row"]
                if parsed["valid"]:
                    winner = parsed["winner"]
                    preferred_response = pair_row["response_a"] if winner == "A" else pair_row["response_b"]
                    rejected_response = pair_row["response_b"] if winner == "A" else pair_row["response_a"]
                    chosen_class = pair_row["response_a_class"] if winner == "A" else pair_row["response_b_class"]
                    valid = True
                    failure_reason = None
                else:
                    preferred_response = None
                    rejected_response = None
                    chosen_class = None
                    valid = False
                    failure_reason = "preference_parse_failed"
                record = {
                    "example_id": source_row["example_id"],
                    "source_row_index": source_row["source_row_index"],
                    "source_split": source_row["source_split"],
                    "chosen_behavior_class": source_row["chosen_behavior_class"],
                    "tools": source_row["tools"],
                    "messages": source_row["messages"],
                    "response_a": pair_row["response_a"],
                    "response_b": pair_row["response_b"],
                    "preference": parsed,
                    "preferred_response": preferred_response,
                    "rejected_response": rejected_response,
                    "chosen_class": chosen_class,
                    "valid": valid,
                    "failure_reason": failure_reason,
                }
                existing_by_id[source_row["example_id"]] = record
                batch_records.append(record)
            append_jsonl(master_path, batch_records)

        unload_model(tokenizer, model)

    records = [existing_by_id[row["example_id"]] for row in source_rows]
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
            "source_balance": source.balance,
            "strict_balance_enforced": source.strict_balance,
            "resumed_existing_rows": len(existing_records),
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
