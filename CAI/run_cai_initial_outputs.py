import argparse
import os

from cai_stage_utils import (
    add_bool_flag,
    add_run_mode_args,
    count_values,
    env_flag,
    env_float,
    env_int,
    load_existing_stage_records,
    parse_stage_args,
    prepare_stage_source,
    sanitized_args_dict,
)
from cai_utils import (
    append_jsonl,
    build_policy_prompt,
    ensure_dir,
    evaluate_candidate_response,
    generate_responses,
    load_generation_model,
    progress,
    save_json,
    unload_model,
    write_jsonl,
)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate initial CAI policy outputs for a source split.",
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
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", 2))
    parser.add_argument("--max-prompt-length", type=int, default=env_int("MAX_PROMPT_LENGTH", 1024))
    parser.add_argument("--temperature", type=float, default=env_float("POLICY_TEMPERATURE", 0.7))
    parser.add_argument("--top-p", type=float, default=env_float("POLICY_TOP_P", 0.95))
    add_run_mode_args(parser)
    add_bool_flag(parser, "--load-in-4bit", env_flag("LOAD_IN_4BIT", True), "Load model in 4-bit.")
    add_bool_flag(parser, "--trust-remote-code", env_flag("TRUST_REMOTE_CODE", False), "Allow custom model code.")
    return parse_stage_args(parser, argv)

def summarize_records(records):
    return {
        "output_class_counts": count_values(record["initial_output_class"] for record in records),
        "structural_kind_counts": count_values(record["initial_output_structural_kind"] for record in records),
        "valid_rows": sum(bool(record["initial_output_valid"]) for record in records),
        "invalid_reason_counts": count_values(
            record["initial_output_validation_reason"]
            for record in records
            if not record["initial_output_valid"] and record["initial_output_validation_reason"]
        ),
    }

def main(argv=None):
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    out_dir = ensure_dir(args.output_dir)
    save_json(out_dir / "run_config.json", sanitized_args_dict(args))

    source = prepare_stage_source(args, enforce_strict_balance=True)
    source_rows = source.rows

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
                "source_balance": source.balance,
                "saved_preview": str(out_dir / "prompt_previews.jsonl"),
            },
        )
        return

    output_path = out_dir / "initial_outputs.jsonl"
    existing_records, existing_by_id = load_existing_stage_records(output_path, source_rows, "initial_outputs")
    if output_path.exists():
        print(f"Found existing initial outputs file at {output_path} with {len(existing_records)} parsed rows.")
    remaining_rows = [row for row in source_rows if row["example_id"] not in existing_by_id]

    if existing_records:
        print(
            f"Resuming initial outputs from {output_path} with "
            f"{len(existing_records)} completed rows and {len(remaining_rows)} remaining."
        )

    tokenizer = None
    model = None
    if remaining_rows:
        tokenizer, model = load_generation_model(
            model_name_or_path=args.policy_model_name_or_path,
            hf_token=args.hf_token,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            load_in_4bit=args.load_in_4bit,
            trust_remote_code=args.trust_remote_code,
        )

        iterator = progress(
            range(0, len(remaining_rows), args.batch_size),
            total=(len(remaining_rows) + args.batch_size - 1) // args.batch_size,
            desc=f"{args.policy_family} initial outputs",
            leave=False,
        )
        for batch_offset in iterator:
            batch_rows = remaining_rows[batch_offset : batch_offset + args.batch_size]
            prompts = [build_policy_prompt(args.policy_family, row) for row in batch_rows]
            initial_outputs_raw = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                model_family=args.policy_family,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + batch_rows[0]["source_row_index"],
                max_prompt_length=args.max_prompt_length,
            )
            batch_records = []
            for row, initial_output_raw in zip(batch_rows, initial_outputs_raw):
                evaluation = evaluate_candidate_response(initial_output_raw, row["tools"])
                record = {
                    "example_id": row["example_id"],
                    "source_row_index": row["source_row_index"],
                    "source_split": row["source_split"],
                    "chosen_behavior_class": row["chosen_behavior_class"],
                    "tools": row["tools"],
                    "messages": row["messages"],
                    "user_request": row["user_request"],
                    "initial_output_raw": initial_output_raw,
                    "initial_output": evaluation["canonical"],
                    "initial_output_class": evaluation["class"],
                    "initial_output_valid": evaluation["valid"],
                    "initial_output_validation_reason": evaluation["reason"],
                    "initial_output_structural_kind": evaluation["structural_kind"],
                }
                existing_by_id[row["example_id"]] = record
                batch_records.append(record)
            append_jsonl(output_path, batch_records)

        unload_model(tokenizer, model)

    ordered_records = [existing_by_id[row["example_id"]] for row in source_rows]
    write_jsonl(output_path, ordered_records)
    print(f"Wrote complete initial outputs to {output_path} ({len(ordered_records)} rows).")
    save_json(
        out_dir / "summary.json",
        {
            "policy_model_name_or_path": args.policy_model_name_or_path,
            "policy_family": args.policy_family,
            "source_file": args.source_file,
            "source_examples": len(source_rows),
            "source_balance": source.balance,
            "strict_balance_enforced": source.strict_balance,
            "resumed_existing_rows": len(existing_records),
            "outputs": {
                "initial_outputs": str(output_path),
            },
            "result_summary": summarize_records(ordered_records),
        },
    )
    print(f"Wrote initial output summary to {out_dir / 'summary.json'}.")

if __name__ == "__main__":
    main()
