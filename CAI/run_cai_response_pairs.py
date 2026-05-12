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
        description="Generate CAI DPO response pairs for a source split.",
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
    parser.add_argument("--temperature", type=float, default=env_float("POLICY_TEMPERATURE", 0.9))
    parser.add_argument("--top-p", type=float, default=env_float("POLICY_TOP_P", 0.95))
    parser.add_argument(
        "--candidate-attempts",
        type=int,
        default=env_int("PAIR_CANDIDATE_ATTEMPTS", 6),
        help="Maximum number of sampled attempts per prompt while searching for two distinct canonical responses.",
    )
    add_run_mode_args(parser)
    add_bool_flag(parser, "--load-in-4bit", env_flag("LOAD_IN_4BIT", True), "Load model in 4-bit.")
    add_bool_flag(parser, "--trust-remote-code", env_flag("TRUST_REMOTE_CODE", False), "Allow custom model code.")
    return parse_stage_args(parser, argv)

def summarize_records(records):
    return {
        "response_a_class_counts": count_values(record["response_a_class"] for record in records),
        "response_b_class_counts": count_values(record["response_b_class"] for record in records),
        "duplicate_rows": sum(record["response_a"] == record["response_b"] for record in records),
        "resampled_rows": sum(record.get("sampling_attempts", 0) > 2 for record in records),
        "max_sampling_attempts": max((int(record.get("sampling_attempts", 0)) for record in records), default=0),
    }

def build_response_pair_record(state):
    row = state["row"]
    sampled_candidates = state["sampled_candidates"]
    distinct_candidates = state["distinct_candidates"]
    if not sampled_candidates:
        raise RuntimeError(f"No sampled candidates were generated for example_id={row['example_id']}.")
    response_a = distinct_candidates[0] if distinct_candidates else sampled_candidates[0]
    response_b = distinct_candidates[1] if len(distinct_candidates) >= 2 else sampled_candidates[-1]
    return {
        "example_id": row["example_id"],
        "source_row_index": row["source_row_index"],
        "source_split": row["source_split"],
        "chosen_behavior_class": row["chosen_behavior_class"],
        "tools": row["tools"],
        "messages": row["messages"],
        "user_request": row["user_request"],
        "sampling_attempts": len(sampled_candidates),
        "distinct_candidates_found": len(distinct_candidates),
        "response_a_raw": response_a["raw"],
        "response_a": response_a["canonical"],
        "response_a_class": response_a["class"],
        "response_a_valid": response_a["valid"],
        "response_a_validation_reason": response_a["validation_reason"],
        "response_a_structural_kind": response_a["structural_kind"],
        "response_b_raw": response_b["raw"],
        "response_b": response_b["canonical"],
        "response_b_class": response_b["class"],
        "response_b_valid": response_b["valid"],
        "response_b_validation_reason": response_b["validation_reason"],
        "response_b_structural_kind": response_b["structural_kind"],
    }

def main(argv=None):
    args = parse_args(argv)
    if args.candidate_attempts < 2:
        raise ValueError("--candidate-attempts must be at least 2.")
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

    output_path = out_dir / "response_pairs.jsonl"
    existing_records, existing_by_id = load_existing_stage_records(output_path, source_rows, "response_pairs")
    remaining_rows = [row for row in source_rows if row["example_id"] not in existing_by_id]
    if existing_records:
        print(
            f"Resuming response pairs from {output_path} with "
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

        states = []
        for idx, row in enumerate(remaining_rows):
            states.append(
                {
                    "row_index": idx,
                    "row": row,
                    "prompt": build_policy_prompt(args.policy_family, row),
                    "sampled_candidates": [],
                    "distinct_candidates": [],
                    "seen_canonicals": set(),
                    "completed": False,
                }
            )

        attempt_iterator = progress(
            range(args.candidate_attempts),
            total=args.candidate_attempts,
            desc=f"{args.policy_family} response pair attempts",
            leave=False,
        )
        for attempt_idx in attempt_iterator:
            active_state_indices = [
                idx for idx, state in enumerate(states) if not state["completed"] and len(state["distinct_candidates"]) < 2
            ]
            if not active_state_indices:
                break
            batch_iterator = progress(
                range(0, len(active_state_indices), args.batch_size),
                total=(len(active_state_indices) + args.batch_size - 1) // args.batch_size,
                desc=f"{args.policy_family} response pair batches",
                leave=False,
            )
            for batch_start in batch_iterator:
                state_index_batch = active_state_indices[batch_start : batch_start + args.batch_size]
                prompt_batch = [states[state_idx]["prompt"] for state_idx in state_index_batch]
                raw_outputs = generate_responses(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompt_batch,
                    model_family=args.policy_family,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed + (attempt_idx * 100_000) + batch_start,
                    max_prompt_length=args.max_prompt_length,
                )
                completed_batch_records = []
                for state_idx, response_raw in zip(state_index_batch, raw_outputs):
                    state = states[state_idx]
                    evaluation = evaluate_candidate_response(response_raw, state["row"]["tools"])
                    candidate = {
                        "raw": response_raw,
                        "canonical": evaluation["canonical"],
                        "class": evaluation["class"],
                        "valid": evaluation["valid"],
                        "validation_reason": evaluation["reason"],
                        "structural_kind": evaluation["structural_kind"],
                    }
                    state["sampled_candidates"].append(candidate)
                    canonical_key = candidate["canonical"]
                    if canonical_key not in state["seen_canonicals"]:
                        state["seen_canonicals"].add(canonical_key)
                        state["distinct_candidates"].append(candidate)
                    if len(state["distinct_candidates"]) >= 2 and not state["completed"]:
                        record = build_response_pair_record(state)
                        existing_by_id[state["row"]["example_id"]] = record
                        completed_batch_records.append(record)
                        state["completed"] = True
                append_jsonl(output_path, completed_batch_records)

        trailing_records = []
        for state in states:
            if state["completed"]:
                continue
            record = build_response_pair_record(state)
            existing_by_id[state["row"]["example_id"]] = record
            trailing_records.append(record)
            state["completed"] = True
        append_jsonl(output_path, trailing_records)

        unload_model(tokenizer, model)

    records = [existing_by_id[row["example_id"]] for row in source_rows]
    write_jsonl(output_path, records)
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
                "response_pairs": str(output_path),
            },
            "result_summary": summarize_records(records),
        },
    )

if __name__ == "__main__":
    main()
