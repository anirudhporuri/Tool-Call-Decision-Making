import argparse
import os
from collections import namedtuple
from pathlib import Path

from cai_utils import count_label_values, load_jsonl, source_prompt_messages, source_user_request_text, validate_balanced_counts

SMOKE_LABEL_ORDER = ["tool_call", "request_for_info", "cannot_answer"]

StageSource = namedtuple("StageSource", "rows max_examples strict_balance balance")

def env_flag(name, default):
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}

def env_int(name, default):
    value = os.getenv(name)
    return int(value) if value is not None else default

def env_float(name, default):
    value = os.getenv(name)
    return float(value) if value is not None else default

def add_bool_flag(parser, name, default, help_text):
    dest = name[2:].replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", default=default, help=help_text)
    parser.add_argument(f"--no-{name[2:]}", dest=dest, action="store_false", help=argparse.SUPPRESS)

def add_run_mode_args(parser):
    parser.add_argument("--dry-run", action="store_true", default=env_flag("DRY_RUN", False))
    parser.add_argument("--smoke-run", action="store_true", default=env_flag("SMOKE_RUN", False))
    parser.add_argument("--dry-run-max-examples", type=int, default=env_int("DRY_RUN_MAX_EXAMPLES", 8))
    parser.add_argument("--smoke-run-max-examples", type=int, default=env_int("SMOKE_RUN_MAX_EXAMPLES", 6))

def parse_stage_args(parser, argv=None):
    args = parser.parse_args(argv)
    if args.dry_run and args.smoke_run:
        parser.error("--dry-run and --smoke-run are mutually exclusive.")
    return args

def sanitized_args_dict(args):
    payload = vars(args).copy()
    if payload.get("hf_token"):
        payload["hf_token"] = "[REDACTED]"
    return payload

def resolve_max_examples(args):
    if args.max_examples is not None:
        return args.max_examples
    if args.dry_run:
        return args.dry_run_max_examples
    if args.smoke_run:
        return args.smoke_run_max_examples
    return None

def count_values(values):
    counts = {}
    for value in values:
        label = str(value)
        counts[label] = counts.get(label, 0) + 1
    return counts

def source_tag(source_file):
    return Path(source_file).stem

def example_id_for_row(source_file, source_row_index):
    return f"{source_tag(source_file)}:{source_row_index:05d}"

def source_counts(rows):
    return count_label_values(rows, "chosen_behavior_class")

def should_enforce_strict_balance(*, dry_run, smoke_run, start_index, max_examples):
    return (
        not dry_run
        and not smoke_run
        and start_index == 0
        and max_examples is None
    )

def load_selected_source_rows(
    *,
    source_file,
    start_index,
    max_examples,
    smoke_run,
    smoke_per_class=2,
):
    all_rows = load_jsonl(source_file)
    enumerated = list(enumerate(all_rows[start_index:], start=start_index))
    if smoke_run:
        if max_examples is not None:
            if max_examples <= 0:
                raise ValueError("--max-examples must be positive when used with --smoke-run.")
            if max_examples % len(SMOKE_LABEL_ORDER) != 0:
                raise ValueError(
                    f"Smoke run requires max_examples to be divisible by {len(SMOKE_LABEL_ORDER)} "
                    f"(one equal bucket per class), got {max_examples}."
                )
            smoke_per_class = max_examples // len(SMOKE_LABEL_ORDER)
        buckets = {label: [] for label in SMOKE_LABEL_ORDER}
        for row_index, row in enumerated:
            label = row["chosen_behavior_class"]
            if label in buckets and len(buckets[label]) < smoke_per_class:
                buckets[label].append((row_index, row))
        missing = [label for label, rows in buckets.items() if len(rows) < smoke_per_class]
        if missing:
            raise ValueError(
                f"Smoke run needs {smoke_per_class} examples per class, but could not satisfy: {missing}"
            )
        selected = [item for label in SMOKE_LABEL_ORDER for item in buckets[label]]
    else:
        selected = enumerated[:max_examples] if max_examples is not None else enumerated

    annotated = []
    split_name = source_tag(source_file)
    for row_index, row in selected:
        annotated_row = dict(row)
        annotated_row["messages"] = source_prompt_messages(annotated_row)
        annotated_row["source_row_index"] = row_index
        annotated_row["example_id"] = example_id_for_row(source_file, row_index)
        annotated_row["source_split"] = split_name
        annotated_row["user_request"] = source_user_request_text(annotated_row)
        annotated.append(annotated_row)
    return annotated

def source_balance(rows, strict):
    if strict:
        return validate_balanced_counts(rows, "chosen_behavior_class")
    return source_counts(rows)

def prepare_stage_source(args, *, enforce_strict_balance=False):
    max_examples = resolve_max_examples(args)
    strict_balance = enforce_strict_balance and should_enforce_strict_balance(
        dry_run=args.dry_run,
        smoke_run=args.smoke_run,
        start_index=args.start_index,
        max_examples=max_examples,
    )
    rows = load_selected_source_rows(
        source_file=args.source_file,
        start_index=args.start_index,
        max_examples=max_examples,
        smoke_run=args.smoke_run and args.max_examples is None,
    )
    return StageSource(
        rows=rows,
        max_examples=max_examples,
        strict_balance=strict_balance,
        balance=source_balance(rows, strict_balance),
    )

def index_stage_rows(rows, stage_name):
    indexed = {}
    for row in rows:
        example_id = row.get("example_id")
        if not example_id:
            raise ValueError(f"{stage_name} row missing example_id: {row}")
        if example_id in indexed:
            raise ValueError(f"Duplicate example_id in {stage_name}: {example_id}")
        indexed[example_id] = row
    return indexed

def ordered_stage_rows(
    source_rows,
    stage_rows,
    stage_name,
):
    indexed = index_stage_rows(stage_rows, stage_name)
    ordered = []
    for source_row in source_rows:
        example_id = source_row["example_id"]
        if example_id not in indexed:
            raise ValueError(f"Missing {stage_name} row for example_id={example_id}")
        ordered.append(indexed[example_id])
    if len(indexed) != len(source_rows):
        extras = sorted(set(indexed) - {row["example_id"] for row in source_rows})
        if extras:
            raise ValueError(f"{stage_name} contains rows not present in source set: {extras[:3]}")
    return ordered

def load_existing_stage_records(
    output_path,
    source_rows,
    stage_name,
):
    output_path = Path(output_path)
    if not output_path.exists():
        return [], {}

    existing_rows = load_jsonl(output_path, allow_partial_last_line=True)
    if output_path.stat().st_size > 0 and not existing_rows:
        raise RuntimeError(
            f"Existing {stage_name} file {output_path} is non-empty but no rows could be recovered. "
            "Refusing to overwrite it automatically."
        )
    indexed = index_stage_rows(existing_rows, stage_name)
    source_example_ids = {row["example_id"] for row in source_rows}
    extras = sorted(set(indexed) - source_example_ids)
    if extras:
        raise ValueError(f"{stage_name} contains rows not present in source set: {extras[:3]}")
    return existing_rows, indexed
