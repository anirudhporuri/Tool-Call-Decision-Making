from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from cai_utils import count_label_values, load_jsonl, source_prompt_messages, source_user_request_text, validate_balanced_counts


SMOKE_LABEL_ORDER = ["tool_call", "request_for_info", "cannot_answer"]


def source_tag(source_file: str | Path) -> str:
    return Path(source_file).stem


def example_id_for_row(source_file: str | Path, source_row_index: int) -> str:
    return f"{source_tag(source_file)}:{source_row_index:05d}"


def source_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return count_label_values(rows, "chosen_behavior_class")


def should_enforce_strict_balance(*, dry_run: bool, smoke_run: bool, start_index: int, max_examples: Optional[int]) -> bool:
    return (
        not dry_run
        and not smoke_run
        and start_index == 0
        and max_examples is None
    )


def load_selected_source_rows(
    *,
    source_file: str | Path,
    start_index: int,
    max_examples: Optional[int],
    smoke_run: bool,
    smoke_per_class: int = 2,
) -> List[Dict[str, Any]]:
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
        buckets: Dict[str, List[tuple[int, Dict[str, Any]]]] = {label: [] for label in SMOKE_LABEL_ORDER}
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

    annotated: List[Dict[str, Any]] = []
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


def source_balance(rows: List[Dict[str, Any]], strict: bool) -> Dict[str, int]:
    if strict:
        return validate_balanced_counts(rows, "chosen_behavior_class")
    return source_counts(rows)


def index_stage_rows(rows: List[Dict[str, Any]], stage_name: str) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        example_id = row.get("example_id")
        if not example_id:
            raise ValueError(f"{stage_name} row missing example_id: {row}")
        if example_id in indexed:
            raise ValueError(f"Duplicate example_id in {stage_name}: {example_id}")
        indexed[example_id] = row
    return indexed


def ordered_stage_rows(
    source_rows: List[Dict[str, Any]],
    stage_rows: List[Dict[str, Any]],
    stage_name: str,
) -> List[Dict[str, Any]]:
    indexed = index_stage_rows(stage_rows, stage_name)
    ordered: List[Dict[str, Any]] = []
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
