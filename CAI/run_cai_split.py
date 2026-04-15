#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import load_dataset, load_from_disk

from cai_utils import (
    canonicalize_assistant_response,
    ensure_dir,
    heuristic_class,
    load_jsonl,
    parse_tools_spec,
    progress,
    save_json,
    validate_single_tool_call,
    write_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "local_datasets"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated_datasets"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare balanced CAI source splits from When2Call train_pref.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default=os.getenv("DATASET_DIR", str(DEFAULT_DATASET_DIR)))
    parser.add_argument("--dataset-name", default="nvidia/When2Call")
    parser.add_argument("--dataset-config", default="train_pref")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--per-class-total",
        type=int,
        default=None,
        help="Optional balanced total to sample per class before splitting. Must be even if provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--allow-hf-fallback",
        action="store_true",
        help="Allow downloading from Hugging Face if the local dataset snapshot is missing.",
    )
    return parser.parse_args(argv)


def safe_dataset_slug(*parts: str) -> str:
    return "__".join(part.replace("/", "__") for part in parts if part)


def load_source_dataset(args: argparse.Namespace):
    dataset_root = ensure_dir(args.dataset_dir)
    snapshot_dir = dataset_root / safe_dataset_slug(args.dataset_name, args.dataset_config)
    hf_cache_dir = ensure_dir(dataset_root / "_hf_cache")

    if snapshot_dir.exists():
        dataset_obj = load_from_disk(str(snapshot_dir))
        return dataset_obj[args.dataset_split]

    if not args.allow_hf_fallback:
        raise FileNotFoundError(
            f"Local dataset snapshot not found: {snapshot_dir}. Pass --allow-hf-fallback to download it."
        )

    dataset_obj = load_dataset(
        args.dataset_name,
        args.dataset_config,
        cache_dir=str(hf_cache_dir),
    )
    dataset_obj.save_to_disk(str(snapshot_dir))
    return dataset_obj[args.dataset_split]


def row_with_metadata(row: Dict[str, Any], chosen_behavior_class: str) -> Dict[str, Any]:
    payload = dict(row)
    payload["chosen_behavior_class"] = chosen_behavior_class
    return payload


def normalized_key(row: Dict[str, Any]) -> Tuple[str, str]:
    user_request = ""
    messages = row.get("messages") or []
    if messages:
        user_request = str(messages[0].get("content", "")).strip()

    parsed_tools = parse_tools_spec(row.get("tools"))
    tools_key = json.dumps(parsed_tools, ensure_ascii=False, sort_keys=True)
    return user_request, tools_key


def rejected_response_is_valid_tool_call(row: Dict[str, Any]) -> bool:
    rejected = str(row.get("rejected_response", {}).get("content", "")).strip()
    canonical = canonicalize_assistant_response(rejected)
    validation = validate_single_tool_call(canonical, row.get("tools"))
    return bool(validation.get("valid"))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = ensure_dir(args.output_dir)
    ds = load_source_dataset(args)

    raw_pools: Dict[str, List[Dict[str, Any]]] = {
        "tool_call": [],
        "request_for_info": [],
        "cannot_answer": [],
    }
    rows_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    ignored = 0
    for row in progress(ds, total=len(ds), desc="Classifying train_pref rows", leave=False):
        label = heuristic_class(row["chosen_response"]["content"])
        if label not in raw_pools:
            ignored += 1
            continue
        payload = row_with_metadata(row, label)
        raw_pools[label].append(payload)
        key = normalized_key(payload)
        rows_by_key.setdefault(key, []).append(payload)

    original_counts = {label: len(rows) for label, rows in raw_pools.items()}

    conflicting_key_count = 0
    conflicting_row_counts = {label: 0 for label in raw_pools}
    conflicting_keys: set[Tuple[str, str]] = set()
    for key, rows in rows_by_key.items():
        labels = {str(row["chosen_behavior_class"]) for row in rows}
        if len(labels) > 1:
            conflicting_key_count += 1
            conflicting_keys.add(key)
            for row in rows:
                conflicting_row_counts[str(row["chosen_behavior_class"])] += 1

    pools: Dict[str, List[Dict[str, Any]]] = {
        "tool_call": [],
        "request_for_info": [],
        "cannot_answer": [],
    }
    valid_tool_rejected_rfi_drop_count = 0
    for label, rows in raw_pools.items():
        for row in rows:
            key = normalized_key(row)
            if key in conflicting_keys:
                continue
            if label == "request_for_info" and rejected_response_is_valid_tool_call(row):
                valid_tool_rejected_rfi_drop_count += 1
                continue
            pools[label].append(row)

    eligible_counts = {label: len(rows) for label, rows in pools.items()}
    min_count = min(eligible_counts.values())
    auto_total = min_count - (min_count % 2)
    if auto_total <= 0:
        raise ValueError(f"Unable to form an even balanced split from counts: {eligible_counts}")

    per_class_total = args.per_class_total or auto_total
    if per_class_total % 2 != 0:
        raise ValueError("--per-class-total must be even.")
    if any(per_class_total > count for count in eligible_counts.values()):
        raise ValueError(
            f"Requested per-class-total={per_class_total}, but eligible counts are {eligible_counts}."
        )

    per_half = per_class_total // 2
    selected_sft: List[Dict[str, Any]] = []
    selected_dpo: List[Dict[str, Any]] = []
    sampling_info: Dict[str, Dict[str, int]] = {}

    import random

    rng = random.Random(args.seed)
    for label, rows in pools.items():
        sample = rng.sample(rows, per_class_total)
        rng.shuffle(sample)
        sft_rows = sample[:per_half]
        dpo_rows = sample[per_half:]
        selected_sft.extend(sft_rows)
        selected_dpo.extend(dpo_rows)
        sampling_info[label] = {
            "source_count": len(rows),
            "selected_total": per_class_total,
            "sft_half": len(sft_rows),
            "dpo_half": len(dpo_rows),
            "dropped_from_source": len(rows) - per_class_total,
        }

    sft_path = output_dir / "train_pref_cai_sft_source.jsonl"
    dpo_path = output_dir / "train_pref_cai_dpo_source.jsonl"
    summary_path = output_dir / "train_pref_cai_split_summary.json"

    write_jsonl(sft_path, selected_sft)
    write_jsonl(dpo_path, selected_dpo)

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "seed": args.seed,
        "ignored_rows": ignored,
        "original_counts": original_counts,
        "filtering": {
            "conflicting_key_count": conflicting_key_count,
            "conflicting_row_counts": conflicting_row_counts,
            "valid_tool_rejected_request_for_info_drops": valid_tool_rejected_rfi_drop_count,
            "eligible_counts": eligible_counts,
        },
        "per_class_total": per_class_total,
        "per_half": per_half,
        "sampling_info": sampling_info,
        "output_files": {
            "cai_sft_source": str(sft_path),
            "cai_dpo_source": str(dpo_path),
        },
    }
    save_json(summary_path, summary)

    sft_rows = load_jsonl(sft_path)
    dpo_rows = load_jsonl(dpo_path)
    sft_counts = {}
    dpo_counts = {}
    for row in sft_rows:
        label = row["chosen_behavior_class"]
        sft_counts[label] = sft_counts.get(label, 0) + 1
    for row in dpo_rows:
        label = row["chosen_behavior_class"]
        dpo_counts[label] = dpo_counts.get(label, 0) + 1

    if len(set(sft_counts.values())) > 1 or len(set(dpo_counts.values())) > 1:
        raise RuntimeError(
            f"Generated CAI split is not balanced. sft_counts={sft_counts}, dpo_counts={dpo_counts}"
        )


if __name__ == "__main__":
    main()
