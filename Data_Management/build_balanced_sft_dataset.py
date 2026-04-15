#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset, load_from_disk


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "local_datasets_train_test"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated_datasets"

TOOLCALL_RE = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)
CANNOT_PATTERNS = [
    r"\bsorry\b",
    r"\bapologies\b",
    r"\bapologize\b",
    r"\bapologise\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bunable\b",
]
REQUEST_PATTERNS = [
    r"\bcould you\b",
    r"\bcan you\b",
    r"\bplease provide\b",
    r"\bplease specify\b",
    r"\bwhat is\b",
    r"\bwhat's\b",
    r"\bwhich\b",
    r"\bto assist you better\b",
    r"\bjust to confirm\b",
    r"\bdo you have\b",
    r"\bmay i have\b",
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced SFT dataset from When2Call SFT + preference data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--dataset-name", default="nvidia/When2Call")
    parser.add_argument("--sft-config", default="train_sft")
    parser.add_argument("--pref-config", default="train_pref")
    parser.add_argument("--sft-split", default="train")
    parser.add_argument("--pref-split", default="train")
    parser.add_argument("--num-per-class", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_OUTPUT_DIR / "when2call_balanced_sft_3x3000.jsonl"),
    )
    parser.add_argument(
        "--output-summary",
        default=str(DEFAULT_OUTPUT_DIR / "when2call_balanced_sft_3x3000.summary.json"),
    )
    parser.add_argument(
        "--allow-hf-fallback",
        action="store_true",
        help="Allow downloading from Hugging Face if a local dataset snapshot is not found.",
    )
    return parser.parse_args(argv)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_dataset_slug(*parts: str) -> str:
    return "__".join(part.replace("/", "__") for part in parts if part)


def load_split(
    *,
    dataset_dir: Path,
    dataset_name: str,
    config: str,
    split: str,
    allow_hf_fallback: bool,
):
    snapshot_dir = dataset_dir / safe_dataset_slug(dataset_name, config) / split
    if snapshot_dir.exists():
        return load_from_disk(str(snapshot_dir))

    if not allow_hf_fallback:
        raise FileNotFoundError(
            f"Local dataset snapshot not found: {snapshot_dir}. "
            "Pass --allow-hf-fallback if you want to download it."
        )

    dataset_obj = load_dataset(dataset_name, config)
    return dataset_obj[split]


def extract_toolcall_payload(text: str) -> str:
    if not isinstance(text, str):
        return ""
    match = TOOLCALL_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def has_tool_call_marker(text: str) -> bool:
    raw = text if isinstance(text, str) else ""
    payload = extract_toolcall_payload(raw)
    return ("<TOOLCALL>" in raw) or payload.startswith("{") or payload.startswith("[")


def has_cannot_answer_terms(text: str) -> bool:
    lower = (text or "").lower()
    return any(re.search(pattern, lower) for pattern in CANNOT_PATTERNS)


def looks_like_request_for_info(text: str) -> bool:
    lower = (text or "").lower().strip()
    if has_tool_call_marker(lower) or has_cannot_answer_terms(lower):
        return False
    return lower.endswith("?") or any(re.search(pattern, lower) for pattern in REQUEST_PATTERNS)


def heuristic_class(text: str) -> str:
    if has_tool_call_marker(text):
        return "tool_call"
    if has_cannot_answer_terms(text):
        return "cannot_answer"
    if looks_like_request_for_info(text):
        return "request_for_info"
    return "other_plain_text"


def canonicalize_record(tools: Any, messages: List[Dict[str, Any]], source_split: str, behavior_class: str) -> Dict[str, Any]:
    return {
        "tools": tools,
        "messages": messages,
        "source_split": source_split,
        "behavior_class": behavior_class,
    }


def build_sft_pool(dataset) -> Dict[str, List[Dict[str, Any]]]:
    pools: Dict[str, List[Dict[str, Any]]] = {
        "cannot_answer": [],
        "request_for_info": [],
        "other_plain_text": [],
    }
    for row in dataset:
        if not row["messages"] or row["messages"][-1].get("role") != "assistant":
            continue
        answer = row["messages"][-1].get("content", "")
        label = heuristic_class(answer)
        if label in pools:
            pools[label].append(
                canonicalize_record(
                    tools=row["tools"],
                    messages=row["messages"],
                    source_split="train_sft",
                    behavior_class=label,
                )
            )
    return pools


def build_pref_tool_pool(dataset) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    for row in dataset:
        chosen = row["chosen_response"].get("content", "")
        if heuristic_class(chosen) != "tool_call":
            continue
        messages = list(row["messages"]) + [row["chosen_response"]]
        pool.append(
            canonicalize_record(
                tools=row["tools"],
                messages=messages,
                source_split="train_pref_chosen",
                behavior_class="tool_call",
            )
        )
    return pool


def sample_examples(examples: List[Dict[str, Any]], k: int, rng: random.Random, label: str) -> List[Dict[str, Any]]:
    if len(examples) < k:
        raise ValueError(f"Not enough {label} examples: requested {k}, found {len(examples)}")
    return rng.sample(examples, k)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def preview_text(messages: List[Dict[str, Any]]) -> str:
    parts = [f"{msg.get('role', '?')}: {msg.get('content', '').replace(chr(10), ' ')}" for msg in messages]
    return " | ".join(parts)[:240]


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    dataset_dir = ensure_dir(args.dataset_dir)
    output_jsonl = Path(args.output_jsonl)
    output_summary = Path(args.output_summary)
    ensure_dir(output_jsonl.parent)
    ensure_dir(output_summary.parent)

    sft_ds = load_split(
        dataset_dir=dataset_dir,
        dataset_name=args.dataset_name,
        config=args.sft_config,
        split=args.sft_split,
        allow_hf_fallback=args.allow_hf_fallback,
    )
    pref_ds = load_split(
        dataset_dir=dataset_dir,
        dataset_name=args.dataset_name,
        config=args.pref_config,
        split=args.pref_split,
        allow_hf_fallback=args.allow_hf_fallback,
    )

    sft_pools = build_sft_pool(sft_ds)
    pref_tool_pool = build_pref_tool_pool(pref_ds)

    selected_cannot = sample_examples(sft_pools["cannot_answer"], args.num_per_class, rng, "cannot_answer")
    selected_request = sample_examples(sft_pools["request_for_info"], args.num_per_class, rng, "request_for_info")
    selected_tool = sample_examples(pref_tool_pool, args.num_per_class, rng, "tool_call")

    balanced_rows = selected_cannot + selected_request + selected_tool
    rng.shuffle(balanced_rows)

    write_jsonl(output_jsonl, balanced_rows)

    summary = {
        "dataset_name": args.dataset_name,
        "sft_config": args.sft_config,
        "pref_config": args.pref_config,
        "num_per_class": args.num_per_class,
        "seed": args.seed,
        "output_jsonl": str(output_jsonl),
        "total_examples": len(balanced_rows),
        "source_pool_sizes": {
            "sft_cannot_answer": len(sft_pools["cannot_answer"]),
            "sft_request_for_info": len(sft_pools["request_for_info"]),
            "sft_other_plain_text": len(sft_pools["other_plain_text"]),
            "pref_chosen_tool_call": len(pref_tool_pool),
        },
        "selected_counts": dict(Counter(row["behavior_class"] for row in balanced_rows)),
        "selected_source_counts": dict(Counter(row["source_split"] for row in balanced_rows)),
        "sample_rows": [
            {
                "behavior_class": row["behavior_class"],
                "source_split": row["source_split"],
                "preview": preview_text(row["messages"]),
            }
            for row in balanced_rows[:9]
        ],
    }

    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
