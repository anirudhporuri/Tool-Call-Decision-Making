#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from w2c_prompts import ANSWER_ORDER, build_prompt, convert_test_choice_for_model


@dataclass
class CandidateScore:
    label: str
    text: str
    raw_logprob: float
    norm_logprob: float
    num_target_tokens: int



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="When2Call MCQ evaluation for zero/one/few-shot prompting baselines.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_family", type=str, choices=["llama", "gemma"], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="nvidia/When2Call")
    parser.add_argument("--dataset_config", type=str, default="test")
    parser.add_argument("--dataset_split", type=str, default="mcq")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--fewshot_json",
        type=str,
        default=None,
        help="Path to a JSON file containing the frozen few-shot exemplars. Required when num_shots > 0.",
    )
    parser.add_argument(
        "--save_prompt_text",
        action="store_true",
        help="Store the rendered prompt string in each sample JSONL row. Off by default because it is large.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer/model loading.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Transformers device_map. Use 'auto' on a single GPU node unless you need something else.",
    )
    return parser.parse_args()



def load_fewshot_examples(path: Optional[str], num_shots: int) -> List[Dict[str, Any]]:
    if num_shots == 0:
        return []
    if not path:
        raise ValueError("--fewshot_json is required when --num_shots > 0")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("fewshot_json must contain a JSON list of examples")
    if len(data) < num_shots:
        raise ValueError(f"fewshot_json contains {len(data)} examples but num_shots={num_shots}")
    required = {"question", "tools", "answer"}
    for i, ex in enumerate(data[:num_shots]):
        missing = required - set(ex)
        if missing:
            raise ValueError(f"Few-shot example {i} is missing required fields: {sorted(missing)}")
    return data[:num_shots]



def get_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]



def safe_model_slug(name: str) -> str:
    return name.replace("/", "__")



def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p



def flatten_answers_field(answers: Any) -> Dict[str, str]:
    """
    The test split documents `answers` as a dict keyed by category. Keep a fallback in case HF wraps it oddly.
    """
    if isinstance(answers, dict):
        return {k: str(v) for k, v in answers.items()}
    raise TypeError(f"Unsupported answers field type: {type(answers)}")



def score_prompt_plus_choice(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    choices: Sequence[Tuple[str, str]],
) -> List[CandidateScore]:
    """
    Score each choice with summed and length-normalized continuation log probability.

    We batch the four MCQ options together for speed.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
    prompt_len = int(prompt_ids.shape[0])

    texts = [prompt + choice_text for _, choice_text in choices]
    batch = tokenizer(texts, add_special_tokens=False, return_tensors="pt", padding=True)
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    results: List[CandidateScore] = []
    for row_idx, (label, choice_text) in enumerate(choices):
        seq_len = int(attention_mask[row_idx].sum().item())
        target_start = max(prompt_len - 1, 0)
        target_end = max(seq_len - 1, target_start)
        continuation_token_log_probs = token_log_probs[row_idx, target_start:target_end]
        num_target_tokens = int(continuation_token_log_probs.shape[0])
        raw = float(continuation_token_log_probs.sum().item())
        norm = raw / max(num_target_tokens, 1)
        results.append(
            CandidateScore(
                label=label,
                text=choice_text,
                raw_logprob=raw,
                norm_logprob=norm,
                num_target_tokens=num_target_tokens,
            )
        )
    return results



def compute_summary(golds: List[str], preds_raw: List[str], preds_norm: List[str]) -> Dict[str, Any]:
    labels = ANSWER_ORDER
    summary = {
        "n_examples": len(golds),
        "raw": {
            "accuracy": accuracy_score(golds, preds_raw),
            "macro_f1": f1_score(golds, preds_raw, labels=labels, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(golds, preds_raw, labels=labels).tolist(),
            "classification_report": classification_report(
                golds, preds_raw, labels=labels, zero_division=0, output_dict=True
            ),
        },
        "normalized": {
            "accuracy": accuracy_score(golds, preds_norm),
            "macro_f1": f1_score(golds, preds_norm, labels=labels, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(golds, preds_norm, labels=labels).tolist(),
            "classification_report": classification_report(
                golds, preds_norm, labels=labels, zero_division=0, output_dict=True
            ),
        },
        "label_order": labels,
    }
    return summary



def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    fewshot_examples = load_fewshot_examples(args.fewshot_json, args.num_shots)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        token=args.hf_token,
        torch_dtype=get_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    dataset = load_dataset(args.dataset_name, args.dataset_config)[args.dataset_split]
    end_index = len(dataset) if args.max_examples is None else min(len(dataset), args.start_index + args.max_examples)

    sample_path = out_dir / "samples.jsonl"
    summary_path = out_dir / "summary.json"
    config_path = out_dir / "run_config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    golds: List[str] = []
    preds_raw: List[str] = []
    preds_norm: List[str] = []

    with open(sample_path, "w", encoding="utf-8") as fout:
        for idx in tqdm(range(args.start_index, end_index), desc="Evaluating"):
            row = dataset[idx]
            answers = flatten_answers_field(row["answers"])
            prompt = build_prompt(
                model_family=args.model_family,
                question=row["question"],
                tools=row["tools"],
                fewshot_examples=fewshot_examples,
            )
            candidate_pairs = [
                (label, convert_test_choice_for_model(label, answers[label], args.model_family))
                for label in ANSWER_ORDER
            ]
            candidate_scores = score_prompt_plus_choice(model, tokenizer, prompt, candidate_pairs)
            pred_raw = max(candidate_scores, key=lambda x: x.raw_logprob).label
            pred_norm = max(candidate_scores, key=lambda x: x.norm_logprob).label

            golds.append(row["correct_answer"])
            preds_raw.append(pred_raw)
            preds_norm.append(pred_norm)

            sample_record: Dict[str, Any] = {
                "index": idx,
                "uuid": row["uuid"],
                "source": row.get("source"),
                "source_id": row.get("source_id"),
                "question": row["question"],
                "gold": row["correct_answer"],
                "pred_raw": pred_raw,
                "pred_norm": pred_norm,
                "tools": row["tools"],
                "answers_original": answers,
                "choices_scored": [
                    {
                        "label": cs.label,
                        "rendered_choice": cs.text,
                        "raw_logprob": cs.raw_logprob,
                        "norm_logprob": cs.norm_logprob,
                        "num_target_tokens": cs.num_target_tokens,
                    }
                    for cs in candidate_scores
                ],
            }
            if args.save_prompt_text:
                sample_record["prompt"] = prompt
            fout.write(json.dumps(sample_record, ensure_ascii=False) + "\n")

    summary = compute_summary(golds, preds_raw, preds_norm)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "saved_samples": str(sample_path),
        "saved_summary": str(summary_path),
        "raw_accuracy": summary["raw"]["accuracy"],
        "raw_macro_f1": summary["raw"]["macro_f1"],
        "norm_accuracy": summary["normalized"]["accuracy"],
        "norm_macro_f1": summary["normalized"]["macro_f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
