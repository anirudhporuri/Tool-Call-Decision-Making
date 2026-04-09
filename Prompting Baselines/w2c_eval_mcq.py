#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset, load_from_disk
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
    token_norm_logprob: float
    num_target_tokens: int
    num_target_bytes: int



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="When2Call MCQ evaluation for zero/one/few-shot prompting baselines.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_family", type=str, choices=["llama", "gemma"], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="nvidia/When2Call")
    parser.add_argument("--dataset_config", type=str, default="test")
    parser.add_argument("--dataset_split", type=str, default="mcq")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Explicit directory used to persist downloaded datasets for reuse across runs.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate dataset loading and prompt rendering without loading tokenizer/model weights.",
    )
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
    return parser.parse_args(argv)



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

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def safe_dataset_slug(*parts: str) -> str:
    return "__".join(part.replace("/", "__") for part in parts if part)


def load_eval_dataset(args: argparse.Namespace):
    if not args.dataset_dir:
        return load_dataset(args.dataset_name, args.dataset_config)[args.dataset_split]

    dataset_root = ensure_dir(args.dataset_dir)
    snapshot_dir = dataset_root / safe_dataset_slug(args.dataset_name, args.dataset_config)
    hf_cache_dir = ensure_dir(dataset_root / "_hf_cache")

    if snapshot_dir.exists():
        dataset_obj = load_from_disk(str(snapshot_dir))
    else:
        dataset_obj = load_dataset(
            args.dataset_name,
            args.dataset_config,
            cache_dir=str(hf_cache_dir),
        )
        dataset_obj.save_to_disk(str(snapshot_dir))

    return dataset_obj[args.dataset_split]



def flatten_answers_field(answers: Any) -> Dict[str, str]:
    """
    The test split documents `answers` as a dict keyed by category. Keep a fallback in case HF wraps it oddly.
    """
    if isinstance(answers, dict):
        return {k: str(v) for k, v in answers.items()}
    raise TypeError(f"Unsupported answers field type: {type(answers)}")


def get_end_index(dataset_size: int, start_index: int, max_examples: Optional[int]) -> int:
    if max_examples is None:
        return dataset_size
    return min(dataset_size, start_index + max_examples)


def build_candidate_pairs(answers: Dict[str, str], model_family: str) -> List[Tuple[str, str]]:
    return [
        (label, convert_test_choice_for_model(label, answers[label], model_family))
        for label in ANSWER_ORDER
    ]


def build_sample_base(row: Dict[str, Any], index: int, answers: Dict[str, str]) -> Dict[str, Any]:
    return {
        "index": index,
        "uuid": row["uuid"],
        "source": row.get("source"),
        "source_id": row.get("source_id"),
        "question": row["question"],
        "gold": row["correct_answer"],
        "tools": row["tools"],
        "answers_original": answers,
    }



def is_peft_adapter_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "adapter_config.json").is_file()


def load_tokenizer_and_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "token": args.hf_token,
        "torch_dtype": get_dtype(args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": args.attn_implementation,
    }

    if is_peft_adapter_checkpoint(args.model_name_or_path):
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise ImportError(
                "Evaluating a LoRA/PEFT checkpoint requires `peft`. Install it in the prompting-baselines "
                "environment or pass a merged model directory instead."
            ) from exc

        peft_config = PeftConfig.from_pretrained(args.model_name_or_path, token=args.hf_token)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            args.model_name_or_path,
            token=args.hf_token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_kwargs,
        )

    return tokenizer, model


def score_prompt_plus_choice(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    choices: Sequence[Tuple[str, str]],
) -> List[CandidateScore]:
    """
    Score each choice with summed continuation log probability and a When2Call-style
    normalized score.

    The paper reports byte-length-normalized accuracy via LM Evaluation Harness rather
    than token-count normalization. We therefore normalize by the UTF-8 byte length of
    the rendered answer choice. We also keep token-count normalization as an auxiliary
    diagnostic because it can still be useful when inspecting outputs.

    We batch the four MCQ options together for speed.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
    prompt_len = int(prompt_ids.shape[0])

    texts = [prompt + choice_text for _, choice_text in choices]
    batch = tokenizer(texts, add_special_tokens=False, return_tensors="pt", padding=True)
    model_device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(model_device)
    attention_mask = batch["attention_mask"].to(model_device)

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
        num_target_bytes = len(choice_text.encode("utf-8"))
        raw = float(continuation_token_log_probs.sum().item())
        norm = raw / max(num_target_bytes, 1)
        token_norm = raw / max(num_target_tokens, 1)
        results.append(
            CandidateScore(
                label=label,
                text=choice_text,
                raw_logprob=raw,
                norm_logprob=norm,
                token_norm_logprob=token_norm,
                num_target_tokens=num_target_tokens,
                num_target_bytes=num_target_bytes,
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


def write_dry_run_outputs(
    *,
    args: argparse.Namespace,
    dataset,
    out_dir: Path,
    fewshot_examples: List[Dict[str, Any]],
    end_index: int,
) -> Dict[str, Any]:
    sample_path = out_dir / "samples.jsonl"
    summary_path = out_dir / "summary.json"
    config_path = out_dir / "run_config.json"

    save_json(config_path, vars(args))

    with open(sample_path, "w", encoding="utf-8") as fout:
        for idx in tqdm(range(args.start_index, end_index), desc="Dry run"):
            row = dataset[idx]
            answers = flatten_answers_field(row["answers"])
            prompt = build_prompt(
                model_family=args.model_family,
                question=row["question"],
                tools=row["tools"],
                fewshot_examples=fewshot_examples,
            )
            candidate_pairs = build_candidate_pairs(answers, args.model_family)
            sample_record: Dict[str, Any] = {
                "mode": "dry_run",
                **build_sample_base(row, idx, answers),
                "rendered_choices": [
                    {"label": label, "rendered_choice": choice_text}
                    for label, choice_text in candidate_pairs
                ],
                "prompt_char_length": len(prompt),
            }
            if args.save_prompt_text:
                sample_record["prompt"] = prompt
            fout.write(json.dumps(sample_record, ensure_ascii=False) + "\n")

    summary = {
        "mode": "dry_run",
        "model_loaded": False,
        "tokenizer_loaded": False,
        "n_examples_previewed": max(end_index - args.start_index, 0),
        "saved_samples": str(sample_path),
        "saved_summary": str(summary_path),
        "checks_completed": [
            "fewshot_loading",
            "dataset_loading",
            "prompt_rendering",
            "answer_choice_rendering",
            "output_writes",
        ],
    }
    save_json(summary_path, summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = ensure_dir(args.output_dir)
    fewshot_examples = load_fewshot_examples(args.fewshot_json, args.num_shots)

    dataset = load_eval_dataset(args)
    end_index = get_end_index(len(dataset), args.start_index, args.max_examples)

    sample_path = out_dir / "samples.jsonl"
    summary_path = out_dir / "summary.json"
    config_path = out_dir / "run_config.json"

    if args.dry_run:
        summary = write_dry_run_outputs(
            args=args,
            dataset=dataset,
            out_dir=out_dir,
            fewshot_examples=fewshot_examples,
            end_index=end_index,
        )
        print(json.dumps(summary, indent=2))
        return

    tokenizer, model = load_tokenizer_and_model(args)
    model.eval()

    save_json(config_path, vars(args))

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
            candidate_pairs = build_candidate_pairs(answers, args.model_family)
            candidate_scores = score_prompt_plus_choice(model, tokenizer, prompt, candidate_pairs)
            pred_raw = max(candidate_scores, key=lambda x: x.raw_logprob).label
            pred_norm = max(candidate_scores, key=lambda x: x.norm_logprob).label

            golds.append(row["correct_answer"])
            preds_raw.append(pred_raw)
            preds_norm.append(pred_norm)

            sample_record: Dict[str, Any] = {
                **build_sample_base(row, idx, answers),
                "pred_raw": pred_raw,
                "pred_norm": pred_norm,
                "choices_scored": [
                    {
                        "label": cs.label,
                        "rendered_choice": cs.text,
                        "raw_logprob": cs.raw_logprob,
                        "norm_logprob": cs.norm_logprob,
                        "token_norm_logprob": cs.token_norm_logprob,
                        "num_target_tokens": cs.num_target_tokens,
                        "num_target_bytes": cs.num_target_bytes,
                    }
                    for cs in candidate_scores
                ],
            }
            if args.save_prompt_text:
                sample_record["prompt"] = prompt
            fout.write(json.dumps(sample_record, ensure_ascii=False) + "\n")

    summary = compute_summary(golds, preds_raw, preds_norm)
    save_json(summary_path, summary)

    print(
        json.dumps(
            {
                "saved_samples": str(sample_path),
                "saved_summary": str(summary_path),
                "raw_accuracy": summary["raw"]["accuracy"],
                "raw_macro_f1": summary["raw"]["macro_f1"],
                "norm_accuracy": summary["normalized"]["accuracy"],
                "norm_macro_f1": summary["normalized"]["macro_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
