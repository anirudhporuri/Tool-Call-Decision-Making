#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset, load_from_disk
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from w2c_train_format import format_sft_example


@dataclass
class EncodedExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LoRA SFT on When2Call train_sft.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--model_family", type=str, choices=["llama", "gemma"], required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--dataset_name", type=str, default="nvidia/When2Call")
    p.add_argument("--dataset_config", type=str, default="train_sft")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Explicit directory used to persist downloaded datasets for reuse across runs.",
    )
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--val_size", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default=None)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules.",
    )
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args(argv)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def safe_dataset_slug(*parts: str) -> str:
    return "__".join(part.replace("/", "__") for part in parts if part)


def get_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]
def load_source_dataset(args: argparse.Namespace):
    cache_dir = None
    if args.dataset_dir:
        cache_dir = str(ensure_dir(Path(args.dataset_dir) / "_hf_cache"))

    if args.train_file:
        ds = load_dataset("json", data_files=args.train_file, cache_dir=cache_dir)["train"]
    else:
        if args.dataset_dir:
            dataset_root = ensure_dir(args.dataset_dir)
            snapshot_dir = dataset_root / safe_dataset_slug(args.dataset_name, args.dataset_config)
            if snapshot_dir.exists():
                dataset_obj = load_from_disk(str(snapshot_dir))
            else:
                dataset_obj = load_dataset(
                    args.dataset_name,
                    args.dataset_config,
                    cache_dir=cache_dir,
                )
                dataset_obj.save_to_disk(str(snapshot_dir))
            ds = dataset_obj[args.dataset_split]
        else:
            ds = load_dataset(args.dataset_name, args.dataset_config)[args.dataset_split]
    return ds


def train_eval_split(ds, val_size: float, seed: int):
    if val_size <= 0.0:
        return ds, None
    split = ds.train_test_split(test_size=val_size, seed=seed, shuffle=True)
    return split["train"], split["test"]


def preprocess_row(row: Dict[str, Any], model_family: str) -> Dict[str, str]:
    prompt, target = format_sft_example(model_family=model_family, row=row)
    return {"prompt": prompt, "target": target}


def maybe_limit_dataset(dataset: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def encode_example(tokenizer, prompt: str, target: str, max_length: int) -> EncodedExample:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        target_ids = target_ids + [eos_id]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]

    attention_mask = [1] * len(input_ids)
    return EncodedExample(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


class SupervisedDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        pad_id = self.tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def format_dataset(dataset: Dataset, model_family: str, desc: str) -> Dataset:
    return dataset.map(
        lambda row: preprocess_row(row, model_family),
        remove_columns=dataset.column_names,
        desc=desc,
    )


def tokenize_dataset(dataset: Optional[Dataset], tokenizer, max_length: int, desc: str) -> Optional[Dataset]:
    if dataset is None:
        return None
    return dataset.map(
        lambda row: asdict(encode_example(tokenizer, row["prompt"], row["target"], max_length)),
        remove_columns=dataset.column_names,
        desc=desc,
    )


def build_quant_config(args: argparse.Namespace) -> Optional[BitsAndBytesConfig]:
    if not args.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=get_dtype(args.dtype),
        bnb_4bit_use_double_quant=True,
    )


def save_preview(dataset: Dataset, path: Path, n: int = 20) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(min(n, len(dataset))):
            f.write(json.dumps(dataset[i], ensure_ascii=False) + "\n")


def write_dry_run_summary(
    *,
    out_dir: Path,
    train_examples: int,
    eval_examples: int,
    train_preview_path: Path,
    eval_preview_path: Optional[Path],
) -> Dict[str, Any]:
    summary = {
        "mode": "dry_run",
        "model_loaded": False,
        "tokenizer_loaded": False,
        "train_examples_previewed": train_examples,
        "eval_examples_previewed": eval_examples,
        "saved_train_preview": str(train_preview_path),
        "saved_eval_preview": str(eval_preview_path) if eval_preview_path is not None else None,
        "checks_completed": [
            "dataset_loading",
            "train_eval_split",
            "prompt_formatting",
            "preview_writes",
        ],
    }
    with open(out_dir / "dry_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = ensure_dir(args.output_dir)
    ensure_dir(out_dir / "logs")

    save_json(out_dir / "run_config.json", vars(args))

    raw_ds = load_source_dataset(args)
    raw_ds = maybe_limit_dataset(raw_ds, args.max_train_samples)
    train_ds, eval_ds = train_eval_split(raw_ds, args.val_size, args.seed)
    eval_ds = maybe_limit_dataset(eval_ds, args.max_eval_samples) if eval_ds is not None else None

    processed_train = format_dataset(train_ds, args.model_family, "Formatting train prompts")
    processed_eval = format_dataset(eval_ds, args.model_family, "Formatting eval prompts") if eval_ds is not None else None

    train_preview_path = out_dir / "formatted_train_preview.jsonl"
    eval_preview_path = out_dir / "formatted_eval_preview.jsonl" if processed_eval is not None else None

    save_preview(processed_train, train_preview_path)
    if processed_eval is not None:
        save_preview(processed_eval, eval_preview_path)

    if args.dry_run:
        summary = write_dry_run_summary(
            out_dir=out_dir,
            train_examples=len(processed_train),
            eval_examples=(len(processed_eval) if processed_eval is not None else 0),
            train_preview_path=train_preview_path,
            eval_preview_path=eval_preview_path,
        )
        print(json.dumps(summary, indent=2))
        return

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_train = tokenize_dataset(processed_train, tokenizer, args.max_length, "Tokenizing train data")
    tokenized_eval = tokenize_dataset(processed_eval, tokenizer, args.max_length, "Tokenizing eval data")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        token=args.hf_token,
        torch_dtype=get_dtype(args.dtype),
        quantization_config=build_quant_config(args),
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if tokenized_eval is not None else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        report_to=args.report_to,
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_num_workers=2,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=SupervisedDataCollator(tokenizer),
        tokenizer=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(out_dir)

    metrics = train_result.metrics
    save_json(out_dir / "train_metrics.json", metrics)

    if tokenized_eval is not None:
        eval_metrics = trainer.evaluate()
        save_json(out_dir / "eval_metrics.json", eval_metrics)


if __name__ == "__main__":
    main()
