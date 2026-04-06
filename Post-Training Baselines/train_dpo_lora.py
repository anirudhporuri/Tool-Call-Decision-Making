#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from w2c_train_format import format_pref_example


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LoRA DPO on When2Call train_pref.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--model_family", type=str, choices=["llama", "gemma"], required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--dataset_name", type=str, default="nvidia/When2Call")
    p.add_argument("--dataset_config", type=str, default="train")
    p.add_argument("--dataset_split", type=str, default="pref")
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--val_size", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--max_prompt_length", type=int, default=1536)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default=None)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]




def load_source_dataset(args: argparse.Namespace):
    if args.train_file:
        ds = load_dataset("json", data_files=args.train_file)["train"]
    else:
        ds = load_dataset(args.dataset_name, args.dataset_config)[args.dataset_split]
    return ds


def preprocess_row(row: Dict[str, Any], model_family: str) -> Dict[str, str]:
    prompt, chosen, rejected = format_pref_example(model_family=model_family, row=row)
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    raw_ds = load_source_dataset(args)
    if args.max_train_samples:
        raw_ds = raw_ds.select(range(min(args.max_train_samples, len(raw_ds))))
    split = raw_ds.train_test_split(test_size=args.val_size, seed=args.seed, shuffle=True) if args.val_size > 0 else {"train": raw_ds}
    train_ds = split["train"]
    eval_ds = split.get("test")
    if eval_ds is not None and args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    train_ds = train_ds.map(
        lambda row: preprocess_row(row, args.model_family),
        remove_columns=train_ds.column_names,
        desc="Formatting DPO train prompts",
    )
    if eval_ds is not None:
        eval_ds = eval_ds.map(
            lambda row: preprocess_row(row, args.model_family),
            remove_columns=eval_ds.column_names,
            desc="Formatting DPO eval prompts",
        )

    with open(out_dir / "formatted_train_preview.jsonl", "w", encoding="utf-8") as f:
        for i in range(min(20, len(train_ds))):
            f.write(json.dumps(train_ds[i], ensure_ascii=False) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=get_dtype(args.dtype),
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        token=args.hf_token,
        torch_dtype=get_dtype(args.dtype),
        quantization_config=quant_config,
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

    dpo_args = DPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        seed=args.seed,
        report_to=args.report_to,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        remove_unused_columns=False,
    )

    try:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
        )

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(out_dir)

    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2)

    if eval_ds is not None:
        eval_metrics = trainer.evaluate()
        with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2)


if __name__ == "__main__":
    main()
