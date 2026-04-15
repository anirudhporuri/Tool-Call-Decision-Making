#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence

from train_dpo_lora import main as train_main


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "local_datasets"


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.getenv(name)
    return int(value) if value is not None else default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name[2:].replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", default=default, help=help_text)
    parser.add_argument(f"--no-{name[2:]}", dest=dest, action="store_false", help=argparse.SUPPRESS)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Friendly launcher for LoRA DPO training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_name_or_path")
    parser.add_argument("model_family", choices=["llama", "gemma"])
    parser.add_argument("output_dir")
    parser.add_argument("train_file", nargs="?", default=None)
    parser.add_argument("--dataset-dir", default=os.getenv("DATASET_DIR", str(DEFAULT_DATASET_DIR)))
    parser.add_argument("--dataset-name", default="nvidia/When2Call")
    parser.add_argument("--dataset-config", default="train_pref")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "bfloat16"))
    parser.add_argument("--attn-implementation", default=os.getenv("ATTN_IMPL"))
    parser.add_argument("--val-size", type=float, default=env_float("VAL_SIZE", 0.02))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))
    parser.add_argument("--max-train-samples", type=int, default=env_int("MAX_TRAIN_SAMPLES", None))
    parser.add_argument("--max-eval-samples", type=int, default=env_int("MAX_EVAL_SAMPLES", None))
    parser.add_argument("--max-steps", type=int, default=env_int("MAX_STEPS", None))
    parser.add_argument("--max-length", type=int, default=env_int("MAX_LENGTH", 2048))
    parser.add_argument("--max-prompt-length", type=int, default=env_int("MAX_PROMPT_LENGTH", 1536))
    parser.add_argument("--num-train-epochs", type=float, default=env_float("EPOCHS", 1.0))
    parser.add_argument("--learning-rate", type=float, default=env_float("LR", 5e-6))
    parser.add_argument("--weight-decay", type=float, default=env_float("WEIGHT_DECAY", 0.0))
    parser.add_argument("--warmup-steps", type=int, default=env_int("WARMUP_STEPS", 10))
    parser.add_argument("--beta", type=float, default=env_float("BETA", 0.1))
    parser.add_argument("--per-device-train-batch-size", type=int, default=env_int("TRAIN_BS", 2))
    parser.add_argument("--per-device-eval-batch-size", type=int, default=env_int("EVAL_BS", 2))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=env_int("GRAD_ACCUM", 8))
    parser.add_argument("--lora-r", type=int, default=env_int("LORA_R", 16))
    parser.add_argument("--lora-alpha", type=int, default=env_int("LORA_ALPHA", 32))
    parser.add_argument("--lora-dropout", type=float, default=env_float("LORA_DROPOUT", 0.05))
    parser.add_argument(
        "--target-modules",
        default=os.getenv("TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"),
    )
    parser.add_argument("--logging-steps", type=int, default=env_int("LOGGING_STEPS", None))
    parser.add_argument("--eval-steps", type=int, default=env_int("EVAL_STEPS", None))
    parser.add_argument("--save-steps", type=int, default=env_int("SAVE_STEPS", None))
    parser.add_argument("--report-to", default=os.getenv("REPORT_TO", "none"))
    parser.add_argument("--dry-run", action="store_true", default=env_flag("DRY_RUN", False))
    parser.add_argument("--smoke-run", action="store_true", default=env_flag("SMOKE_RUN", False))
    parser.add_argument("--dry-run-max-train-samples", type=int, default=env_int("DRY_RUN_MAX_TRAIN_SAMPLES", 32))
    parser.add_argument("--dry-run-max-eval-samples", type=int, default=env_int("DRY_RUN_MAX_EVAL_SAMPLES", 8))
    parser.add_argument("--smoke-run-max-train-samples", type=int, default=env_int("SMOKE_RUN_MAX_TRAIN_SAMPLES", 32))
    parser.add_argument("--smoke-run-max-eval-samples", type=int, default=env_int("SMOKE_RUN_MAX_EVAL_SAMPLES", 8))
    parser.add_argument("--smoke-run-max-steps", type=int, default=env_int("SMOKE_RUN_MAX_STEPS", 2))
    add_bool_flag(parser, "--load-in-4bit", env_flag("LOAD_IN_4BIT", True), "Load the base model in 4-bit.")
    add_bool_flag(
        parser,
        "--gradient-checkpointing",
        env_flag("GRADIENT_CHECKPOINTING", True),
        "Enable gradient checkpointing.",
    )
    add_bool_flag(parser, "--trust-remote-code", env_flag("TRUST_REMOTE_CODE", False), "Allow custom model code.")

    args = parser.parse_args(argv)
    if args.dry_run and args.smoke_run:
        parser.error("--dry-run and --smoke-run are mutually exclusive.")
    return args


def apply_mode_defaults(args: argparse.Namespace) -> None:
    if args.dry_run:
        if args.max_train_samples is None:
            args.max_train_samples = args.dry_run_max_train_samples
        if args.max_eval_samples is None:
            args.max_eval_samples = args.dry_run_max_eval_samples

    if args.smoke_run:
        if args.max_train_samples is None:
            args.max_train_samples = args.smoke_run_max_train_samples
        if args.max_eval_samples is None:
            args.max_eval_samples = args.smoke_run_max_eval_samples
        if args.max_steps is None:
            args.max_steps = args.smoke_run_max_steps
        if args.logging_steps is None:
            args.logging_steps = 1
        if args.eval_steps is None:
            args.eval_steps = 1
        if args.save_steps is None:
            args.save_steps = 1

    if args.max_steps is None:
        args.max_steps = -1
    if args.logging_steps is None:
        args.logging_steps = 10
    if args.eval_steps is None:
        args.eval_steps = 100
    if args.save_steps is None:
        args.save_steps = 100


def build_train_argv(args: argparse.Namespace) -> List[str]:
    forwarded = [
        "--model_name_or_path",
        args.model_name_or_path,
        "--model_family",
        args.model_family,
        "--output_dir",
        args.output_dir,
        "--dataset_name",
        args.dataset_name,
        "--dataset_config",
        args.dataset_config,
        "--dataset_split",
        args.dataset_split,
        "--dataset_dir",
        args.dataset_dir,
        "--dtype",
        args.dtype,
        "--val_size",
        str(args.val_size),
        "--seed",
        str(args.seed),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--warmup_steps",
        str(args.warmup_steps),
        "--beta",
        str(args.beta),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        str(args.lora_dropout),
        "--target_modules",
        args.target_modules,
        "--max_length",
        str(args.max_length),
        "--max_prompt_length",
        str(args.max_prompt_length),
        "--logging_steps",
        str(args.logging_steps),
        "--eval_steps",
        str(args.eval_steps),
        "--save_steps",
        str(args.save_steps),
        "--max_steps",
        str(args.max_steps),
        "--report_to",
        args.report_to,
    ]

    if args.train_file:
        forwarded.extend(["--train_file", args.train_file])
    if args.hf_token:
        forwarded.extend(["--hf_token", args.hf_token])
    if args.attn_implementation:
        forwarded.extend(["--attn_implementation", args.attn_implementation])
    if args.max_train_samples is not None:
        forwarded.extend(["--max_train_samples", str(args.max_train_samples)])
    if args.max_eval_samples is not None:
        forwarded.extend(["--max_eval_samples", str(args.max_eval_samples)])
    if args.dry_run:
        forwarded.append("--dry_run")
    if args.load_in_4bit:
        forwarded.append("--load_in_4bit")
    if args.gradient_checkpointing:
        forwarded.append("--gradient_checkpointing")
    if args.trust_remote_code:
        forwarded.append("--trust_remote_code")
    return forwarded


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    apply_mode_defaults(args)
    train_main(build_train_argv(args))


if __name__ == "__main__":
    main()
