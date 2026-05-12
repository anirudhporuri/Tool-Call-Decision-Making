#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from w2c_eval_mcq import main as eval_main

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "local_datasets"

def env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def env_int(name, default):
    value = os.getenv(name)
    return int(value) if value is not None else default

def add_bool_flag(parser, name, default, help_text):
    dest = name[2:].replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", default=default, help=help_text)
    parser.add_argument(f"--no-{name[2:]}", dest=dest, action="store_false", help=argparse.SUPPRESS)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Friendly launcher for When2Call prompting evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_name_or_path")
    parser.add_argument("model_family", choices=["llama", "gemma"])
    parser.add_argument("num_shots", type=int)
    parser.add_argument("output_dir")
    parser.add_argument("fewshot_json", nargs="?", default=None)
    parser.add_argument("--dataset-dir", default=os.getenv("DATASET_DIR", str(DEFAULT_DATASET_DIR)))
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "bfloat16"))
    parser.add_argument("--attn-implementation", default=os.getenv("ATTN_IMPL"))
    parser.add_argument("--device-map", default=os.getenv("DEVICE_MAP", "auto"))
    parser.add_argument("--max-examples", type=int, default=env_int("MAX_EXAMPLES", None))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--dataset-name", default="nvidia/When2Call")
    parser.add_argument("--dataset-config", default="test")
    parser.add_argument("--dataset-split", default="mcq")
    parser.add_argument("--dry-run", action="store_true", default=env_flag("DRY_RUN", False))
    parser.add_argument("--smoke-run", action="store_true", default=env_flag("SMOKE_RUN", False))
    parser.add_argument("--dry-run-max-examples", type=int, default=env_int("DRY_RUN_MAX_EXAMPLES", 8))
    parser.add_argument("--smoke-run-max-examples", type=int, default=env_int("SMOKE_RUN_MAX_EXAMPLES", 8))
    add_bool_flag(parser, "--trust-remote-code", env_flag("TRUST_REMOTE_CODE", False), "Allow custom model code.")
    add_bool_flag(parser, "--save-prompt-text", env_flag("SAVE_PROMPT_TEXT", False), "Store full rendered prompts.")

    args = parser.parse_args(argv)
    if args.dry_run and args.smoke_run:
        parser.error("--dry-run and --smoke-run are mutually exclusive.")
    if args.num_shots > 0 and not args.fewshot_json:
        parser.error("fewshot_json is required when num_shots > 0.")
    return args

def resolve_max_examples(args):
    if args.max_examples is not None:
        return args.max_examples
    if args.dry_run:
        return args.dry_run_max_examples
    if args.smoke_run:
        return args.smoke_run_max_examples
    return None

def build_eval_argv(args):
    forwarded = [
        "--model_name_or_path",
        args.model_name_or_path,
        "--model_family",
        args.model_family,
        "--num_shots",
        str(args.num_shots),
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
        "--device_map",
        args.device_map,
        "--start_index",
        str(args.start_index),
    ]

    max_examples = resolve_max_examples(args)
    if max_examples is not None:
        forwarded.extend(["--max_examples", str(max_examples)])
    if args.fewshot_json:
        forwarded.extend(["--fewshot_json", args.fewshot_json])
    if args.hf_token:
        forwarded.extend(["--hf_token", args.hf_token])
    if args.attn_implementation:
        forwarded.extend(["--attn_implementation", args.attn_implementation])
    if args.dry_run:
        forwarded.append("--dry_run")
    if args.trust_remote_code:
        forwarded.append("--trust_remote_code")
    if args.save_prompt_text:
        forwarded.append("--save_prompt_text")
    return forwarded

def main(argv=None):
    args = parse_args(argv)
    eval_main(build_eval_argv(args))

if __name__ == "__main__":
    main()
