#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "local_datasets"
DEFAULT_FEWSHOT_JSON = REPO_ROOT / "Prompting Baselines" / "fewshot_examples.template.json"
DEFAULT_TRAIN_SOURCE_JSONLS = [
    REPO_ROOT / "CAI" / "generated_datasets" / "train_pref_cai_sft_source.jsonl",
    REPO_ROOT / "CAI" / "generated_datasets" / "train_pref_cai_dpo_source.jsonl",
]

def env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def env_int(name, default):
    value = os.getenv(name)
    return int(value) if value is not None else default

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Friendly launcher for the When2Call hidden-state probe pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_name_or_path")
    parser.add_argument("model_family", choices=["llama", "gemma"])
    parser.add_argument("output_dir")
    parser.add_argument("eval_samples_jsonl")
    parser.add_argument(
        "--peft-base-model-override",
        default=None,
        help="Optional base-model path/ID to use when model_name_or_path is a PEFT adapter checkpoint.",
    )
    parser.add_argument("--train-source-jsonls", nargs="+", default=[str(path) for path in DEFAULT_TRAIN_SOURCE_JSONLS])
    parser.add_argument("--dataset-dir", default=os.getenv("DATASET_DIR", str(DEFAULT_DATASET_DIR)))
    parser.add_argument("--model-cache-dir", default=os.getenv("MODEL_CACHE_DIR", str(REPO_ROOT / "cluster_cache" / "model_cache")))
    parser.add_argument("--hf-home-dir", default=os.getenv("HF_HOME_DIR", str(REPO_ROOT / "cluster_cache" / "hf_home")))
    parser.add_argument("--fewshot-json", default=None)
    parser.add_argument("--num-shots", type=int, default=0)
    parser.add_argument("--use-4shot-prompt", action="store_true", default=env_flag("USE_4SHOT_PROMPT", False))
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "bfloat16"))
    parser.add_argument("--attn-implementation", default=os.getenv("ATTN_IMPL"))
    parser.add_argument("--device-map", default=os.getenv("DEVICE_MAP", "auto"))
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", 4))
    parser.add_argument("--max-length", type=int, default=env_int("MAX_LENGTH", 2048))
    parser.add_argument("--max-train-examples", type=int, default=env_int("MAX_TRAIN_EXAMPLES", None))
    parser.add_argument("--max-test-examples", type=int, default=env_int("MAX_TEST_EXAMPLES", None))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--dev-size", type=float, default=float(os.getenv("DEV_SIZE", "0.2")))
    parser.add_argument("--random-seed", type=int, default=env_int("RANDOM_SEED", 42))
    parser.add_argument("--max-iter", type=int, default=env_int("MAX_ITER", 4000))
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--dataset-name", default="nvidia/When2Call")
    parser.add_argument("--dataset-config", default="test")
    parser.add_argument("--dataset-split", default="mcq")
    parser.add_argument("--reuse-features", action="store_true", default=env_flag("REUSE_FEATURES", False))
    parser.add_argument("--save-prompt-text", action="store_true", default=env_flag("SAVE_PROMPT_TEXT", False))
    parser.add_argument("--trust-remote-code", action="store_true", default=env_flag("TRUST_REMOTE_CODE", False))
    parser.add_argument("--load-in-4bit", action="store_true", default=env_flag("LOAD_IN_4BIT", False))
    parser.add_argument("--prefetch-models", action="store_true", default=env_flag("PREFETCH_MODELS", True))
    parser.add_argument("--no-prefetch-models", dest="prefetch_models", action="store_false")

    args = parser.parse_args(argv)
    if args.use_4shot_prompt:
        if args.num_shots not in {0, 4}:
            parser.error("--use-4shot-prompt is only compatible with --num-shots 4.")
        args.num_shots = 4
        if not args.fewshot_json:
            args.fewshot_json = str(DEFAULT_FEWSHOT_JSON)
    if args.num_shots > 0 and not args.fewshot_json:
        parser.error("--fewshot-json is required when --num-shots > 0.")
    return args

def build_probe_argv(args):
    forwarded = [
        "--model_name_or_path",
        args.model_name_or_path,
        "--model_family",
        args.model_family,
        "--output_dir",
        args.output_dir,
        "--eval_samples_jsonl",
        args.eval_samples_jsonl,
        "--dataset_dir",
        args.dataset_dir,
        "--model_cache_dir",
        args.model_cache_dir,
        "--hf_home_dir",
        args.hf_home_dir,
        "--dtype",
        args.dtype,
        "--device_map",
        args.device_map,
        "--batch_size",
        str(args.batch_size),
        "--max_length",
        str(args.max_length),
        "--start_index",
        str(args.start_index),
        "--dev_size",
        str(args.dev_size),
        "--random_seed",
        str(args.random_seed),
        "--max_iter",
        str(args.max_iter),
        "--dataset_name",
        args.dataset_name,
        "--dataset_config",
        args.dataset_config,
        "--dataset_split",
        args.dataset_split,
        "--num_shots",
        str(args.num_shots),
        "--train_source_jsonls",
        *args.train_source_jsonls,
        "--c_values",
        *[str(value) for value in args.c_values],
    ]

    if args.peft_base_model_override:
        forwarded.extend(["--peft_base_model_override", args.peft_base_model_override])
    if args.fewshot_json:
        forwarded.extend(["--fewshot_json", args.fewshot_json])
    if args.hf_token:
        forwarded.extend(["--hf_token", args.hf_token])
    if args.attn_implementation:
        forwarded.extend(["--attn_implementation", args.attn_implementation])
    if args.max_train_examples is not None:
        forwarded.extend(["--max_train_examples", str(args.max_train_examples)])
    if args.max_test_examples is not None:
        forwarded.extend(["--max_test_examples", str(args.max_test_examples)])
    if args.reuse_features:
        forwarded.append("--reuse_features")
    if args.save_prompt_text:
        forwarded.append("--save_prompt_text")
    if args.trust_remote_code:
        forwarded.append("--trust_remote_code")
    if args.load_in_4bit:
        forwarded.append("--load_in_4bit")
    if args.use_4shot_prompt:
        forwarded.append("--use_4shot_prompt")
    if args.prefetch_models:
        forwarded.append("--prefetch_models")
    return forwarded

def main(argv=None):
    args = parse_args(argv)
    from w2c_probe import main as probe_main

    probe_main(build_probe_argv(args))

if __name__ == "__main__":
    main()
