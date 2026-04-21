#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - surfaced at runtime on cluster.
    snapshot_download = None


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = Path("/fs/classhomes/mukunds/Tool-Call-Decision-Making")
CAI_DIR = REPO_ROOT / "CAI"
PT_DIR = REPO_ROOT / "Post-Training Baselines"
EVAL_DIR = REPO_ROOT / "Prompting Baselines"
SOURCE_JSONL = REPO_ROOT / "Data_Management" / "generated_datasets" / "when2call_balanced_sft.jsonl"
MODEL_CACHE_DIR = REPO_ROOT / "cluster_cache" / "model_cache"
HF_HOME_DIR = REPO_ROOT / "cluster_cache" / "hf_home"


def timestamp() -> str:
    return subprocess.check_output(["date", "+%Y-%m-%d %H:%M:%S"], text=True).strip()


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_model_name(model_name_or_path: str) -> str:
    return (
        model_name_or_path.strip()
        .replace("/", "__")
        .replace(":", "_")
        .replace("@", "_")
        .replace(" ", "_")
    )


def local_or_cached_model_path(model_name_or_path: str, cache_root: Path, hf_token: Optional[str]) -> str:
    candidate = Path(model_name_or_path).expanduser()
    if candidate.exists():
        resolved = str(candidate.resolve())
        log(f"Model already available locally: {resolved}")
        return resolved

    if candidate.is_absolute():
        raise FileNotFoundError(
            "Model path does not exist on disk: "
            f"{candidate}. If this is meant to be a local checkpoint, verify the directory name/path."
        )

    local_dir = ensure_dir(cache_root / sanitize_model_name(model_name_or_path))
    if any(local_dir.iterdir()):
        resolved = str(local_dir.resolve())
        log(f"Model found in local cache: {resolved}")
        return resolved

    if snapshot_download is None:
        raise ImportError("huggingface_hub is required for model prefetching.")

    log(f"Prefetching model {model_name_or_path} into {local_dir}")
    snapshot_download(
        repo_id=model_name_or_path,
        local_dir=str(local_dir),
        token=hf_token,
        resume_download=True,
    )
    return str(local_dir.resolve())


def resolve_model_reference(
    model_name_or_path: str,
    cache_root: Path,
    hf_token: Optional[str],
    prefetch_models: bool,
) -> str:
    candidate = Path(model_name_or_path).expanduser()
    if candidate.exists():
        resolved = str(candidate.resolve())
        log(f"Model already available locally: {resolved}")
        return resolved

    if candidate.is_absolute():
        raise FileNotFoundError(
            "Model path does not exist on disk: "
            f"{candidate}. If this is meant to be a local checkpoint, verify the directory name/path."
        )

    if prefetch_models:
        return local_or_cached_model_path(model_name_or_path, cache_root, hf_token)
    return model_name_or_path


def build_launcher(mode: str) -> List[str]:
    if mode == "direct":
        return []
    if mode == "srun":
        return ["srun", "--unbuffered", "--ntasks=1"]
    if shutil.which("srun") and os.getenv("SLURM_JOB_ID"):
        return ["srun", "--unbuffered", "--ntasks=1"]
    return []


def run_step(label: str, workdir: Path, command: Sequence[str], env: dict[str, str], launcher: Sequence[str]) -> None:
    full_cmd = [*launcher, *command]
    log(f"START {label}")
    log(f"WORKDIR {workdir}")
    log(f"CMD {shell_join(full_cmd)}")
    subprocess.run(full_cmd, cwd=str(workdir), env=env, check=True)
    log(f"DONE {label}")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def jsonl_has_rows(path: Path, min_rows: int = 1) -> bool:
    return path.is_file() and count_lines(path) >= min_rows


def jsonl_has_exact_rows(path: Path, expected_rows: int) -> bool:
    return path.is_file() and count_lines(path) == expected_rows


def all_paths_exist(paths: Sequence[Path]) -> bool:
    return all(path.is_file() for path in paths)


def run_step_if_needed(
    *,
    skip_completed: bool,
    label: str,
    workdir: Path,
    command: Sequence[str],
    env: dict[str, str],
    launcher: Sequence[str],
    is_complete: Callable[[], bool],
    completion_note: str,
) -> None:
    if skip_completed and is_complete():
        log(f"SKIP {label} ({completion_note})")
        return
    run_step(label, workdir, command, env, launcher)


def maybe_extend(command: List[str], flag: str, value: Optional[str]) -> None:
    if value:
        command.extend([flag, value])


def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name[2:].replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", default=default, help=help_text)
    parser.add_argument(f"--no-{name[2:]}", dest=dest, action="store_false", help=argparse.SUPPRESS)


def build_model_flags(
    *,
    load_in_4bit: bool,
    trust_remote_code: bool,
    dtype: Optional[str],
    attn_implementation: Optional[str],
    hf_token: Optional[str],
) -> List[str]:
    flags: List[str] = []
    if load_in_4bit:
        flags.append("--load-in-4bit")
    if trust_remote_code:
        flags.append("--trust-remote-code")
    maybe_extend(flags, "--dtype", dtype)
    maybe_extend(flags, "--attn-implementation", attn_implementation)
    if hf_token:
        flags.extend(["--hf-token", hf_token])
    return flags


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full staged CAI -> SFT -> Eval -> DPO pipeline on the cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-family", required=True, choices=["llama", "gemma"])
    parser.add_argument("--base-tag", required=True)
    parser.add_argument("--critic-model", default=os.getenv("CRITIC_MODEL", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--critic-family", default=os.getenv("CRITIC_FAMILY", "qwen"), choices=["llama", "gemma", "qwen", "gpt-oss"])
    parser.add_argument("--critic-tag", default=os.getenv("CRITIC_TAG", "qwen3p5_9b"))
    parser.add_argument("--run-tag", default=os.getenv("RUN_TAG"))
    parser.add_argument("--sft-model-tag", default=os.getenv("SFT_MODEL_TAG"))
    parser.add_argument("--dpo-base-model-tag", default=os.getenv("DPO_BASE_MODEL_TAG"))
    parser.add_argument("--dpo-sft-model-tag", default=os.getenv("DPO_SFT_MODEL_TAG"))
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "bfloat16"))
    parser.add_argument("--attn-implementation", default=os.getenv("ATTN_IMPL"))
    parser.add_argument("--base-generation-batch-size", type=int, default=env_int("BASE_GENERATION_BATCH_SIZE", 2))
    parser.add_argument("--critic-batch-size", type=int, default=env_int("CRITIC_BATCH_SIZE", 2))
    parser.add_argument("--base-generation-max-prompt-length", type=int, default=env_int("BASE_GENERATION_MAX_PROMPT_LENGTH", 1024))
    parser.add_argument("--critic-max-prompt-length", type=int, default=env_int("CRITIC_MAX_PROMPT_LENGTH", 1024))
    parser.add_argument("--critique-max-new-tokens", type=int, default=int(os.getenv("CRITIQUE_MAX_NEW_TOKENS", "256")))
    parser.add_argument("--judge-max-new-tokens", type=int, default=int(os.getenv("JUDGE_MAX_NEW_TOKENS", "128")))
    parser.add_argument("--pair-temperature", type=float, default=float(os.getenv("PAIR_TEMPERATURE", "0.9")))
    parser.add_argument("--pair-top-p", type=float, default=float(os.getenv("PAIR_TOP_P", "0.95")))
    parser.add_argument("--pair-candidate-attempts", type=int, default=int(os.getenv("PAIR_CANDIDATE_ATTEMPTS", "6")))
    parser.add_argument("--eval-num-shots", type=int, default=int(os.getenv("EVAL_NUM_SHOTS", "0")))
    parser.add_argument("--launcher", choices=["auto", "srun", "direct"], default=os.getenv("W2C_LAUNCHER", "auto"))
    add_bool_flag(parser, "--prefetch-models", env_flag("PREFETCH_MODELS", True), "Snapshot base and critic models into the local cache before running.")
    add_bool_flag(
        parser,
        "--base-generation-load-in-4bit",
        env_flag("BASE_GENERATION_LOAD_IN_4BIT", env_flag("LOAD_IN_4BIT", True)),
        "Load the base model in 4-bit for initial outputs, revisions, and DPO pair generation.",
    )
    add_bool_flag(
        parser,
        "--critic-load-in-4bit",
        env_flag("CRITIC_LOAD_IN_4BIT", env_flag("LOAD_IN_4BIT", True)),
        "Load the critic/judge model in 4-bit for SFT critiques and DPO judging.",
    )
    add_bool_flag(
        parser,
        "--training-load-in-4bit",
        env_flag("TRAINING_LOAD_IN_4BIT", env_flag("LOAD_IN_4BIT", True)),
        "Load the SFT/DPO training base model in 4-bit.",
    )
    add_bool_flag(parser, "--skip-completed", env_flag("SKIP_COMPLETED", True), "Skip pipeline stages whose expected output artifacts already exist.")
    add_bool_flag(parser, "--trust-remote-code", env_flag("TRUST_REMOTE_CODE", False), "Allow custom model code in Transformers.")
    parser.add_argument("--load-in-4bit", dest="legacy_load_in_4bit", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-load-in-4bit", dest="legacy_load_in_4bit", action="store_false", help=argparse.SUPPRESS)

    args = parser.parse_args(argv)
    if args.legacy_load_in_4bit is not None:
        args.base_generation_load_in_4bit = args.legacy_load_in_4bit
        args.critic_load_in_4bit = args.legacy_load_in_4bit
        args.training_load_in_4bit = args.legacy_load_in_4bit
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    repo_root = REPO_ROOT
    cai_dir = CAI_DIR
    pt_dir = PT_DIR
    eval_dir = EVAL_DIR
    source_jsonl = SOURCE_JSONL
    model_cache_dir = ensure_dir(MODEL_CACHE_DIR)
    hf_home_dir = ensure_dir(HF_HOME_DIR)

    run_tag = args.run_tag or f"{args.base_tag}_{args.critic_tag}_full"
    sft_model_tag = args.sft_model_tag or f"{run_tag}_sft_model"
    dpo_base_model_tag = args.dpo_base_model_tag or f"{run_tag}_dpo_base_model"
    dpo_sft_model_tag = args.dpo_sft_model_tag or f"{run_tag}_dpo_from_sft_model"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HOME"] = str(hf_home_dir)
    env.setdefault("HF_HUB_CACHE", str(hf_home_dir / "hub"))
    env.setdefault("HF_DATASETS_CACHE", str(hf_home_dir / "datasets"))
    env.setdefault("TRANSFORMERS_CACHE", str(hf_home_dir / "transformers"))
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token

    launcher = build_launcher(args.launcher)

    log(
        "Job info: "
        f"id={os.getenv('SLURM_JOB_ID', 'none')} "
        f"name={os.getenv('SLURM_JOB_NAME', 'none')} "
        f"host={socket.gethostname()}"
    )
    log(f"Repository root: {repo_root}")
    log(f"Source JSONL: {source_jsonl}")
    log(f"Model cache dir: {model_cache_dir}")
    log(f"HF_HOME dir: {hf_home_dir}")
    log(
        "Precision routing: "
        f"base_generation_load_in_4bit={args.base_generation_load_in_4bit} "
        f"critic_load_in_4bit={args.critic_load_in_4bit} "
        f"training_load_in_4bit={args.training_load_in_4bit}"
    )
    log(
        "Batch routing: "
        f"base_generation_batch_size={args.base_generation_batch_size} "
        f"critic_batch_size={args.critic_batch_size} "
        f"base_generation_max_prompt_length={args.base_generation_max_prompt_length} "
        f"critic_max_prompt_length={args.critic_max_prompt_length}"
    )

    run_step("Python version", repo_root, [sys.executable, "--version"], env, [])
    if shutil.which("nvidia-smi"):
        run_step("GPU status", repo_root, ["nvidia-smi"], env, [])

    base_model_path = resolve_model_reference(
        args.base_model,
        model_cache_dir,
        args.hf_token,
        args.prefetch_models,
    )
    critic_model_path = resolve_model_reference(
        args.critic_model,
        model_cache_dir,
        args.hf_token,
        args.prefetch_models,
    )

    log(f"Base model path: {base_model_path}")
    log(f"Critic model path: {critic_model_path}")
    log(f"Run tag: {run_tag}")

    sft_source = cai_dir / "generated_datasets" / "train_pref_cai_sft_source.jsonl"
    dpo_source = cai_dir / "generated_datasets" / "train_pref_cai_dpo_source.jsonl"
    split_summary = cai_dir / "generated_datasets" / "train_pref_cai_split_summary.json"
    run_step_if_needed(
        skip_completed=args.skip_completed,
        label="CAI split",
        workdir=cai_dir,
        command=[
            sys.executable,
            "-u",
            "run_cai_split.py",
            "--source-jsonl",
            str(source_jsonl),
        ],
        env=env,
        launcher=launcher,
        is_complete=lambda: all_paths_exist([sft_source, dpo_source, split_summary]),
        completion_note=f"found {sft_source.name}, {dpo_source.name}, and {split_summary.name}",
    )

    full_sft_rows = count_lines(sft_source)
    full_dpo_rows = count_lines(dpo_source)
    log(f"SFT source rows: {full_sft_rows}")
    log(f"DPO source rows: {full_dpo_rows}")

    base_generation_flags = build_model_flags(
        load_in_4bit=args.base_generation_load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        hf_token=args.hf_token,
    )
    critic_generation_flags = build_model_flags(
        load_in_4bit=args.critic_load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        hf_token=args.hf_token,
    )
    training_flags = build_model_flags(
        load_in_4bit=args.training_load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        hf_token=args.hf_token,
    )

    eval_flags: List[str] = []
    if args.trust_remote_code:
        eval_flags.append("--trust-remote-code")
    maybe_extend(eval_flags, "--dtype", args.dtype)
    maybe_extend(eval_flags, "--attn-implementation", args.attn_implementation)
    if args.hf_token:
        eval_flags.extend(["--hf-token", args.hf_token])

    sft_initial_jsonl = cai_dir / "outputs" / run_tag / "sft_initial" / "initial_outputs.jsonl"
    sft_initial_summary = cai_dir / "outputs" / run_tag / "sft_initial" / "summary.json"
    run_step_if_needed(
        skip_completed=args.skip_completed,
        label="SFT initial outputs",
        workdir=cai_dir,
        command=[
            sys.executable,
            "-u",
            "run_cai_initial_outputs.py",
            base_model_path,
            args.base_family,
            f"outputs/{run_tag}/sft_initial",
            str(sft_source),
            "--max-examples",
            str(full_sft_rows),
            "--batch-size",
            str(args.base_generation_batch_size),
            "--max-prompt-length",
            str(args.base_generation_max_prompt_length),
            *base_generation_flags,
        ],
        env=env,
        launcher=launcher,
        is_complete=lambda: all_paths_exist([sft_initial_jsonl, sft_initial_summary])
        and jsonl_has_exact_rows(sft_initial_jsonl, full_sft_rows),
        completion_note=f"found complete {sft_initial_jsonl.name} ({full_sft_rows} rows)",
    )

    sft_critiques_jsonl = cai_dir / "outputs" / run_tag / "sft_critiques" / "critiques.jsonl"
    sft_critiques_summary = cai_dir / "outputs" / run_tag / "sft_critiques" / "summary.json"
    run_step_if_needed(
        skip_completed=args.skip_completed,
        label="SFT critiques",
        workdir=cai_dir,
        command=[
            sys.executable,
            "-u",
            "run_cai_critiques.py",
            critic_model_path,
            args.critic_family,
            f"outputs/{run_tag}/sft_critiques",
            str(sft_source),
            f"outputs/{run_tag}/sft_initial/initial_outputs.jsonl",
            "--max-examples",
            str(full_sft_rows),
            "--max-new-tokens",
            str(args.critique_max_new_tokens),
            "--batch-size",
            str(args.critic_batch_size),
            "--max-prompt-length",
            str(args.critic_max_prompt_length),
            *critic_generation_flags,
        ],
        env=env,
        launcher=launcher,
        is_complete=lambda: all_paths_exist([sft_critiques_jsonl, sft_critiques_summary])
        and jsonl_has_exact_rows(sft_critiques_jsonl, full_sft_rows),
        completion_note=f"found complete {sft_critiques_jsonl.name} ({full_sft_rows} rows)",
    )

    sft_revisions_jsonl = cai_dir / "outputs" / run_tag / "sft_revisions" / "revisions.jsonl"
    sft_revisions_summary = cai_dir / "outputs" / run_tag / "sft_revisions" / "summary.json"
    run_step_if_needed(
        skip_completed=args.skip_completed,
        label="SFT revisions",
        workdir=cai_dir,
        command=[
            sys.executable,
            "-u",
            "run_cai_revisions.py",
            base_model_path,
            args.base_family,
            f"outputs/{run_tag}/sft_revisions",
            str(sft_source),
            f"outputs/{run_tag}/sft_initial/initial_outputs.jsonl",
            f"outputs/{run_tag}/sft_critiques/critiques.jsonl",
            "--max-examples",
            str(full_sft_rows),
            "--batch-size",
            str(args.base_generation_batch_size),
            "--max-prompt-length",
            str(args.base_generation_max_prompt_length),
            *base_generation_flags,
        ],
        env=env,
        launcher=launcher,
        is_complete=lambda: all_paths_exist([sft_revisions_jsonl, sft_revisions_summary])
        and jsonl_has_exact_rows(sft_revisions_jsonl, full_sft_rows),
        completion_note=f"found complete {sft_revisions_jsonl.name} ({full_sft_rows} rows)",
    )

    sft_dataset_jsonl = cai_dir / "outputs" / run_tag / "sft_dataset" / "cai_sft_dataset.jsonl"
    sft_dataset_summary = cai_dir / "outputs" / run_tag / "sft_dataset" / "summary.json"
    run_step_if_needed(
        skip_completed=args.skip_completed,
        label="Build CAI SFT dataset",
        workdir=cai_dir,
        command=[
            sys.executable,
            "-u",
            "build_cai_sft_dataset.py",
            f"outputs/{run_tag}/sft_dataset",
            str(sft_source),
            f"outputs/{run_tag}/sft_initial/initial_outputs.jsonl",
            f"outputs/{run_tag}/sft_critiques/critiques.jsonl",
            f"outputs/{run_tag}/sft_revisions/revisions.jsonl",
            "--max-examples",
            str(full_sft_rows),
        ],
        env=env,
        launcher=launcher,
        is_complete=lambda: all_paths_exist([sft_dataset_jsonl, sft_dataset_summary]) and jsonl_has_rows(sft_dataset_jsonl),
        completion_note=f"found {sft_dataset_jsonl.name} and {sft_dataset_summary.name}",
    )
    log(f"CAI SFT dataset rows: {count_lines(sft_dataset_jsonl)}")

    sft_adapter_dir = pt_dir / "outputs" / sft_model_tag
    run_step_if_needed(
        skip_completed=args.skip_completed,
        label="Train SFT adapter",
        workdir=pt_dir,
        command=[
            sys.executable,
            "-u",
            "run_sft.py",
            base_model_path,
            args.base_family,
            f"outputs/{sft_model_tag}",
            str(sft_dataset_jsonl),
            *training_flags,
        ],
        env=env,
        launcher=launcher,
        is_complete=lambda: all_paths_exist([sft_adapter_dir / "adapter_config.json", sft_adapter_dir / "tokenizer_config.json"]),
        completion_note=f"found adapter_config.json in {sft_adapter_dir.name}",
    )

    sft_eval_dir = eval_dir / "outputs" / f"{sft_model_tag}_eval"
    run_step_if_needed(
        skip_completed=args.skip_completed,
        label="Eval SFT adapter",
        workdir=eval_dir,
        command=[
            sys.executable,
            "-u",
            "run_eval.py",
            str(sft_adapter_dir),
            args.base_family,
            str(args.eval_num_shots),
            f"outputs/{sft_model_tag}_eval",
            *eval_flags,
        ],
        env=env,
        launcher=launcher,
        is_complete=lambda: all_paths_exist([sft_eval_dir / "summary.json", sft_eval_dir / "samples.jsonl"]),
        completion_note=f"found summary.json and samples.jsonl in {sft_eval_dir.name}",
    )

    dpo_base_pairs_jsonl = cai_dir / "outputs" / run_tag / "dpo_base_pairs" / "response_pairs.jsonl"
    dpo_base_pairs_summary = cai_dir / "outputs" / run_tag / "dpo_base_pairs" / "summary.json"
    run_step_if_needed(
        skip_completed=args.skip_completed,
        label="DPO response pairs from base model",
        workdir=cai_dir,
        command=[
            sys.executable,
            "-u",
            "run_cai_response_pairs.py",
            base_model_path,
            args.base_family,
            f"outputs/{run_tag}/dpo_base_pairs",
            str(dpo_source),
            "--max-examples",
            str(full_dpo_rows),
            "--temperature",
            str(args.pair_temperature),
            "--top-p",
            str(args.pair_top_p),
            "--candidate-attempts",
            str(args.pair_candidate_attempts),
            "--batch-size",
            str(args.base_generation_batch_size),
            "--max-prompt-length",
            str(args.base_generation_max_prompt_length),
            *base_generation_flags,
        ],
        env=env,
        launcher=launcher,
        is_complete=lambda: all_paths_exist([dpo_base_pairs_jsonl, dpo_base_pairs_summary])
        and jsonl_has_exact_rows(dpo_base_pairs_jsonl, full_dpo_rows),
        completion_note=f"found complete {dpo_base_pairs_jsonl.name} ({full_dpo_rows} rows)",
    )

    dpo_base_dataset_jsonl = cai_dir / "outputs" / run_tag / "dpo_base_dataset" / "cai_dpo_dataset.jsonl"
    dpo_base_master_jsonl = cai_dir / "outputs" / run_tag / "dpo_base_dataset" / "master_records.jsonl"
    dpo_base_dataset_summary = cai_dir / "outputs" / run_tag / "dpo_base_dataset" / "summary.json"
    run_step_if_needed(
        skip_completed=args.skip_completed,
        label="Judge DPO base pairs",
        workdir=cai_dir,
        command=[
            sys.executable,
            "-u",
            "run_cai_preferences.py",
            critic_model_path,
            args.critic_family,
            f"outputs/{run_tag}/dpo_base_dataset",
            str(dpo_source),
            f"outputs/{run_tag}/dpo_base_pairs/response_pairs.jsonl",
            "--max-examples",
            str(full_dpo_rows),
            "--max-new-tokens",
            str(args.judge_max_new_tokens),
            "--batch-size",
            str(args.critic_batch_size),
            "--max-prompt-length",
            str(args.critic_max_prompt_length),
            *critic_generation_flags,
        ],
        env=env,
        launcher=launcher,
        is_complete=lambda: all_paths_exist([dpo_base_dataset_jsonl, dpo_base_master_jsonl, dpo_base_dataset_summary])
        and jsonl_has_exact_rows(dpo_base_master_jsonl, full_dpo_rows)
        and jsonl_has_rows(dpo_base_dataset_jsonl),
        completion_note=f"found complete {dpo_base_master_jsonl.name} and exported DPO dataset",
    )
    log(f"CAI DPO base dataset rows: {count_lines(dpo_base_dataset_jsonl)}")

    dpo_base_adapter_dir = pt_dir / "outputs" / dpo_base_model_tag
    run_step_if_needed(
        skip_completed=args.skip_completed,
        label="Train DPO adapter from base model",
        workdir=pt_dir,
        command=[
            sys.executable,
            "-u",
            "run_dpo.py",
            base_model_path,
            args.base_family,
            f"outputs/{dpo_base_model_tag}",
            str(dpo_base_dataset_jsonl),
            *training_flags,
        ],
        env=env,
        launcher=launcher,
        is_complete=lambda: all_paths_exist([dpo_base_adapter_dir / "adapter_config.json", dpo_base_adapter_dir / "tokenizer_config.json"]),
        completion_note=f"found adapter_config.json in {dpo_base_adapter_dir.name}",
    )

    dpo_base_eval_dir = eval_dir / "outputs" / f"{dpo_base_model_tag}_eval"
    run_step_if_needed(
        skip_completed=args.skip_completed,
        label="Eval DPO adapter from base model",
        workdir=eval_dir,
        command=[
            sys.executable,
            "-u",
            "run_eval.py",
            str(dpo_base_adapter_dir),
            args.base_family,
            str(args.eval_num_shots),
            f"outputs/{dpo_base_model_tag}_eval",
            *eval_flags,
        ],
        env=env,
        launcher=launcher,
        is_complete=lambda: all_paths_exist([dpo_base_eval_dir / "summary.json", dpo_base_eval_dir / "samples.jsonl"]),
        completion_note=f"found summary.json and samples.jsonl in {dpo_base_eval_dir.name}",
    )

    dpo_sft_pairs_jsonl = cai_dir / "outputs" / run_tag / "dpo_sft_pairs" / "response_pairs.jsonl"
    dpo_sft_pairs_summary = cai_dir / "outputs" / run_tag / "dpo_sft_pairs" / "summary.json"
    run_step_if_needed(
        skip_completed=args.skip_completed,
        label="DPO response pairs from SFT adapter",
        workdir=cai_dir,
        command=[
            sys.executable,
            "-u",
            "run_cai_response_pairs.py",
            str(sft_adapter_dir),
            args.base_family,
            f"outputs/{run_tag}/dpo_sft_pairs",
            str(dpo_source),
            "--max-examples",
            str(full_dpo_rows),
            "--temperature",
            str(args.pair_temperature),
            "--top-p",
            str(args.pair_top_p),
            "--candidate-attempts",
            str(args.pair_candidate_attempts),
            "--batch-size",
            str(args.base_generation_batch_size),
            "--max-prompt-length",
            str(args.base_generation_max_prompt_length),
            *base_generation_flags,
        ],
        env=env,
        launcher=launcher,
        is_complete=lambda: all_paths_exist([dpo_sft_pairs_jsonl, dpo_sft_pairs_summary])
        and jsonl_has_exact_rows(dpo_sft_pairs_jsonl, full_dpo_rows),
        completion_note=f"found complete {dpo_sft_pairs_jsonl.name} ({full_dpo_rows} rows)",
    )

    dpo_sft_dataset_jsonl = cai_dir / "outputs" / run_tag / "dpo_sft_dataset" / "cai_dpo_dataset.jsonl"
    dpo_sft_master_jsonl = cai_dir / "outputs" / run_tag / "dpo_sft_dataset" / "master_records.jsonl"
    dpo_sft_dataset_summary = cai_dir / "outputs" / run_tag / "dpo_sft_dataset" / "summary.json"
    run_step_if_needed(
        skip_completed=args.skip_completed,
        label="Judge DPO SFT pairs",
        workdir=cai_dir,
        command=[
            sys.executable,
            "-u",
            "run_cai_preferences.py",
            critic_model_path,
            args.critic_family,
            f"outputs/{run_tag}/dpo_sft_dataset",
            str(dpo_source),
            f"outputs/{run_tag}/dpo_sft_pairs/response_pairs.jsonl",
            "--max-examples",
            str(full_dpo_rows),
            "--max-new-tokens",
            str(args.judge_max_new_tokens),
            "--batch-size",
            str(args.critic_batch_size),
            "--max-prompt-length",
            str(args.critic_max_prompt_length),
            *critic_generation_flags,
        ],
        env=env,
        launcher=launcher,
        is_complete=lambda: all_paths_exist([dpo_sft_dataset_jsonl, dpo_sft_master_jsonl, dpo_sft_dataset_summary])
        and jsonl_has_exact_rows(dpo_sft_master_jsonl, full_dpo_rows)
        and jsonl_has_rows(dpo_sft_dataset_jsonl),
        completion_note=f"found complete {dpo_sft_master_jsonl.name} and exported DPO dataset",
    )
    log(f"CAI DPO SFT dataset rows: {count_lines(dpo_sft_dataset_jsonl)}")

    dpo_sft_adapter_dir = pt_dir / "outputs" / dpo_sft_model_tag
    run_step_if_needed(
        skip_completed=args.skip_completed,
        label="Train DPO adapter from SFT adapter",
        workdir=pt_dir,
        command=[
            sys.executable,
            "-u",
            "run_dpo.py",
            str(sft_adapter_dir),
            args.base_family,
            f"outputs/{dpo_sft_model_tag}",
            str(dpo_sft_dataset_jsonl),
            *training_flags,
        ],
        env=env,
        launcher=launcher,
        is_complete=lambda: all_paths_exist([dpo_sft_adapter_dir / "adapter_config.json", dpo_sft_adapter_dir / "tokenizer_config.json"]),
        completion_note=f"found adapter_config.json in {dpo_sft_adapter_dir.name}",
    )

    dpo_sft_eval_dir = eval_dir / "outputs" / f"{dpo_sft_model_tag}_eval"
    run_step_if_needed(
        skip_completed=args.skip_completed,
        label="Eval DPO adapter from SFT adapter",
        workdir=eval_dir,
        command=[
            sys.executable,
            "-u",
            "run_eval.py",
            str(dpo_sft_adapter_dir),
            args.base_family,
            str(args.eval_num_shots),
            f"outputs/{dpo_sft_model_tag}_eval",
            *eval_flags,
        ],
        env=env,
        launcher=launcher,
        is_complete=lambda: all_paths_exist([dpo_sft_eval_dir / "summary.json", dpo_sft_eval_dir / "samples.jsonl"]),
        completion_note=f"found summary.json and samples.jsonl in {dpo_sft_eval_dir.name}",
    )

    log("Pipeline finished successfully")
    log(f"Cached base model: {base_model_path}")
    log(f"Cached critic model: {critic_model_path}")
    log(f"CAI outputs: {cai_dir / 'outputs' / run_tag}")
    log(f"SFT adapter: {sft_adapter_dir}")
    log(f"DPO base adapter: {dpo_base_adapter_dir}")
    log(f"DPO from SFT adapter: {dpo_sft_adapter_dir}")


if __name__ == "__main__":
    main()
