#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - surfaced at runtime in the probe env.
    snapshot_download = None


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTING_BASELINES_DIR = REPO_ROOT / "Prompting Baselines"
if str(PROMPTING_BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(PROMPTING_BASELINES_DIR))

from w2c_prompts import build_prompt  # noqa: E402


DEFAULT_DATASET_DIR = REPO_ROOT / "local_datasets"
DEFAULT_MODEL_CACHE_DIR = REPO_ROOT / "cluster_cache" / "model_cache"
DEFAULT_HF_HOME_DIR = REPO_ROOT / "cluster_cache" / "hf_home"
DEFAULT_FEWSHOT_JSON = REPO_ROOT / "Prompting Baselines" / "fewshot_examples.template.json"
DEFAULT_BALANCED_SOURCE_JSONL = (
    REPO_ROOT / "Data_Management" / "generated_datasets" / "when2call_balanced_sft.jsonl"
)
CAI_DIR = REPO_ROOT / "CAI"
CAI_SPLIT_SCRIPT = CAI_DIR / "run_cai_split.py"
DEFAULT_TRAIN_SOURCE_JSONLS = [
    REPO_ROOT / "CAI" / "generated_datasets" / "train_pref_cai_sft_source.jsonl",
    REPO_ROOT / "CAI" / "generated_datasets" / "train_pref_cai_dpo_source.jsonl",
]
PROBE_LABELS = ["tool_call", "request_for_info", "cannot_answer"]
LABEL_TO_ID = {label: index for index, label in enumerate(PROBE_LABELS)}
TOKENIZER_FILE_HINTS = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "spiece.model",
    "vocab.json",
    "merges.txt",
}


@dataclass
class ProbeExample:
    example_id: str
    question: str
    tools: Any
    label: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class LayerSpec:
    tag: str
    transformer_layer: int
    hidden_state_index: int


def log(message: str) -> None:
    print(message, flush=True)


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hidden states, train a linear probe, and compare it to saved When2Call MCQ outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--model_family", choices=["llama", "gemma"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_samples_jsonl", required=True)
    parser.add_argument(
        "--peft_base_model_override",
        default=None,
        help="Optional base-model path/ID to use when model_name_or_path is a PEFT adapter checkpoint.",
    )
    parser.add_argument(
        "--train_source_jsonls",
        nargs="+",
        default=[str(path) for path in DEFAULT_TRAIN_SOURCE_JSONLS],
        help="Balanced CAI source splits used to build the 9k probe train corpus.",
    )
    parser.add_argument("--dataset_name", default="nvidia/When2Call")
    parser.add_argument("--dataset_config", default="test")
    parser.add_argument("--dataset_split", default="mcq")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--model_cache_dir", default=str(DEFAULT_MODEL_CACHE_DIR))
    parser.add_argument("--hf_home_dir", default=str(DEFAULT_HF_HOME_DIR))
    parser.add_argument("--fewshot_json", default=None)
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--use_4shot_prompt", action="store_true", default=env_flag("USE_4SHOT_PROMPT", False))
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_test_examples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--dev_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=4000)
    parser.add_argument(
        "--c_values",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0, 100.0],
        help="Regularization strengths tried on the dev split.",
    )
    parser.add_argument(
        "--reuse_features",
        action="store_true",
        help="Reuse saved feature artifacts in output_dir if they already exist.",
    )
    parser.add_argument(
        "--save_prompt_text",
        action="store_true",
        help="Include rendered prompt text in train/test metadata JSONL files.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer/model loading.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the policy model in 4-bit. Off by default so probe states match the regular eval path more closely.",
    )
    parser.add_argument(
        "--prefetch_models",
        action="store_true",
        default=env_flag("PREFETCH_MODELS", True),
        help="Resolve HF model IDs into the shared local cache before extraction.",
    )
    parser.add_argument(
        "--no-prefetch-models",
        dest="prefetch_models",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.use_4shot_prompt:
        if args.num_shots not in {0, 4}:
            parser.error("--use_4shot_prompt is only compatible with --num_shots 4.")
        args.num_shots = 4
        if not args.fewshot_json:
            args.fewshot_json = str(DEFAULT_FEWSHOT_JSON)
    if args.num_shots > 0 and not args.fewshot_json:
        parser.error("--fewshot_json is required when --num_shots > 0.")
    return args


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_dataset_slug(*parts: str) -> str:
    return "__".join(part.replace("/", "__") for part in parts if part)


def sanitize_model_name(model_name_or_path: str) -> str:
    return (
        model_name_or_path.strip()
        .replace("/", "__")
        .replace(":", "_")
        .replace("@", "_")
        .replace(" ", "_")
    )


def sanitized_args_dict(args: argparse.Namespace) -> Dict[str, Any]:
    payload = vars(args).copy()
    if payload.get("hf_token"):
        payload["hf_token"] = "[REDACTED]"
    return payload


def load_fewshot_examples(path: Optional[str], num_shots: int) -> List[Dict[str, Any]]:
    if num_shots == 0:
        return []
    if not path:
        raise ValueError("--fewshot_json is required when --num_shots > 0")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("fewshot_json must contain a JSON list of examples")
    if len(data) < num_shots:
        raise ValueError(f"fewshot_json contains {len(data)} examples but num_shots={num_shots}")
    required = {"question", "tools", "answer"}
    for index, example in enumerate(data[:num_shots]):
        missing = required - set(example)
        if missing:
            raise ValueError(f"Few-shot example {index} is missing required fields: {sorted(missing)}")
    return data[:num_shots]


def get_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


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


def ensure_probe_inputs(args: argparse.Namespace) -> None:
    eval_samples_path = Path(args.eval_samples_jsonl)
    if not eval_samples_path.exists():
        raise FileNotFoundError(
            "Missing MCQ evaluation samples file for probe comparison: "
            f"{eval_samples_path}. Run the matching prompting/eval job first."
        )

    missing_train_sources = [
        Path(source_path_text)
        for source_path_text in args.train_source_jsonls
        if not Path(source_path_text).exists()
    ]
    if not missing_train_sources:
        return

    default_source_set = {path.resolve() for path in DEFAULT_TRAIN_SOURCE_JSONLS}
    requested_source_set = {path.resolve() for path in missing_train_sources}

    if requested_source_set.issubset(default_source_set):
        if not DEFAULT_BALANCED_SOURCE_JSONL.exists():
            raise FileNotFoundError(
                "Missing CAI source splits and the balanced source dataset needed to regenerate them.\n"
                f"Expected balanced source at: {DEFAULT_BALANCED_SOURCE_JSONL}\n"
                "Build it first with Data_Management/build_balanced_sft_dataset.py."
            )
        if not CAI_SPLIT_SCRIPT.exists():
            raise FileNotFoundError(
                "Missing CAI split generator script needed to bootstrap probe training sources: "
                f"{CAI_SPLIT_SCRIPT}"
            )

        log("Missing CAI probe train sources; regenerating them via run_cai_split.py")
        subprocess.run(
            [
                sys.executable,
                str(CAI_SPLIT_SCRIPT),
                "--source-jsonl",
                str(DEFAULT_BALANCED_SOURCE_JSONL),
            ],
            cwd=str(CAI_DIR),
            check=True,
        )

        still_missing = [path for path in missing_train_sources if not path.exists()]
        if still_missing:
            missing_text = "\n".join(str(path) for path in still_missing)
            raise FileNotFoundError(
                "CAI source split regeneration completed, but these probe train source files are still missing:\n"
                f"{missing_text}"
            )
        return

    missing_text = "\n".join(str(path) for path in missing_train_sources)
    raise FileNotFoundError(
        "Probe training source files are missing:\n"
        f"{missing_text}\n"
        "Either create them first or point --train_source_jsonls at existing files."
    )


def is_peft_adapter_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "adapter_config.json").is_file()


def has_local_tokenizer_files(path: str | Path) -> bool:
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_dir():
        return False
    if (candidate / "tokenizer.json").exists():
        return True
    if any(
        (candidate / filename).exists()
        for filename in ("tokenizer.model", "sentencepiece.bpe.model", "spiece.model")
    ):
        return True
    return (candidate / "vocab.json").exists() and (candidate / "merges.txt").exists()


def resolve_peft_base_model_source(
    model_name_or_path: str,
    hf_token: Optional[str],
    override: Optional[str] = None,
) -> str:
    if override:
        return override

    model_path = Path(model_name_or_path)
    if not is_peft_adapter_checkpoint(model_path):
        return model_name_or_path

    from peft import PeftConfig

    peft_config = PeftConfig.from_pretrained(model_name_or_path, token=hf_token)
    return peft_config.base_model_name_or_path


def resolve_tokenizer_source(
    model_name_or_path: str,
    hf_token: Optional[str],
    peft_base_model_override: Optional[str] = None,
) -> str:
    if peft_base_model_override:
        return peft_base_model_override

    model_path = Path(model_name_or_path)
    if not is_peft_adapter_checkpoint(model_path):
        return model_name_or_path

    if has_local_tokenizer_files(model_path):
        return model_name_or_path

    return resolve_peft_base_model_source(
        model_name_or_path,
        hf_token,
        override=peft_base_model_override,
    )


def build_quant_config(dtype_name: str) -> BitsAndBytesConfig:
    compute_dtype = get_dtype(dtype_name)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_tokenizer_and_model(args: argparse.Namespace):
    tokenizer_source = resolve_tokenizer_source(
        args.model_name_or_path,
        args.hf_token,
        peft_base_model_override=args.peft_base_model_override,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Probe extraction uses the final prompt token as the representation target, so
    # if we must truncate, preserve the end of the prompt (current question + answer slot).
    tokenizer.truncation_side = "left"

    model_kwargs: Dict[str, Any] = {
        "token": args.hf_token,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = build_quant_config(args.dtype)
    else:
        model_kwargs["torch_dtype"] = get_dtype(args.dtype)

    if is_peft_adapter_checkpoint(args.model_name_or_path):
        from peft import PeftModel

        peft_base_model_source = resolve_peft_base_model_source(
            args.model_name_or_path,
            args.hf_token,
            override=args.peft_base_model_override,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_base_model_source,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            args.model_name_or_path,
            token=args.hf_token,
            is_trainable=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_kwargs,
        )

    model.eval()
    return tokenizer, model


def infer_num_hidden_layers_from_config(config: Any) -> int:
    if config is None:
        return 0

    direct_value = getattr(config, "num_hidden_layers", None)
    if isinstance(direct_value, int) and direct_value > 0:
        return direct_value

    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            text_config = get_text_config()
        except TypeError:
            text_config = None
        nested_value = infer_num_hidden_layers_from_config(text_config)
        if nested_value > 0:
            return nested_value

    for attr_name in ("text_config", "language_config", "llm_config", "decoder", "base_model"):
        nested = getattr(config, attr_name, None)
        nested_value = infer_num_hidden_layers_from_config(nested)
        if nested_value > 0:
            return nested_value

    if isinstance(config, dict):
        direct_value = config.get("num_hidden_layers")
        if isinstance(direct_value, int) and direct_value > 0:
            return direct_value
        for key in ("text_config", "language_config", "llm_config", "decoder", "base_model"):
            nested_value = infer_num_hidden_layers_from_config(config.get(key))
            if nested_value > 0:
                return nested_value

    return 0


def resolve_layer_specs(model: AutoModelForCausalLM) -> List[LayerSpec]:
    num_hidden_layers = infer_num_hidden_layers_from_config(getattr(model, "config", None))
    if num_hidden_layers <= 0:
        if hasattr(model, "get_base_model"):
            try:
                base_model = model.get_base_model()
            except Exception:
                base_model = None
            if base_model is not None:
                num_hidden_layers = infer_num_hidden_layers_from_config(getattr(base_model, "config", None))
        if num_hidden_layers <= 0:
            raise ValueError(
                "Could not determine num_hidden_layers from model config for probe extraction. "
                "Expected either config.num_hidden_layers or a nested text_config.num_hidden_layers."
            )

    requested_layers = [
        ("middle", max(1, math.ceil(num_hidden_layers * 0.50))),
        ("layer_75pct", max(1, math.ceil(num_hidden_layers * 0.75))),
        ("last", num_hidden_layers),
    ]

    resolved_specs: List[LayerSpec] = []
    seen_layers = set()
    for tag, transformer_layer in requested_layers:
        transformer_layer = min(num_hidden_layers, max(1, transformer_layer))
        if transformer_layer in seen_layers:
            continue
        resolved_specs.append(
            LayerSpec(
                tag=tag,
                transformer_layer=transformer_layer,
                hidden_state_index=transformer_layer,
            )
        )
        seen_layers.add(transformer_layer)
    return resolved_specs


def same_layer_specs(left: Sequence[LayerSpec], right: Sequence[LayerSpec]) -> bool:
    return [asdict(spec) for spec in left] == [asdict(spec) for spec in right]


def source_prompt_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if messages and messages[-1].get("role") == "assistant":
        return list(messages[:-1])
    return list(messages)


def question_from_messages(messages: Sequence[Dict[str, Any]]) -> str:
    if len(messages) == 1 and messages[0].get("role") == "user":
        return str(messages[0].get("content", "")).strip()
    rendered_turns = []
    for message in messages:
        role = str(message.get("role", "user")).strip().capitalize()
        content = str(message.get("content", "")).strip()
        rendered_turns.append(f"{role}: {content}")
    return "\n".join(rendered_turns).strip()


def balanced_cap_examples(
    examples: Sequence[ProbeExample],
    max_examples: Optional[int],
    *,
    seed: int,
) -> List[ProbeExample]:
    if max_examples is None or max_examples >= len(examples):
        return list(examples)

    grouped: Dict[str, List[ProbeExample]] = {label: [] for label in PROBE_LABELS}
    for example in examples:
        grouped[example.label].append(example)

    rng = random.Random(seed)
    for label_examples in grouped.values():
        rng.shuffle(label_examples)

    base_take = max_examples // len(PROBE_LABELS)
    remainder = max_examples % len(PROBE_LABELS)
    capped: List[ProbeExample] = []

    for label_index, label in enumerate(PROBE_LABELS):
        target_count = base_take + (1 if label_index < remainder else 0)
        capped.extend(grouped[label][:target_count])

    rng.shuffle(capped)
    return capped


def load_probe_train_examples(args: argparse.Namespace) -> List[ProbeExample]:
    examples: List[ProbeExample] = []
    for source_path_text in args.train_source_jsonls:
        source_path = Path(source_path_text)
        with source_path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                row = json.loads(line)
                label = row.get("chosen_behavior_class") or row.get("behavior_class")
                if label not in LABEL_TO_ID:
                    continue
                prompt_messages = source_prompt_messages(row.get("messages", []))
                question = question_from_messages(prompt_messages)
                metadata = {
                    "example_id": f"{source_path.stem}:{line_index:05d}",
                    "source_file": str(source_path),
                    "source_split": row.get("source_split"),
                    "question": question,
                    "label": label,
                    "behavior_class": row.get("behavior_class"),
                    "chosen_behavior_class": row.get("chosen_behavior_class"),
                }
                examples.append(
                    ProbeExample(
                        example_id=metadata["example_id"],
                        question=question,
                        tools=row.get("tools"),
                        label=label,
                        metadata=metadata,
                    )
                )
    return balanced_cap_examples(examples, args.max_train_examples, seed=args.random_seed)


def load_probe_test_examples(args: argparse.Namespace) -> List[ProbeExample]:
    dataset = load_eval_dataset(args)
    examples: List[ProbeExample] = []
    start_index = max(0, args.start_index)
    for dataset_index in range(start_index, len(dataset)):
        if args.max_test_examples is not None and len(examples) >= args.max_test_examples:
            break
        row = dataset[dataset_index]
        gold = row["correct_answer"]
        if gold not in LABEL_TO_ID:
            continue
        metadata = {
            "uuid": row["uuid"],
            "dataset_index": dataset_index,
            "source": row.get("source"),
            "source_id": row.get("source_id"),
            "question": row["question"],
            "gold": gold,
        }
        examples.append(
            ProbeExample(
                example_id=row["uuid"],
                question=row["question"],
                tools=row["tools"],
                label=gold,
                metadata=metadata,
            )
        )
    return examples


def extract_features(
    *,
    examples: Sequence[ProbeExample],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    layer_specs: Sequence[LayerSpec],
    args: argparse.Namespace,
    fewshot_examples: Sequence[Dict[str, Any]],
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    if not examples:
        raise ValueError(f"No {split_name} examples available for probe extraction.")

    model_device = next(model.parameters()).device
    feature_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    metadata_rows: List[Dict[str, Any]] = []
    num_truncated_prompts = 0

    for start in tqdm(range(0, len(examples), args.batch_size), desc=f"Extracting {split_name} features"):
        batch_examples = list(examples[start : start + args.batch_size])
        prompts = [
            build_prompt(
                model_family=args.model_family,
                question=example.question,
                tools=example.tools,
                fewshot_examples=list(fewshot_examples),
            )
            for example in batch_examples
        ]
        untruncated_lengths = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=True,
        )["length"]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        encoded = {key: value.to(model_device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        attention_mask = encoded["attention_mask"]
        last_prompt_token_indices = attention_mask.sum(dim=1) - 1
        layer_feature_slices: List[np.ndarray] = []
        for layer_spec in layer_specs:
            layer_hidden = outputs.hidden_states[layer_spec.hidden_state_index]
            row_indices = torch.arange(layer_hidden.size(0), device=layer_hidden.device)
            layer_feature_slices.append(
                layer_hidden[row_indices, last_prompt_token_indices].float().cpu().numpy().astype(np.float32)
            )
        batch_features = np.stack(layer_feature_slices, axis=1)
        batch_labels = np.asarray([LABEL_TO_ID[example.label] for example in batch_examples], dtype=np.int64)

        feature_batches.append(batch_features)
        label_batches.append(batch_labels)

        prompt_lengths = attention_mask.sum(dim=1).detach().cpu().tolist()
        for example, prompt, prompt_length, original_length in zip(
            batch_examples,
            prompts,
            prompt_lengths,
            untruncated_lengths,
        ):
            was_truncated = int(original_length) > args.max_length
            if was_truncated:
                num_truncated_prompts += 1
            row_metadata = dict(example.metadata)
            row_metadata["prompt_length_tokens"] = int(prompt_length)
            row_metadata["prompt_original_length_tokens"] = int(original_length)
            row_metadata["prompt_was_truncated"] = bool(was_truncated)
            row_metadata["extracted_token_index"] = int(prompt_length) - 1
            row_metadata["extracted_token_kind"] = "last_non_padding_token_of_prompt_only_input"
            row_metadata["prompt_includes_target_or_mcq_answers"] = False
            row_metadata["label_id"] = LABEL_TO_ID[example.label]
            if args.save_prompt_text:
                row_metadata["prompt_text"] = prompt
            metadata_rows.append(row_metadata)

    features = np.concatenate(feature_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    if num_truncated_prompts:
        log(
            f"{split_name}: {num_truncated_prompts}/{len(examples)} prompts exceeded max_length={args.max_length} "
            "and were left-truncated to preserve the current question."
        )
    return features, labels, metadata_rows


def save_feature_artifacts(
    *,
    feature_path: Path,
    metadata_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    metadata_rows: Sequence[Dict[str, Any]],
    layer_specs: Sequence[LayerSpec],
) -> None:
    np.savez_compressed(
        feature_path,
        X=features,
        y=labels,
        label_names=np.asarray(PROBE_LABELS),
        layer_tags=np.asarray([spec.tag for spec in layer_specs]),
        transformer_layers=np.asarray([spec.transformer_layer for spec in layer_specs], dtype=np.int64),
        hidden_state_indices=np.asarray([spec.hidden_state_index for spec in layer_specs], dtype=np.int64),
    )
    write_jsonl(metadata_path, metadata_rows)


def load_feature_artifacts(
    feature_path: Path,
    metadata_path: Path,
) -> tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[LayerSpec]]:
    with np.load(feature_path, allow_pickle=False) as payload:
        features = payload["X"]
        labels = payload["y"]
        layer_tags = payload["layer_tags"].tolist() if "layer_tags" in payload else ["last"]
        transformer_layers = (
            payload["transformer_layers"].tolist()
            if "transformer_layers" in payload
            else [features.shape[1] if features.ndim == 3 else -1]
        )
        hidden_state_indices = (
            payload["hidden_state_indices"].tolist()
            if "hidden_state_indices" in payload
            else [transformer_layers[0] if transformer_layers[0] != -1 else -1]
        )
    metadata_rows = read_jsonl(metadata_path)
    if features.ndim == 2:
        features = features[:, None, :]
    layer_specs = [
        LayerSpec(
            tag=str(tag),
            transformer_layer=int(transformer_layer),
            hidden_state_index=int(hidden_state_index),
        )
        for tag, transformer_layer, hidden_state_index in zip(layer_tags, transformer_layers, hidden_state_indices)
    ]
    return features, labels, metadata_rows, layer_specs


def compute_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any],
    target_names: Sequence[str],
) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "confusion_matrix_labels": list(target_names),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(labels),
            target_names=list(target_names),
            output_dict=True,
            zero_division=0,
        ),
    }


def train_probe_for_single_layer(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Pipeline, Dict[str, Any]]:
    X_train, X_dev, y_train, y_dev = train_test_split(
        train_features,
        train_labels,
        test_size=args.dev_size,
        stratify=train_labels,
        random_state=args.random_seed,
    )

    dev_results: List[Dict[str, Any]] = []
    best_pipeline: Optional[Pipeline] = None
    best_result: Optional[Dict[str, Any]] = None

    for c_value in args.c_values:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c_value,
                        max_iter=args.max_iter,
                        solver="lbfgs",
                        random_state=args.random_seed,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        dev_predictions = pipeline.predict(X_dev)
        result = {
            "C": c_value,
            "dev_accuracy": float(accuracy_score(y_dev, dev_predictions)),
            "dev_macro_f1": float(f1_score(y_dev, dev_predictions, average="macro", zero_division=0)),
        }
        dev_results.append(result)
        if best_result is None or result["dev_macro_f1"] > best_result["dev_macro_f1"]:
            best_pipeline = pipeline
            best_result = result

    assert best_pipeline is not None
    assert best_result is not None

    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=best_result["C"],
                    max_iter=args.max_iter,
                    solver="lbfgs",
                    random_state=args.random_seed,
                ),
            ),
        ]
    )
    final_pipeline.fit(train_features, train_labels)

    training_summary = {
        "label_names": list(PROBE_LABELS),
        "num_train_examples": int(train_features.shape[0]),
        "hidden_size": int(train_features.shape[1]),
        "dev_size": args.dev_size,
        "random_seed": args.random_seed,
        "c_values_tried": list(args.c_values),
        "dev_results": dev_results,
        "best_hparams": best_result,
    }
    return final_pipeline, training_summary


def train_probe_suite(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    layer_specs: Sequence[LayerSpec],
    args: argparse.Namespace,
) -> tuple[Dict[str, Pipeline], Dict[str, Any]]:
    probe_pipelines: Dict[str, Pipeline] = {}
    layer_summaries: Dict[str, Any] = {}

    for layer_index, layer_spec in enumerate(layer_specs):
        pipeline, summary = train_probe_for_single_layer(
            train_features=train_features[:, layer_index, :],
            train_labels=train_labels,
            args=args,
        )
        probe_pipelines[layer_spec.tag] = pipeline
        layer_summaries[layer_spec.tag] = {
            **summary,
            "transformer_layer": layer_spec.transformer_layer,
            "hidden_state_index": layer_spec.hidden_state_index,
        }

    training_summary = {
        "label_names": list(PROBE_LABELS),
        "layer_specs": [asdict(spec) for spec in layer_specs],
        "layers": layer_summaries,
    }
    return probe_pipelines, training_summary


def load_eval_samples(path: Path) -> Dict[str, Dict[str, Any]]:
    rows_by_uuid: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_by_uuid[row["uuid"]] = row
    return rows_by_uuid


def compare_probe_to_eval(
    *,
    test_metadata: Sequence[Dict[str, Any]],
    probe_predictions_by_layer: Dict[str, np.ndarray],
    probe_probabilities_by_layer: Dict[str, np.ndarray],
    eval_samples_path: Path,
    layer_specs: Sequence[LayerSpec],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    eval_rows = load_eval_samples(eval_samples_path)
    combined_rows: List[Dict[str, Any]] = []
    missing_eval_rows = 0
    gold_mismatches = 0

    ordered_layer_tags = [spec.tag for spec in layer_specs]
    per_layer_predictions = {
        layer_tag: probe_predictions_by_layer[layer_tag].tolist() for layer_tag in ordered_layer_tags
    }
    per_layer_probabilities = {
        layer_tag: probe_probabilities_by_layer[layer_tag].tolist() for layer_tag in ordered_layer_tags
    }

    for row_index, metadata in enumerate(test_metadata):
        uuid = metadata["uuid"]
        eval_row = eval_rows.get(uuid)
        if eval_row is None:
            missing_eval_rows += 1
            continue
        if eval_row["gold"] != metadata["gold"]:
            gold_mismatches += 1
        combined_rows.append(
            {
                "uuid": uuid,
                "gold": metadata["gold"],
                "question": metadata["question"],
                "probe_preds": {
                    layer_tag: PROBE_LABELS[int(per_layer_predictions[layer_tag][row_index])]
                    for layer_tag in ordered_layer_tags
                },
                "probe_probs": {
                    layer_tag: {
                        label: float(probability)
                        for label, probability in zip(PROBE_LABELS, per_layer_probabilities[layer_tag][row_index])
                    }
                    for layer_tag in ordered_layer_tags
                },
                "model_pred_norm": eval_row["pred_norm"],
                "model_pred_raw": eval_row["pred_raw"],
                "source": metadata.get("source"),
                "source_id": metadata.get("source_id"),
            }
        )

    if not combined_rows:
        raise ValueError(
            f"No overlapping UUIDs were found between extracted test features and {eval_samples_path}."
        )

    model_pred_norm = [row["model_pred_norm"] for row in combined_rows]
    model_pred_raw = [row["model_pred_raw"] for row in combined_rows]

    summary = {
        "num_joined_examples": len(combined_rows),
        "num_missing_eval_rows": missing_eval_rows,
        "num_gold_mismatches_between_test_dataset_and_eval_samples": gold_mismatches,
        "model_pred_norm_vs_gold": compute_metrics(
            [row["gold"] for row in combined_rows],
            model_pred_norm,
            labels=PROBE_LABELS,
            target_names=PROBE_LABELS,
        ),
        "model_pred_raw_vs_gold": compute_metrics(
            [row["gold"] for row in combined_rows],
            model_pred_raw,
            labels=PROBE_LABELS,
            target_names=PROBE_LABELS,
        ),
        "layer_specs": [asdict(spec) for spec in layer_specs],
        "layers": {},
    }

    gold = [row["gold"] for row in combined_rows]
    for layer_spec in layer_specs:
        layer_tag = layer_spec.tag
        probe_pred = [row["probe_preds"][layer_tag] for row in combined_rows]
        norm_in_support = [row for row in combined_rows if row["model_pred_norm"] in LABEL_TO_ID]
        raw_in_support = [row for row in combined_rows if row["model_pred_raw"] in LABEL_TO_ID]

        layer_summary = {
            "transformer_layer": layer_spec.transformer_layer,
            "hidden_state_index": layer_spec.hidden_state_index,
            "probe_vs_gold": compute_metrics(
                gold,
                probe_pred,
                labels=PROBE_LABELS,
                target_names=PROBE_LABELS,
            ),
            "probe_vs_model_pred_norm": {
                "overall_agreement": float(
                    np.mean([row["probe_preds"][layer_tag] == row["model_pred_norm"] for row in combined_rows])
                ),
                "num_model_predictions_in_probe_label_space": len(norm_in_support),
                "num_model_predictions_outside_probe_label_space": len(combined_rows) - len(norm_in_support),
            },
            "probe_vs_model_pred_raw": {
                "overall_agreement": float(
                    np.mean([row["probe_preds"][layer_tag] == row["model_pred_raw"] for row in combined_rows])
                ),
                "num_model_predictions_in_probe_label_space": len(raw_in_support),
                "num_model_predictions_outside_probe_label_space": len(combined_rows) - len(raw_in_support),
            },
        }

        if norm_in_support:
            layer_summary["probe_vs_model_pred_norm"]["restricted_to_probe_label_space"] = compute_metrics(
                [row["model_pred_norm"] for row in norm_in_support],
                [row["probe_preds"][layer_tag] for row in norm_in_support],
                labels=PROBE_LABELS,
                target_names=PROBE_LABELS,
            )
        if raw_in_support:
            layer_summary["probe_vs_model_pred_raw"]["restricted_to_probe_label_space"] = compute_metrics(
                [row["model_pred_raw"] for row in raw_in_support],
                [row["probe_preds"][layer_tag] for row in raw_in_support],
                labels=PROBE_LABELS,
                target_names=PROBE_LABELS,
            )

        summary["layers"][layer_tag] = layer_summary

    return combined_rows, summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = ensure_dir(args.output_dir)
    model_cache_dir = ensure_dir(args.model_cache_dir)
    hf_home_dir = ensure_dir(args.hf_home_dir)

    os.environ["HF_HOME"] = str(hf_home_dir)
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home_dir / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home_dir / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home_dir / "transformers"))
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    train_feature_path = output_dir / "train_features.npz"
    train_metadata_path = output_dir / "train_metadata.jsonl"
    test_feature_path = output_dir / "test_features.npz"
    test_metadata_path = output_dir / "test_metadata.jsonl"
    probe_model_path = output_dir / "probe_model.pkl"
    training_summary_path = output_dir / "probe_training_summary.json"
    evaluation_summary_path = output_dir / "probe_evaluation_summary.json"
    comparison_jsonl_path = output_dir / "probe_comparison_samples.jsonl"
    run_config_path = output_dir / "run_config.json"

    save_json(run_config_path, sanitized_args_dict(args))

    ensure_probe_inputs(args)

    log("Loading prompt few-shot exemplars")
    fewshot_examples = load_fewshot_examples(args.fewshot_json, args.num_shots)

    tokenizer = None
    model = None
    layer_specs: List[LayerSpec] = []

    need_train_extraction = not (args.reuse_features and train_feature_path.exists() and train_metadata_path.exists())
    need_test_extraction = not (args.reuse_features and test_feature_path.exists() and test_metadata_path.exists())

    if need_train_extraction or need_test_extraction:
        if args.prefetch_models:
            args.model_name_or_path = local_or_cached_model_path(
                args.model_name_or_path,
                model_cache_dir,
                args.hf_token,
            )
            if args.peft_base_model_override:
                args.peft_base_model_override = local_or_cached_model_path(
                    args.peft_base_model_override,
                    model_cache_dir,
                    args.hf_token,
                )
        log("Loading tokenizer and model for hidden-state extraction")
        tokenizer, model = load_tokenizer_and_model(args)
        layer_specs = resolve_layer_specs(model)
        log(
            "Selected probe layers: "
            + ", ".join(f"{spec.tag}=L{spec.transformer_layer}" for spec in layer_specs)
        )

    if need_train_extraction:
        log("Preparing combined CAI source rows for probe training")
        train_examples = load_probe_train_examples(args)
        log(f"Train examples: {len(train_examples)}")
        train_features, train_labels, train_metadata = extract_features(
            examples=train_examples,
            tokenizer=tokenizer,
            model=model,
            layer_specs=layer_specs,
            args=args,
            fewshot_examples=fewshot_examples,
            split_name="train",
        )
        save_feature_artifacts(
            feature_path=train_feature_path,
            metadata_path=train_metadata_path,
            features=train_features,
            labels=train_labels,
            metadata_rows=train_metadata,
            layer_specs=layer_specs,
        )
        log(f"Saved train features to {train_feature_path}")
    else:
        log(f"Reusing train features from {train_feature_path}")
        train_features, train_labels, train_metadata, layer_specs = load_feature_artifacts(
            train_feature_path,
            train_metadata_path,
        )
        log(
            "Loaded cached probe layers: "
            + ", ".join(f"{spec.tag}=L{spec.transformer_layer}" for spec in layer_specs)
        )

    if need_test_extraction:
        log("Preparing When2Call test MCQ rows for probe evaluation")
        test_examples = load_probe_test_examples(args)
        log(f"Test examples (filtered to probe label space): {len(test_examples)}")
        test_features, test_labels, test_metadata = extract_features(
            examples=test_examples,
            tokenizer=tokenizer,
            model=model,
            layer_specs=layer_specs,
            args=args,
            fewshot_examples=fewshot_examples,
            split_name="test",
        )
        save_feature_artifacts(
            feature_path=test_feature_path,
            metadata_path=test_metadata_path,
            features=test_features,
            labels=test_labels,
            metadata_rows=test_metadata,
            layer_specs=layer_specs,
        )
        log(f"Saved test features to {test_feature_path}")
    else:
        log(f"Reusing test features from {test_feature_path}")
        test_features, test_labels, test_metadata, cached_test_layer_specs = load_feature_artifacts(
            test_feature_path,
            test_metadata_path,
        )
        if not layer_specs:
            layer_specs = cached_test_layer_specs
        elif not same_layer_specs(layer_specs, cached_test_layer_specs):
            raise ValueError(
                "Cached test features were extracted from different probe layers than the current train features. "
                "Delete the cached feature files or rerun without --reuse_features."
            )

    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log("Training multinomial logistic regression probes")
    probe_pipelines, training_summary = train_probe_suite(
        train_features=train_features,
        train_labels=train_labels,
        layer_specs=layer_specs,
        args=args,
    )
    with open(probe_model_path, "wb") as handle:
        pickle.dump(
            {
                "pipelines": probe_pipelines,
                "label_names": PROBE_LABELS,
                "layer_specs": [asdict(spec) for spec in layer_specs],
            },
            handle,
        )
    save_json(training_summary_path, training_summary)
    log(f"Saved probe model to {probe_model_path}")

    log("Running probes on test hidden states")
    probe_predictions_by_layer = {
        layer_spec.tag: probe_pipelines[layer_spec.tag].predict(test_features[:, layer_index, :])
        for layer_index, layer_spec in enumerate(layer_specs)
    }
    probe_probabilities_by_layer = {
        layer_spec.tag: probe_pipelines[layer_spec.tag].predict_proba(test_features[:, layer_index, :])
        for layer_index, layer_spec in enumerate(layer_specs)
    }

    combined_rows, evaluation_summary = compare_probe_to_eval(
        test_metadata=test_metadata,
        probe_predictions_by_layer=probe_predictions_by_layer,
        probe_probabilities_by_layer=probe_probabilities_by_layer,
        eval_samples_path=Path(args.eval_samples_jsonl),
        layer_specs=layer_specs,
    )
    save_json(evaluation_summary_path, evaluation_summary)
    write_jsonl(comparison_jsonl_path, combined_rows)

    log("Probe pipeline complete")
    log(f"Training summary: {training_summary_path}")
    log(f"Evaluation summary: {evaluation_summary_path}")
    log(f"Joined comparison rows: {comparison_jsonl_path}")


if __name__ == "__main__":
    main()
