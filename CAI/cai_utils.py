from __future__ import annotations

import ast
import copy
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError: 
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None


REPO_ROOT = Path(__file__).resolve().parents[1]
CAI_DIR = Path(__file__).resolve().parent
POST_TRAINING_DIR = REPO_ROOT / "Post-Training Baselines"
if str(POST_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(POST_TRAINING_DIR))

from w2c_train_format import build_training_prompt, split_context_and_target


DEFAULT_SELF_MODELS = {
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma": "google/gemma-3-4b-it",
}
DEFAULT_CROSS_MODELS = {
    "llama": DEFAULT_SELF_MODELS["gemma"],
    "gemma": DEFAULT_SELF_MODELS["llama"],
}

TOOLCALL_RE = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)
ALT_TOOLCALL_RE = re.compile(r"^\[TOOLCALL\]\s*(\{.*\})\s*$", re.DOTALL)
VERDICT_RE = re.compile(r"Verdict:\s*(NO_ISSUES|ISSUES)", re.IGNORECASE)
PRIMARY_ISSUE_RE = re.compile(r"Primary Issue:\s*(.+)", re.IGNORECASE)
CRITIQUE_RE = re.compile(r"Critique:\s*(.+)", re.IGNORECASE | re.DOTALL)
WINNER_RE = re.compile(r"Winner:\s*([AB])", re.IGNORECASE)
REASON_RE = re.compile(r"Reason:\s*(.+)", re.IGNORECASE | re.DOTALL)

ALLOWED_PRIMARY_ISSUES = {
    "should_have_called_tool",
    "missing_required_information",
    "tools_insufficient",
    "invented_tool_or_arguments",
    "direct_answer_when_tool_required",
    "wrong_tool",
    "unsupported_arguments",
    "bad_tool_call_format",
}

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


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path_obj = Path(path)
    temp_path = path_obj.with_name(f"{path_obj.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path_obj)


def load_jsonl(path: str | Path, *, allow_partial_last_line: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    path_obj = Path(path)
    lines = path_obj.read_text(encoding="utf-8").splitlines()
    for line_idx, line in enumerate(lines, start=1):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if allow_partial_last_line and line_idx == len(lines):
                    break
                raise
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path_obj = Path(path)
    temp_path = path_obj.with_name(f"{path_obj.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path_obj)


def append_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        for row in rows_list:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def progress(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, dynamic_ncols=True, **kwargs)
    except Exception:
        return iterable


def load_template(name: str) -> str:
    return (CAI_DIR / name).read_text(encoding="utf-8").strip()


def get_constitution() -> str:
    return load_template("Constitution.txt")


def render_template(template_text: str, **values: str) -> str:
    rendered = template_text
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def format_conversation(messages: List[Dict[str, Any]]) -> str:
    if not messages:
        return ""
    lines = []
    for msg in messages:
        role = msg.get("role", "").capitalize() or "Unknown"
        content = str(msg.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def source_prompt_messages(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = row.get("messages", [])
    if not isinstance(messages, list):
        return []
    copied_messages = [dict(message) for message in messages if isinstance(message, dict)]
    if copied_messages and copied_messages[-1].get("role") == "assistant":
        try:
            context_messages, _ = split_context_and_target(copied_messages)
            return [dict(message) for message in context_messages]
        except Exception:
            return copied_messages
    return copied_messages


def source_user_request_text(row: Dict[str, Any]) -> str:
    return format_conversation(source_prompt_messages(row))


def serialize_tools(tools: Any) -> str:
    if isinstance(tools, str):
        return tools
    try:
        return json.dumps(tools, ensure_ascii=False, indent=2)
    except Exception:
        return str(tools)


def extract_toolcall_payload(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    match = TOOLCALL_RE.search(text)
    if match:
        return match.group(1).strip()
    alt_match = ALT_TOOLCALL_RE.match(text.strip())
    if alt_match:
        return alt_match.group(1).strip()
    return None


def maybe_json_load(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def extract_leading_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped or stripped[0] not in {"{", "["}:
        return None
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(stripped)
        return obj
    except Exception:
        return None


def _ast_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        if isinstance(node, ast.Name):
            lowered = node.id.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            if lowered in {"none", "null"}:
                return None
            return node.id
        if isinstance(node, ast.Attribute):
            return ast.unparse(node)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return ast.unparse(node)
        if isinstance(node, ast.Call):
            return ast.unparse(node)
        if isinstance(node, ast.Subscript):
            return ast.unparse(node)
        try:
            return ast.unparse(node)
        except Exception:
            raise


def parse_llama_single_toolcall(text: str) -> Optional[Dict[str, Any]]:
    raw = text.strip()
    if not raw:
        return None
    try:
        tree = ast.parse(raw, mode="eval")
    except SyntaxError:
        return None

    body = tree.body
    if isinstance(body, ast.List):
        if len(body.elts) != 1:
            return None
        body = body.elts[0]

    if not isinstance(body, ast.Call):
        return None
    if not isinstance(body.func, ast.Name):
        return None
    if body.args:
        return None

    arguments: Dict[str, Any] = {}
    for keyword in body.keywords:
        if keyword.arg is None:
            return None
        arguments[keyword.arg] = _ast_literal(keyword.value)

    return {"name": body.func.id, "arguments": arguments}


def _json_safe_tool_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_tool_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_tool_value(item) for item in value]
    if isinstance(value, set):
        normalized_items = [_json_safe_tool_value(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
    return str(value)


def normalize_tool_call(call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(call, dict):
        return None
    if "name" not in call:
        return None
    args = call.get("arguments", call.get("parameters", {}))
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return {
        "name": str(call["name"]),
        "arguments": _json_safe_tool_value(args),
    }


def parse_single_tool_call(text: str) -> Optional[Dict[str, Any]]:
    try:
        if not isinstance(text, str):
            return None

        stripped = text.strip()
        if not stripped:
            return None

        payload = extract_toolcall_payload(stripped)
        candidate = payload if payload is not None else stripped

        parsed_json = maybe_json_load(candidate)
        if isinstance(parsed_json, dict):
            return normalize_tool_call(parsed_json)
        if isinstance(parsed_json, list) and len(parsed_json) == 1 and isinstance(parsed_json[0], dict):
            return normalize_tool_call(parsed_json[0])

        leading_json = extract_leading_json(candidate)
        if isinstance(leading_json, dict):
            return normalize_tool_call(leading_json)
        if isinstance(leading_json, list) and len(leading_json) == 1 and isinstance(leading_json[0], dict):
            return normalize_tool_call(leading_json[0])

        return parse_llama_single_toolcall(candidate)
    except Exception:
        return None


def wrap_canonical_toolcall(call: Dict[str, Any]) -> str:
    payload = json.dumps(call, ensure_ascii=False, separators=(",", ":"))
    return f"<TOOLCALL>{payload}</TOOLCALL>"


def canonicalize_assistant_response(text: str) -> str:
    if not isinstance(text, str):
        return ""

    stripped = text.strip()
    if not stripped:
        return ""

    try:
        parsed_call = parse_single_tool_call(stripped)
        if parsed_call is not None:
            return wrap_canonical_toolcall(parsed_call)
    except Exception:
        return stripped

    return stripped


def has_tool_call_marker(text: str) -> bool:
    raw = text if isinstance(text, str) else ""
    try:
        return (
            "<TOOLCALL>" in raw
            or "[TOOLCALL]" in raw
            or parse_single_tool_call(raw) is not None
        )
    except Exception:
        return False


def _recognized_type_tokens(type_text: str) -> List[str]:
    text = (type_text or "").lower()
    recognized = []
    for token in ["str", "string", "int", "integer", "float", "number", "bool", "boolean", "dict", "object", "list", "array"]:
        if token in text:
            recognized.append(token)
    return recognized


def _value_matches_type(value: Any, type_text: str) -> bool:
    tokens = _recognized_type_tokens(type_text)
    if not tokens:
        return True
    for token in tokens:
        if token in {"str", "string"} and isinstance(value, str):
            return True
        if token in {"int", "integer"}:
            if isinstance(value, int) and not isinstance(value, bool):
                return True
            if isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
                return True
        if token in {"float", "number"}:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
            if isinstance(value, str):
                try:
                    float(value.strip())
                    return True
                except Exception:
                    pass
        if token in {"bool", "boolean"}:
            if isinstance(value, bool):
                return True
            if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                return True
        if token in {"dict", "object"} and isinstance(value, dict):
            return True
        if token in {"list", "array"} and isinstance(value, list):
            return True
    return False


def parse_tools_spec(tools: Any) -> Dict[str, Dict[str, Any]]:
    try:
        if isinstance(tools, str):
            candidate = maybe_json_load(tools)
            tool_items = candidate if isinstance(candidate, list) else []
        elif isinstance(tools, list):
            tool_items = tools
        else:
            tool_items = []

        parsed_tools: Dict[str, Dict[str, Any]] = {}
        for item in tool_items:
            if isinstance(item, str):
                item = maybe_json_load(item)
            if not isinstance(item, dict) or "name" not in item:
                continue
            parsed_tools[str(item["name"])] = item
        return parsed_tools
    except Exception:
        return {}


def validate_single_tool_call(text: str, tools: Any) -> Dict[str, Any]:
    try:
        call = parse_single_tool_call(text)
        if call is None:
            return {"valid": False, "reason": "not_single_tool_call", "call": None}

        tool_specs = parse_tools_spec(tools)
        tool_name = str(call["name"])
        tool_spec = tool_specs.get(tool_name)
        if tool_spec is None:
            return {"valid": False, "reason": "unknown_tool", "call": call}

        args = call.get("arguments", {})
        if not isinstance(args, dict):
            return {"valid": False, "reason": "arguments_not_object", "call": call}

        params = tool_spec.get("parameters", {}) if isinstance(tool_spec.get("parameters"), dict) else {}
        properties = params.get("properties", {}) if isinstance(params.get("properties"), dict) else {}
        required = tool_spec.get("required", [])
        if not isinstance(required, list):
            required = []

        missing_required = [name for name in required if name not in args]
        if missing_required:
            return {"valid": False, "reason": f"missing_required:{','.join(missing_required)}", "call": call}

        unsupported = [name for name in args if properties and name not in properties]
        if unsupported:
            return {"valid": False, "reason": f"unsupported_arguments:{','.join(unsupported)}", "call": call}

        for arg_name, arg_value in args.items():
            spec = properties.get(arg_name, {})
            spec_type = spec.get("type", "") if isinstance(spec, dict) else ""
            if spec_type and not _value_matches_type(arg_value, spec_type):
                return {"valid": False, "reason": f"bad_argument_type:{arg_name}", "call": call}

        return {"valid": True, "reason": None, "call": call}
    except Exception:
        return {"valid": False, "reason": "tool_validation_error", "call": None}


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


def validate_balanced_counts(rows: List[Dict[str, Any]], label_key: str) -> Dict[str, int]:
    counts = count_label_values(rows, label_key)
    if len(set(counts.values())) > 1:
        raise ValueError(f"Expected balanced counts for {label_key}, found {counts}")
    return counts


def count_label_values(rows: List[Dict[str, Any]], label_key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        label = str(row[label_key])
        counts[label] = counts.get(label, 0) + 1
    return counts


def get_dtype(name: str) -> torch.dtype:
    if torch is None:
        raise ImportError("torch is required for model generation utilities.")
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_quant_config(load_in_4bit: bool, dtype_name: str) -> Optional[BitsAndBytesConfig]:
    if BitsAndBytesConfig is None and load_in_4bit:
        raise ImportError("transformers with bitsandbytes support is required for 4-bit loading.")
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=get_dtype(dtype_name),
        bnb_4bit_use_double_quant=True,
    )


def is_peft_adapter_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "adapter_config.json").is_file()


def load_generation_model(
    *,
    model_name_or_path: str,
    hf_token: Optional[str],
    dtype: str,
    attn_implementation: Optional[str],
    load_in_4bit: bool,
    trust_remote_code: bool,
) -> Tuple[Any, Any]:
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise ImportError("transformers is required for model generation utilities.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        token=hf_token,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model_kwargs = {
        "token": hf_token,
        "dtype": get_dtype(dtype),
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    quant_config = build_quant_config(load_in_4bit, dtype)
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    def load_causal_lm(target_model_name_or_path: str, kwargs: Dict[str, Any]) -> Any:
        try:
            return AutoModelForCausalLM.from_pretrained(
                target_model_name_or_path,
                **kwargs,
            )
        except ValueError as exc:
            message = str(exc)
            if (
                "The model is quantized with" in message
                and "BitsAndBytesConfig" in message
                and "quantization_config" in kwargs
            ):
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("quantization_config", None)
                print(
                    "Model already defines a native quantization config; "
                    "retrying load without BitsAndBytes override."
                )
                return AutoModelForCausalLM.from_pretrained(
                    target_model_name_or_path,
                    **retry_kwargs,
                )
            raise

    if is_peft_adapter_checkpoint(model_name_or_path):
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(model_name_or_path, token=hf_token)
        base_model = load_causal_lm(
            peft_config.base_model_name_or_path,
            model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_name_or_path,
            token=hf_token,
        )
    else:
        model = load_causal_lm(
            model_name_or_path,
            model_kwargs,
        )

    model.eval()
    return tokenizer, model


def unload_model(tokenizer: Any, model: Any) -> None:
    del tokenizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _prepare_inputs(tokenizer: Any, prompt: str, model_family: str, device: torch.device) -> Dict[str, torch.Tensor]:
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    batch = {key: value.to(device) for key, value in batch.items()}
    if model_family == "gemma" and "token_type_ids" not in batch:
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
    return batch


def _prepare_batched_inputs(
    tokenizer: Any,
    prompts: List[str],
    model_family: str,
    device: torch.device,
    max_prompt_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    tokenizer_kwargs: Dict[str, Any] = {
        "return_tensors": "pt",
        "add_special_tokens": False,
        "padding": True,
    }
    if max_prompt_length is not None:
        tokenizer_kwargs["truncation"] = True
        tokenizer_kwargs["max_length"] = max_prompt_length
    batch = tokenizer(prompts, **tokenizer_kwargs)
    batch = {key: value.to(device) for key, value in batch.items()}
    if model_family == "gemma" and "token_type_ids" not in batch:
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
    return batch


def _build_generation_config(
    *,
    model: Any,
    tokenizer: Any,
    do_sample: bool,
    max_new_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
) -> Any:
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = do_sample
    generation_config.max_new_tokens = max_new_tokens
    if generation_config.pad_token_id is None:
        generation_config.pad_token_id = tokenizer.pad_token_id
    if generation_config.eos_token_id is None:
        generation_config.eos_token_id = tokenizer.eos_token_id
    if do_sample:
        generation_config.temperature = temperature if temperature is not None else 1.0
        generation_config.top_p = top_p if top_p is not None else 1.0
    else:
        if hasattr(generation_config, "temperature"):
            generation_config.temperature = None
        if hasattr(generation_config, "top_p"):
            generation_config.top_p = None
        if hasattr(generation_config, "top_k"):
            generation_config.top_k = None
    return generation_config


def generate_responses(
    *,
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    model_family: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    max_prompt_length: Optional[int] = None,
) -> List[str]:
    if not prompts:
        return []

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = _model_device(model)
    inputs = _prepare_batched_inputs(
        tokenizer,
        prompts,
        model_family,
        device,
        max_prompt_length=max_prompt_length,
    )
    input_width = int(inputs["input_ids"].shape[1])
    generation_config = _build_generation_config(
        model=model,
        tokenizer=tokenizer,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    with torch.no_grad():
        generated = model.generate(**inputs, generation_config=generation_config)

    outputs: List[str] = []
    for row_idx in range(len(prompts)):
        new_tokens = generated[row_idx][input_width:]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def generate_response(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    model_family: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
) -> str:
    return generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        model_family=model_family,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )[0]


def build_policy_prompt(model_family: str, row: Dict[str, Any]) -> str:
    return build_training_prompt(
        model_family=model_family,
        tools=row["tools"],
        context_messages=source_prompt_messages(row),
    )


def _wrap_llama_chat(user_text: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_text}\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def _wrap_gemma_chat(user_text: str) -> str:
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"


def _wrap_qwen_chat(user_text: str) -> str:
    return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


def _wrap_with_tokenizer_chat_template(tokenizer: Any, user_text: str, model_family: str) -> Optional[str]:
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return None
    template_kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model_family == "qwen":
        template_kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            **template_kwargs,
        )
    except Exception:
        return None


def wrap_chat_prompt(model_family: str, user_text: str, tokenizer: Any = None) -> str:
    template_wrapped = _wrap_with_tokenizer_chat_template(tokenizer, user_text, model_family)
    if template_wrapped is not None and model_family in {"qwen", "gpt-oss", "phi", "mistral"}:
        return template_wrapped
    if model_family == "llama":
        return _wrap_llama_chat(user_text)
    if model_family == "gemma":
        return _wrap_gemma_chat(user_text)
    if model_family == "qwen":
        return _wrap_qwen_chat(user_text)
    if model_family in {"gpt-oss", "phi", "mistral"}:
        return user_text
    raise ValueError(f"Unsupported model_family={model_family!r}")


def build_critique_prompt(
    *,
    model_family: str,
    constitution: str,
    user_request: str,
    tools: Any,
    assistant_response: str,
    tokenizer: Any = None,
) -> str:
    template = load_template("critique_prompt.txt")
    prompt = render_template(
        template,
        CONSTITUTION=constitution,
        USER_REQUEST=user_request,
        TOOLS=serialize_tools(tools),
        ASSISTANT_RESPONSE=assistant_response.strip(),
    )
    return wrap_chat_prompt(model_family, prompt, tokenizer=tokenizer)


def build_revision_prompt(
    *,
    model_family: str,
    constitution: str,
    user_request: str,
    tools: Any,
    assistant_response: str,
    critique_text: str,
    tokenizer: Any = None,
) -> str:
    template = load_template("revision_prompt.txt")
    prompt = render_template(
        template,
        CONSTITUTION=constitution,
        USER_REQUEST=user_request,
        TOOLS=serialize_tools(tools),
        ASSISTANT_RESPONSE=assistant_response.strip(),
        CRITIQUE=critique_text.strip(),
    )
    return wrap_chat_prompt(model_family, prompt, tokenizer=tokenizer)


def build_preference_prompt(
    *,
    model_family: str,
    constitution: str,
    user_request: str,
    tools: Any,
    response_a: str,
    response_b: str,
    tokenizer: Any = None,
) -> str:
    template = load_template("preference_prompt.txt")
    prompt = render_template(
        template,
        CONSTITUTION=constitution,
        USER_REQUEST=user_request,
        TOOLS=serialize_tools(tools),
        RESPONSE_A=response_a.strip(),
        RESPONSE_B=response_b.strip(),
    )
    return wrap_chat_prompt(model_family, prompt, tokenizer=tokenizer)


def parse_critique_output(text: str) -> Dict[str, Any]:
    raw_text = text.strip() if isinstance(text, str) else str(text).strip()
    try:
        verdict_match = VERDICT_RE.search(raw_text)
        primary_issue_match = PRIMARY_ISSUE_RE.search(raw_text)
        critique_match = CRITIQUE_RE.search(raw_text)
        verdict = verdict_match.group(1).upper() if verdict_match else None
        primary_issue = primary_issue_match.group(1).strip() if primary_issue_match else None
        critique_text = critique_match.group(1).strip() if critique_match else None
        valid = verdict in {"NO_ISSUES", "ISSUES"} and primary_issue is not None and critique_text is not None
        if valid and verdict == "NO_ISSUES":
            valid = primary_issue == "none"
        elif valid and verdict == "ISSUES":
            valid = primary_issue in ALLOWED_PRIMARY_ISSUES
        return {
            "valid": valid,
            "verdict": verdict,
            "primary_issue": primary_issue,
            "critique": critique_text,
            "raw_text": raw_text,
        }
    except Exception:
        return {
            "valid": False,
            "verdict": None,
            "primary_issue": None,
            "critique": None,
            "raw_text": raw_text,
        }


def parse_preference_output(text: str) -> Dict[str, Any]:
    raw_text = text.strip() if isinstance(text, str) else str(text).strip()
    try:
        winner_match = WINNER_RE.search(raw_text)
        reason_match = REASON_RE.search(raw_text)
        winner = winner_match.group(1).upper() if winner_match else None
        reason = reason_match.group(1).strip() if reason_match else None
        valid = winner in {"A", "B"} and bool(reason)
        return {
            "valid": valid,
            "winner": winner,
            "reason": reason,
            "raw_text": raw_text,
        }
    except Exception:
        return {
            "valid": False,
            "winner": None,
            "reason": None,
            "raw_text": raw_text,
        }


def critique_missing_fields(parsed: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    if parsed.get("verdict") not in {"NO_ISSUES", "ISSUES"}:
        missing.append("Verdict")
    if parsed.get("primary_issue") is None:
        missing.append("Primary Issue")
    elif parsed.get("verdict") == "NO_ISSUES" and parsed.get("primary_issue") != "none":
        missing.append("Primary Issue")
    elif parsed.get("verdict") == "ISSUES" and not parsed.get("valid"):
        missing.append("Primary Issue")
    if not parsed.get("critique"):
        missing.append("Critique")
    return missing


def build_fallback_critique_text(raw_attempts: List[str], parsed_attempts: List[Dict[str, Any]]) -> str:
    non_empty_attempts = [text.strip() for text in raw_attempts if text and text.strip()]
    if not non_empty_attempts:
        return (
            "Critique parsing failed.\n"
            "Missing or invalid fields: Verdict, Primary Issue, Critique.\n"
            "Use the constitution to rewrite the response correctly."
        )

    missing_fields = sorted(
        {
            field
            for parsed in parsed_attempts
            if not parsed.get("valid")
            for field in critique_missing_fields(parsed)
        }
    )
    fields_text = ", ".join(missing_fields) if missing_fields else "unknown"
    lines = [
        "Critique parsing failed.",
        f"Missing or invalid fields: {fields_text}.",
        "Use the constitution and any useful critique content below to rewrite the response correctly.",
    ]
    for attempt_idx, text in enumerate(non_empty_attempts, start=1):
        lines.append(f"Attempt {attempt_idx} raw critique:")
        lines.append(text)
    return "\n".join(lines)


def evaluate_candidate_response(response_text: str, tools: Any) -> Dict[str, Any]:
    raw_text = response_text if isinstance(response_text, str) else str(response_text or "")
    fallback_canonical = raw_text.strip()
    try:
        canonical = canonicalize_assistant_response(raw_text)
    except Exception:
        canonical = fallback_canonical
    try:
        response_class = heuristic_class(canonical)
    except Exception:
        response_class = "other_plain_text"
    try:
        tool_call_like = has_tool_call_marker(raw_text) or has_tool_call_marker(canonical)
    except Exception:
        tool_call_like = False
    try:
        tool_validation = validate_single_tool_call(canonical, tools)
    except Exception:
        tool_validation = {"valid": False, "reason": "tool_validation_error", "call": None}

    if not canonical:
        valid = False
        reason = "empty_response"
        score = 0
        structural_kind = "empty"
    elif tool_validation["valid"]:
        valid = True
        reason = None
        score = 2
        structural_kind = "valid_tool_call"
    elif tool_call_like:
        valid = False
        reason = tool_validation["reason"]
        score = 0
        structural_kind = "invalid_tool_call"
    else:
        valid = True
        reason = None
        score = 1
        structural_kind = "plain_text"

    return {
        "canonical": canonical,
        "class": response_class,
        "valid": valid,
        "reason": reason,
        "score": score,
        "structural_kind": structural_kind,
        "tool_validation": tool_validation,
    }


def student_display_name(model_family: str) -> str:
    return "Gemma" if model_family == "gemma" else "Llama"
