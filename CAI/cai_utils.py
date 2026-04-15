from __future__ import annotations

import ast
import copy
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CAI_DIR = Path(__file__).resolve().parent
POST_TRAINING_DIR = REPO_ROOT / "Post-Training Baselines"
if str(POST_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(POST_TRAINING_DIR))

from w2c_train_format import build_training_prompt  # noqa: E402


DEFAULT_SELF_MODELS = {
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma": "google/gemma-3-4b-it",
}
DEFAULT_CROSS_MODELS = {
    "llama": DEFAULT_SELF_MODELS["gemma"],
    "gemma": DEFAULT_SELF_MODELS["llama"],
}

TOOLCALL_RE = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)
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
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    if not match:
        return None
    return match.group(1).strip()


def maybe_json_load(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def _ast_literal(node: ast.AST) -> Any:
    return ast.literal_eval(node)


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


def wrap_canonical_toolcall(call: Dict[str, Any]) -> str:
    payload = json.dumps(call, ensure_ascii=False, separators=(",", ":"))
    return f"<TOOLCALL>{payload}</TOOLCALL>"


def canonicalize_assistant_response(text: str) -> str:
    if not isinstance(text, str):
        return ""

    stripped = text.strip()
    if not stripped:
        return ""

    payload = extract_toolcall_payload(stripped)
    candidate = payload if payload is not None else stripped

    parsed_json = maybe_json_load(candidate)
    if isinstance(parsed_json, dict) and "name" in parsed_json:
        return wrap_canonical_toolcall(parsed_json)
    if (
        isinstance(parsed_json, list)
        and len(parsed_json) == 1
        and isinstance(parsed_json[0], dict)
        and "name" in parsed_json[0]
    ):
        return wrap_canonical_toolcall(parsed_json[0])

    parsed_llama = parse_llama_single_toolcall(candidate)
    if parsed_llama is not None:
        return wrap_canonical_toolcall(parsed_llama)

    return stripped


def has_tool_call_marker(text: str) -> bool:
    raw = text if isinstance(text, str) else ""
    payload = extract_toolcall_payload(raw)
    candidate = payload if payload is not None else raw.strip()
    return (
        "<TOOLCALL>" in raw
        or candidate.startswith("{")
        or candidate.startswith("[")
        or parse_llama_single_toolcall(candidate) is not None
    )


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
    counts: Dict[str, int] = {}
    for row in rows:
        label = str(row[label_key])
        counts[label] = counts.get(label, 0) + 1
    if len(set(counts.values())) > 1:
        raise ValueError(f"Expected balanced counts for {label_key}, found {counts}")
    return counts


def get_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_quant_config(load_in_4bit: bool, dtype_name: str) -> Optional[BitsAndBytesConfig]:
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        token=hf_token,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "token": hf_token,
        "torch_dtype": get_dtype(dtype),
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    quant_config = build_quant_config(load_in_4bit, dtype)
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    if is_peft_adapter_checkpoint(model_name_or_path):
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(model_name_or_path, token=hf_token)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_name_or_path,
            token=hf_token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
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
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = _model_device(model)
    inputs = _prepare_inputs(tokenizer, prompt, model_family, device)
    prompt_len = int(inputs["input_ids"].shape[1])
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = do_sample
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
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "generation_config": generation_config,
    }

    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)
    new_tokens = generated[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_policy_prompt(model_family: str, row: Dict[str, Any]) -> str:
    return build_training_prompt(
        model_family=model_family,
        tools=row["tools"],
        context_messages=row["messages"],
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


def wrap_chat_prompt(model_family: str, user_text: str) -> str:
    if model_family == "llama":
        return _wrap_llama_chat(user_text)
    if model_family == "gemma":
        return _wrap_gemma_chat(user_text)
    raise ValueError(f"Unsupported model_family={model_family!r}")


def build_critique_prompt(
    *,
    model_family: str,
    constitution: str,
    user_request: str,
    tools: Any,
    assistant_response: str,
) -> str:
    template = load_template("critique_prompt.txt")
    prompt = render_template(
        template,
        CONSTITUTION=constitution,
        USER_REQUEST=user_request,
        TOOLS=serialize_tools(tools),
        ASSISTANT_RESPONSE=assistant_response.strip(),
    )
    return wrap_chat_prompt(model_family, prompt)


def build_revision_prompt(
    *,
    model_family: str,
    constitution: str,
    user_request: str,
    tools: Any,
    assistant_response: str,
    critique_text: str,
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
    return wrap_chat_prompt(model_family, prompt)


def build_preference_prompt(
    *,
    model_family: str,
    constitution: str,
    user_request: str,
    tools: Any,
    response_a: str,
    response_b: str,
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
    return wrap_chat_prompt(model_family, prompt)


def parse_critique_output(text: str) -> Dict[str, Any]:
    verdict_match = VERDICT_RE.search(text)
    primary_issue_match = PRIMARY_ISSUE_RE.search(text)
    critique_match = CRITIQUE_RE.search(text)
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
        "raw_text": text.strip(),
    }


def parse_preference_output(text: str) -> Dict[str, Any]:
    winner_match = WINNER_RE.search(text)
    reason_match = REASON_RE.search(text)
    winner = winner_match.group(1).upper() if winner_match else None
    reason = reason_match.group(1).strip() if reason_match else None
    valid = winner in {"A", "B"} and bool(reason)
    return {
        "valid": valid,
        "winner": winner,
        "reason": reason,
        "raw_text": text.strip(),
    }


def student_display_name(model_family: str) -> str:
    return "Gemma" if model_family == "gemma" else "Llama"
