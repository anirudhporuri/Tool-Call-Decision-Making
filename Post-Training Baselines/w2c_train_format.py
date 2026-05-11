from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

LLAMA_SYSTEM = (
    "You are an expert in composing functions. You are given a question and a set of possible functions.\n"
    "Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\n"
    "If none of the functions can be used, point it out.\n"
    "If the given question lacks the parameters required by the function, also point it out.\n"
    "You should only return the function call in tool call sections.\n"
    "If you decide to invoke any of the function(s), you MUST put it in the format of\n"
    "[func_name1(param1=value1, param2=value2...), func_name2(...)]\n"
    "You SHOULD NOT include any other text in the response if you call a function.\n"
)

GEMMA_USER_INSTRUCTIONS = (
    "You have access to functions.\n\n"
    "Decide how to respond to the question using the provided functions.\n"
    "- If a provided function directly answers the question and the required arguments are available, return a tool call.\n"
    "- If the correct function exists but a required argument is missing, ask for that missing information.\n"
    "- If no provided function can answer the question, say that you cannot answer with the provided tools.\n\n"
    "If you decide to invoke a function, you MUST put it in the format:\n"
    '{"name": "tool_name", "arguments": {"argument1": "value1", "argument2": "value2"}}\n\n'
    "You SHOULD NOT include any other text in the response if you call a function.\n"
)

_TOOLCALL_RE = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)


def normalize_tools(tools: Any) -> str:
    if isinstance(tools, str):
        return tools
    try:
        return json.dumps(tools, ensure_ascii=False, indent=2)
    except Exception:
        return str(tools)


def extract_toolcall_payload(text: str) -> Optional[str]:
    match = _TOOLCALL_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def maybe_json_load(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def python_literal(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(value)


def canonical_toolcall_to_llama(text: str) -> str:
    payload = extract_toolcall_payload(text) or text.strip()
    obj = maybe_json_load(payload)
    if isinstance(obj, dict):
        calls = [obj]
    elif isinstance(obj, list):
        calls = obj
    else:
        return text.strip()

    rendered_calls = []
    for call in calls:
        if not isinstance(call, dict) or "name" not in call:
            return text.strip()
        args = call.get("arguments", call.get("parameters", {})) or {}
        if not isinstance(args, dict):
            return text.strip()
        rendered_args = ", ".join(f"{k}={python_literal(v)}" for k, v in args.items())
        rendered_calls.append(f"{call['name']}({rendered_args})")
    return "[" + ", ".join(rendered_calls) + "]"


def canonical_toolcall_to_gemma(text: str) -> str:
    payload = extract_toolcall_payload(text) or text.strip()
    obj = maybe_json_load(payload)

    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict) and "name" in obj[0]:
        return json.dumps(obj[0], ensure_ascii=False)
    if isinstance(obj, dict):
        return json.dumps(obj, ensure_ascii=False)
    return payload


TOOLCALL_RENDERERS = (
    ("llama", canonical_toolcall_to_llama),
    ("gemma", canonical_toolcall_to_gemma),
)


def render_assistant_content_for_model(content: str, model_family: str) -> str:
    payload = extract_toolcall_payload(content)
    if payload is None:
        return content.strip()
    for family, renderer in TOOLCALL_RENDERERS:
        if model_family == family:
            return renderer(content)
    raise ValueError(f"Unsupported model_family={model_family!r}")


def split_context_and_target(messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
    if not messages:
        raise ValueError("messages is empty")
    if messages[-1].get("role") != "assistant":
        raise ValueError("Expected last message to be assistant for SFT/preference data")
    return messages[:-1], messages[-1].get("content", "")


def _llama_render_turn(role: str, content: str) -> str:
    return f"<|start_header_id|>{role}<|end_header_id|>\n{content}\n<|eot_id|>\n"


def build_training_prompt(*, model_family: str, tools: Any, context_messages: List[Dict[str, str]]) -> str:
    tools_text = normalize_tools(tools)
    model_family = model_family.lower()

    if model_family == "llama":
        parts: List[str] = ["<|begin_of_text|>", _llama_render_turn("system", LLAMA_SYSTEM.rstrip())]
        if not context_messages:
            raise ValueError("Expected at least one context message")
        first = context_messages[0]
        if first["role"] != "user":
            raise ValueError("Expected first context message to be user")
        first_user = (
            "Here is a list of functions in JSON format that you can invoke.\n"
            + tools_text
            + "\n\n"
            + first["content"].strip()
        )
        parts.append(_llama_render_turn("user", first_user))
        for msg in context_messages[1:]:
            parts.append(_llama_render_turn(msg["role"], msg["content"].strip()))
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
        return "".join(parts)

    if model_family == "gemma":
        if not context_messages:
            raise ValueError("Expected at least one context message")
        first = context_messages[0]
        if first["role"] != "user":
            raise ValueError("Expected first context message to be user")
        parts = [
            "<start_of_turn>user\n",
            GEMMA_USER_INSTRUCTIONS,
            "\nFunctions:\n",
            tools_text,
            "\nQuestion:\n",
            first["content"].strip(),
            "\n",
        ]
        for msg in context_messages[1:]:
            role = msg["role"].capitalize()
            parts.extend([f"{role}:\n", msg["content"].strip(), "\n"])
        parts.append("<end_of_turn>\n<start_of_turn>model\n")
        return "".join(parts)

    raise ValueError(f"Unsupported model_family={model_family!r}")


def format_sft_example(*, model_family: str, row: Dict[str, Any]) -> Tuple[str, str]:
    context_messages, target = split_context_and_target(row["messages"])
    prompt = build_training_prompt(model_family=model_family, tools=row["tools"], context_messages=context_messages)
    target_text = render_assistant_content_for_model(target, model_family)
    return prompt, target_text


def format_pref_example(*, model_family: str, row: Dict[str, Any]) -> Tuple[str, str, str]:
    prompt = build_training_prompt(model_family=model_family, tools=row["tools"], context_messages=row["messages"])
    chosen = render_assistant_content_for_model(row["chosen_response"]["content"], model_family)
    rejected = render_assistant_content_for_model(row["rejected_response"]["content"], model_family)
    return prompt, chosen, rejected
