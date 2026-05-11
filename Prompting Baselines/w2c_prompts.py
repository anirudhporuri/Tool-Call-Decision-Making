from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

ANSWER_ORDER = ["direct", "tool_call", "request_for_info", "cannot_answer"]


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


def serialize_tools(tools: Any) -> str:
    if isinstance(tools, str):
        return tools
    try:
        return json.dumps(tools, ensure_ascii=False, indent=2)
    except TypeError:
        return str(tools)


def extract_toolcall_payload(text: str) -> Optional[str]:
    match = _TOOLCALL_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _maybe_json_load(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except Exception:
        return text


def _python_literal(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(value)


def canonical_toolcall_to_llama(choice_text: str) -> str:
    payload = extract_toolcall_payload(choice_text) or choice_text.strip()
    parsed = _maybe_json_load(payload)

    if isinstance(parsed, dict):
        calls = [parsed]
    elif isinstance(parsed, list):
        calls = parsed
    else:
        return choice_text

    rendered_calls = []
    for call in calls:
        if not isinstance(call, dict) or "name" not in call:
            return choice_text
        arguments = call.get("arguments", call.get("parameters", {})) or {}
        if not isinstance(arguments, dict):
            return choice_text
        rendered_args = ", ".join(f"{key}={_python_literal(value)}" for key, value in arguments.items())
        rendered_calls.append(f"{call['name']}({rendered_args})")
    return "[" + ", ".join(rendered_calls) + "]"


def serialize_fewshot_answer(answer_text: str, model_family: str) -> str:
    payload = extract_toolcall_payload(answer_text)
    if payload is None:
        return answer_text.strip()
    if model_family == "llama":
        return canonical_toolcall_to_llama(answer_text)
    return payload


def build_llama_prompt(question: str, tools: Any, fewshot_examples: List[Dict[str, Any]]) -> str:
    parts: List[str] = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n",
        LLAMA_SYSTEM,
        "<|eot_id|>\n",
    ]

    for example in fewshot_examples:
        parts.extend(
            [
                "<|start_header_id|>user<|end_header_id|>\n",
                "Here is a list of functions in JSON format that you can invoke.\n",
                serialize_tools(example["tools"]),
                "\n\n",
                example["question"].strip(),
                "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
                serialize_fewshot_answer(example["answer"], "llama"),
                "\n<|eot_id|>\n",
            ]
        )

    parts.extend(
        [
            "<|start_header_id|>user<|end_header_id|>\n",
            "Here is a list of functions in JSON format that you can invoke.\n",
            serialize_tools(tools),
            "\n\n",
            question.strip(),
            "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        ]
    )
    return "".join(parts)


def build_gemma_prompt(question: str, tools: Any, fewshot_examples: List[Dict[str, Any]]) -> str:
    parts = ["<start_of_turn>user\n", GEMMA_USER_INSTRUCTIONS]

    for i, example in enumerate(fewshot_examples, start=1):
        parts.extend(
            [
                f"\nExample {i}\n",
                "Functions:\n",
                serialize_tools(example["tools"]),
                "\nQuestion:\n",
                example["question"].strip(),
                "\nAnswer:\n",
                serialize_fewshot_answer(example["answer"], "gemma"),
                "\n",
            ]
        )

    parts.extend(
        [
            "\nFunctions:\n",
            serialize_tools(tools),
            "\nQuestion:\n",
            question.strip(),
            "<end_of_turn>\n<start_of_turn>model\n",
        ]
    )
    return "".join(parts)


def build_prompt(
    *,
    model_family: str,
    question: str,
    tools: Any,
    fewshot_examples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    fewshot_examples = fewshot_examples or []
    model_family = model_family.lower()
    builders = {"llama": build_llama_prompt, "gemma": build_gemma_prompt}
    if model_family in builders:
        return builders[model_family](question, tools, fewshot_examples)
    raise ValueError(f"Unsupported model_family={model_family!r}. Use 'llama' or 'gemma'.")


def convert_test_choice_for_model(choice_label: str, choice_text: str, model_family: str) -> str:
    if choice_label != "tool_call":
        return choice_text.strip()
    model_family = model_family.lower()
    if model_family == "llama":
        return canonical_toolcall_to_llama(choice_text)
    if model_family == "gemma":
        payload = extract_toolcall_payload(choice_text)
        return payload if payload is not None else choice_text
    return choice_text.strip()
