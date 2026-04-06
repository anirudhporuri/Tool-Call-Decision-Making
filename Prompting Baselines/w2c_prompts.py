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


def _normalize_tools(tools: Any) -> str:
    """Turn tools into a readable JSON-ish blob for the prompt."""
    if isinstance(tools, str):
        return tools
    try:
        return json.dumps(tools, ensure_ascii=False, indent=2)
    except TypeError:
        # Some rows may already contain serialized tool strings.
        return str(tools)


_TOOLCALL_RE = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)


def extract_toolcall_payload(text: str) -> Optional[str]:
    match = _TOOLCALL_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


_STRING_RE = re.compile(r'^"(.*)"$', re.DOTALL)


def _maybe_json_load(s: str) -> Any:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        return s


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
    """
    Convert a dataset-style tool call payload into Llama 3.2's pythonic list-of-calls surface form.

    Accepted inputs include either:
      - raw JSON: {"name": ..., "arguments": {...}}
      - wrapped JSON: <TOOLCALL>[{"name": ..., "arguments": {...}}]</TOOLCALL>
      - list form already
    """
    payload = extract_toolcall_payload(choice_text) or choice_text.strip()
    obj = _maybe_json_load(payload)

    # The test split usually uses a single JSON object; train tool-call strings can be JSON lists.
    if isinstance(obj, dict):
        calls = [obj]
    elif isinstance(obj, list):
        calls = obj
    else:
        return choice_text

    rendered_calls = []
    for call in calls:
        if not isinstance(call, dict) or "name" not in call:
            return choice_text
        name = call["name"]
        arguments = call.get("arguments", call.get("parameters", {})) or {}
        if not isinstance(arguments, dict):
            return choice_text
        args_str = ", ".join(f"{k}={_python_literal(v)}" for k, v in arguments.items())
        rendered_calls.append(f"{name}({args_str})")
    return "[" + ", ".join(rendered_calls) + "]"



def _serialize_fewshot_answer(answer_text: str, model_family: str) -> str:
    payload = extract_toolcall_payload(answer_text)
    if payload is None:
        return answer_text.strip()
    if model_family == "llama":
        return canonical_toolcall_to_llama(answer_text)
    # Gemma baseline uses JSON tool calls.
    return payload



def build_prompt(
    *,
    model_family: str,
    question: str,
    tools: Any,
    fewshot_examples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    model_family = model_family.lower()
    tools_text = _normalize_tools(tools)
    fewshot_examples = fewshot_examples or []

    if model_family == "llama":
        parts: List[str] = []
        parts.append("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n")
        parts.append(LLAMA_SYSTEM)
        parts.append("<|eot_id|>\n")
        for ex in fewshot_examples:
            ex_tools = _normalize_tools(ex["tools"])
            ex_answer = _serialize_fewshot_answer(ex["answer"], model_family)
            parts.append("<|start_header_id|>user<|end_header_id|>\n")
            parts.append("Here is a list of functions in JSON format that you can invoke.\n")
            parts.append(ex_tools)
            parts.append("\n\n")
            parts.append(ex["question"].strip())
            parts.append("\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")
            parts.append(ex_answer)
            parts.append("\n<|eot_id|>\n")
        parts.append("<|start_header_id|>user<|end_header_id|>\n")
        parts.append("Here is a list of functions in JSON format that you can invoke.\n")
        parts.append(tools_text)
        parts.append("\n\n")
        parts.append(question.strip())
        parts.append("\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")
        return "".join(parts)

    if model_family == "gemma":
        parts = ["<start_of_turn>user\n", GEMMA_USER_INSTRUCTIONS]
        for i, ex in enumerate(fewshot_examples, start=1):
            ex_tools = _normalize_tools(ex["tools"])
            ex_answer = _serialize_fewshot_answer(ex["answer"], model_family)
            parts.append(f"\nExample {i}\n")
            parts.append("Functions:\n")
            parts.append(ex_tools)
            parts.append("\nQuestion:\n")
            parts.append(ex["question"].strip())
            parts.append("\nAnswer:\n")
            parts.append(ex_answer)
            parts.append("\n")
        parts.append("\nFunctions:\n")
        parts.append(tools_text)
        parts.append("\nQuestion:\n")
        parts.append(question.strip())
        parts.append("<end_of_turn>\n<start_of_turn>model\n")
        return "".join(parts)

    raise ValueError(f"Unsupported model_family={model_family!r}. Use 'llama' or 'gemma'.")



def convert_test_choice_for_model(choice_label: str, choice_text: str, model_family: str) -> str:
    """Map the dataset's canonical answer strings into the model-specific surface form used for scoring."""
    if choice_label == "tool_call" and model_family.lower() == "llama":
        return canonical_toolcall_to_llama(choice_text)
    if choice_label == "tool_call" and model_family.lower() == "gemma":
        # Keep JSON-style tool-call output for Gemma.
        payload = extract_toolcall_payload(choice_text)
        return payload if payload is not None else choice_text
    return choice_text.strip()
