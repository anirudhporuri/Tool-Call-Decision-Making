#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" && -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

# Usage examples:
#   bash run_eval.sh meta-llama/Llama-3.2-3B-Instruct llama 0 outputs/llama3_2_zeroshot
#   bash run_eval.sh meta-llama/Llama-3.2-3B-Instruct llama 4 outputs/llama3_2_4shot fewshot_examples.json
#   bash run_eval.sh google/gemma-3-4b-it gemma 4 outputs/gemma3_4b_4shot fewshot_examples.json

MODEL_NAME_OR_PATH="${1:?arg1=model_name_or_path}"
MODEL_FAMILY="${2:?arg2=model_family (llama|gemma)}"
NUM_SHOTS="${3:?arg3=num_shots (0|1|4)}"
OUTPUT_DIR="${4:?arg4=output_dir}"
FEWSHOT_JSON="${5:-}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE_RUN="${SMOKE_RUN:-0}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/local_datasets}"
DRY_RUN_MAX_EXAMPLES="${DRY_RUN_MAX_EXAMPLES:-8}"
SMOKE_RUN_MAX_EXAMPLES="${SMOKE_RUN_MAX_EXAMPLES:-8}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
HF_TOKEN="${HF_TOKEN:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-}"
SAVE_PROMPT_TEXT="${SAVE_PROMPT_TEXT:-0}"

if [[ "${DRY_RUN}" == "1" && "${SMOKE_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 and SMOKE_RUN=1 are mutually exclusive." >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "1" && -z "${MAX_EXAMPLES}" ]]; then
  MAX_EXAMPLES="${DRY_RUN_MAX_EXAMPLES}"
fi
if [[ "${SMOKE_RUN}" == "1" && -z "${MAX_EXAMPLES}" ]]; then
  MAX_EXAMPLES="${SMOKE_RUN_MAX_EXAMPLES}"
fi

EXTRA_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA_ARGS+=(--dry_run)
fi
if [[ -n "${HF_TOKEN}" ]]; then
  EXTRA_ARGS+=(--hf_token "${HF_TOKEN}")
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  EXTRA_ARGS+=(--trust_remote_code)
fi
if [[ "${SAVE_PROMPT_TEXT}" == "1" ]]; then
  EXTRA_ARGS+=(--save_prompt_text)
fi
if [[ -n "${MAX_EXAMPLES}" ]]; then
  EXTRA_ARGS+=(--max_examples "${MAX_EXAMPLES}")
fi
if [[ -n "${ATTN_IMPL}" ]]; then
  EXTRA_ARGS+=(--attn_implementation "${ATTN_IMPL}")
fi
if [[ "${NUM_SHOTS}" != "0" ]]; then
  EXTRA_ARGS+=(--fewshot_json "${FEWSHOT_JSON:?fewshot_json is required when num_shots > 0}")
fi

PYTHON_ARGS=(
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --model_family "${MODEL_FAMILY}"
  --num_shots "${NUM_SHOTS}"
  --dtype "${DTYPE}"
  --output_dir "${OUTPUT_DIR}"
  --dataset_dir "${DATASET_DIR}"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  PYTHON_ARGS+=("${EXTRA_ARGS[@]}")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/w2c_eval_mcq.py" "${PYTHON_ARGS[@]}"
