#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash run_eval.sh meta-llama/Llama-3.2-3B-Instruct llama 0 outputs/llama3_2_zeroshot
#   bash run_eval.sh meta-llama/Llama-3.2-3B-Instruct llama 4 outputs/llama3_2_4shot fewshot_examples.json
#   bash run_eval.sh google/gemma-3-4b-it gemma 4 outputs/gemma3_4b_4shot fewshot_examples.json

MODEL_NAME_OR_PATH="${1:?arg1=model_name_or_path}"
MODEL_FAMILY="${2:?arg2=model_family (llama|gemma)}"
NUM_SHOTS="${3:?arg3=num_shots (0|1|4)}"
OUTPUT_DIR="${4:?arg4=output_dir}"
FEWSHOT_JSON="${5:-}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
HF_TOKEN="${HF_TOKEN:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-}"
SAVE_PROMPT_TEXT="${SAVE_PROMPT_TEXT:-0}"

EXTRA_ARGS=()
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

python3 w2c_eval_mcq.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --model_family "${MODEL_FAMILY}" \
  --num_shots "${NUM_SHOTS}" \
  --dtype "${DTYPE}" \
  --output_dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
