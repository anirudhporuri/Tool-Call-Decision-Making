#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH="${1:?arg1=model_name_or_path}"
MODEL_FAMILY="${2:?arg2=model_family (llama|gemma)}"
OUTPUT_DIR="${3:?arg3=output_dir}"
TRAIN_FILE="${4:-}"

HF_TOKEN="${HF_TOKEN:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-6}"
BETA="${BETA:-0.1}"
TRAIN_BS="${TRAIN_BS:-2}"
EVAL_BS="${EVAL_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
VAL_SIZE="${VAL_SIZE:-0.02}"
REPORT_TO="${REPORT_TO:-none}"

EXTRA_ARGS=()
[[ -n "${HF_TOKEN}" ]] && EXTRA_ARGS+=(--hf_token "${HF_TOKEN}")
[[ "${TRUST_REMOTE_CODE}" == "1" ]] && EXTRA_ARGS+=(--trust_remote_code)
[[ "${LOAD_IN_4BIT}" == "1" ]] && EXTRA_ARGS+=(--load_in_4bit)
[[ "${GRADIENT_CHECKPOINTING}" == "1" ]] && EXTRA_ARGS+=(--gradient_checkpointing)
[[ -n "${ATTN_IMPL}" ]] && EXTRA_ARGS+=(--attn_implementation "${ATTN_IMPL}")
[[ -n "${MAX_TRAIN_SAMPLES}" ]] && EXTRA_ARGS+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
[[ -n "${MAX_EVAL_SAMPLES}" ]] && EXTRA_ARGS+=(--max_eval_samples "${MAX_EVAL_SAMPLES}")
[[ -n "${TRAIN_FILE}" ]] && EXTRA_ARGS+=(--train_file "${TRAIN_FILE}")

python3 train_dpo_lora.py   --model_name_or_path "${MODEL_NAME_OR_PATH}"   --model_family "${MODEL_FAMILY}"   --output_dir "${OUTPUT_DIR}"   --dtype "${DTYPE}"   --num_train_epochs "${EPOCHS}"   --learning_rate "${LR}"   --beta "${BETA}"   --per_device_train_batch_size "${TRAIN_BS}"   --per_device_eval_batch_size "${EVAL_BS}"   --gradient_accumulation_steps "${GRAD_ACCUM}"   --max_length "${MAX_LENGTH}"   --max_prompt_length "${MAX_PROMPT_LENGTH}"   --val_size "${VAL_SIZE}"   --report_to "${REPORT_TO}"   "${EXTRA_ARGS[@]}"
