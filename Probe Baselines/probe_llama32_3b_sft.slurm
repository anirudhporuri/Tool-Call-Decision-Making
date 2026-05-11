#!/bin/bash
#SBATCH --job-name=probe_llama32_3b_sft
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --mem=32g
#SBATCH --gres=gpu:1

set -euo pipefail

source ~/.bashrc
REPO_ROOT="/fs/classhomes/mukunds/Tool-Call-Decision-Making"
VENV_DIR="${REPO_ROOT}/.venv"
LLAMA_BASE_MODEL_DIR="${REPO_ROOT}/cluster_cache/model_cache/meta-llama__Llama-3.2-3B-Instruct"

echo "Activating virtualenv at ${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

OUTPUT_DIR="${REPO_ROOT}/Probe Baselines/outputs/llama32_3b_sft_probe"
EVAL_SAMPLES="${REPO_ROOT}/Prompting Baselines/outputs/llama32_3b_sft_zeroshot_eval/samples.jsonl"

python3 -u "${REPO_ROOT}/Probe Baselines/run_probe.py" \
  "${REPO_ROOT}/Post-Training Baselines/outputs/llama32_3b_sft" \
  "llama" \
  "${OUTPUT_DIR}" \
  "${EVAL_SAMPLES}" \
  --peft-base-model-override "${LLAMA_BASE_MODEL_DIR}" \
  --dataset-dir "${REPO_ROOT}/local_datasets" \
  "$@"
