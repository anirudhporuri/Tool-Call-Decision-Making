#!/bin/bash
#SBATCH --job-name=probe_gemma
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtxa5000:1

set -euo pipefail

source ~/.bashrc
REPO_ROOT="/fs/classhomes/mukunds/Tool-Call-Decision-Making"
VENV_DIR="${REPO_ROOT}/.venv"
GEMMA_MODEL_PATH="/fs/class-projects/spring2026/cmsc848q/mukunds/google__gemma-3-4b-it"
GEMMA_MODEL_ID="google/gemma-3-4b-it"

echo "Activating virtualenv at ${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

OUTPUT_DIR="${REPO_ROOT}/Probe Baselines/outputs/gemma3_4b_probe_base"
EVAL_SAMPLES="${REPO_ROOT}/Prompting Baselines/gemma3_4b_zeroshot/samples.jsonl"

for arg in "$@"; do
  if [[ "${arg}" == "--use-4shot-prompt" ]]; then
    OUTPUT_DIR="${REPO_ROOT}/Probe Baselines/outputs/gemma3_4b_probe_4shot"
    EVAL_SAMPLES="${REPO_ROOT}/Prompting Baselines/gemma3_4b_4shot/samples.jsonl"
    break
  fi
done

MODEL_PATH_OR_ID="${GEMMA_MODEL_ID}"
if [[ -d "${GEMMA_MODEL_PATH}" ]]; then
  MODEL_PATH_OR_ID="${GEMMA_MODEL_PATH}"
fi

python3 -u "${REPO_ROOT}/Probe Baselines/run_probe.py" \
  "${MODEL_PATH_OR_ID}" \
  "gemma" \
  "${OUTPUT_DIR}" \
  "${EVAL_SAMPLES}" \
  --dataset-dir "${REPO_ROOT}/local_datasets" \
  "$@"
