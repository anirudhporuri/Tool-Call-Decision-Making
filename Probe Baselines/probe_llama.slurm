#!/bin/bash
#SBATCH --job-name=probe_llama
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --mem=32g
#SBATCH --gres=gpu:1

# Submit from the Probe Baselines directory or with an absolute path, for example:
#   sbatch probe_llama_class.slurm
# Override optional settings at submit time, for example:
#   HF_TOKEN=... sbatch probe_llama_class.slurm
#   sbatch probe_llama_class.slurm --reuse-features
#   sbatch probe_llama_class.slurm --load-in-4bit

set -euo pipefail

source ~/.bashrc
REPO_ROOT="/fs/classhomes/mukunds/Tool-Call-Decision-Making"
VENV_DIR="${REPO_ROOT}/.venv"

echo "Activating virtualenv at ${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

OUTPUT_DIR="${REPO_ROOT}/Probe Baselines/outputs/llama32_3b_probe_base"
EVAL_SAMPLES="${REPO_ROOT}/Prompting Baselines/llama32_3b_zeroshot/samples.jsonl"

for arg in "$@"; do
  if [[ "${arg}" == "--use-4shot-prompt" ]]; then
    OUTPUT_DIR="${REPO_ROOT}/Probe Baselines/outputs/llama32_3b_probe_4shot"
    EVAL_SAMPLES="${REPO_ROOT}/Prompting Baselines/llama32_3b_4shot/samples.jsonl"
    break
  fi
done

python3 -u "${REPO_ROOT}/Probe Baselines/run_probe.py" \
  "meta-llama/Llama-3.2-3B-Instruct" \
  "llama" \
  "${OUTPUT_DIR}" \
  "${EVAL_SAMPLES}" \
  --dataset-dir "${REPO_ROOT}/local_datasets" \
  "$@"
