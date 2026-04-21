#!/bin/bash
#SBATCH --job-name=cai_full_gemma
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --time=42:00:00
#SBATCH --mem=32g
#SBATCH --gres=gpu:1

# Generic Gemma full-pipeline entrypoint.
# Preferred explicit wrappers:
#   sbatch cai_full_pipeline_gemma_self_class.slurm
#   sbatch cai_full_pipeline_gemma_qwen_class.slurm
#
# This generic wrapper currently matches the qwen-judge mode.

set -euo pipefail

source ~/.bashrc
REPO_ROOT="/fs/classhomes/mukunds/Tool-Call-Decision-Making"
VENV_DIR="${REPO_ROOT}/.venv"
GEMMA_BASE_MODEL="/fs/class-projects/spring2026/cmsc848q/mukunds/google__gemma-3-4b-it"

echo "Activating virtualenv at ${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python3 -u "${REPO_ROOT}/CAI/cai_full_pipeline_common.py" \
  --base-model "${GEMMA_BASE_MODEL}" \
  --base-family "gemma" \
  --base-tag "gemma" \
  --critic-model "Qwen/Qwen3.5-9B" \
  --critic-family "qwen" \
  --critic-tag "qwen3p5_9b" \
  --base-generation-batch-size 2 \
  --critic-batch-size 2 \
  --base-generation-max-prompt-length 1792 \
  --critic-max-prompt-length 1792 \
  --base-generation-load-in-4bit \
  --critic-load-in-4bit \
  --training-load-in-4bit \
  "$@"
