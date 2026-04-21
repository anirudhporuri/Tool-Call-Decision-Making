#!/bin/bash
#SBATCH --job-name=cai_gemma_qwen
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --time=24:00:00
#SBATCH --mem=32g
#SBATCH --gres=gpu:1

# Gemma full pipeline with Qwen 3.5 9B for SFT critiques and DPO judging.
# Submit with:
#   sbatch cai_full_pipeline_gemma_qwen_class.slurm
# Qwen stays on the current Transformers + runtime 4-bit path in this setup.

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
  --base-generation-load-in-4bit \
  --critic-load-in-4bit \
  --training-load-in-4bit \
  "$@"
