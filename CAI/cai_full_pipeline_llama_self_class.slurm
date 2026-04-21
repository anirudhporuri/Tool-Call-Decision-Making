#!/bin/bash
#SBATCH --job-name=cai_llama_self
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --time=60:00:00
#SBATCH --mem=32g
#SBATCH --gres=gpu:1

# Llama full pipeline with self-critiques and self-judging.
# Submit with:
#   sbatch cai_full_pipeline_llama_self_class.slurm
# Optional overrides still work, for example:
#   HF_TOKEN=... sbatch cai_full_pipeline_llama_self_class.slurm --no-skip-completed

set -euo pipefail

source ~/.bashrc
REPO_ROOT="/fs/classhomes/mukunds/Tool-Call-Decision-Making"
VENV_DIR="${REPO_ROOT}/.venv"
LLAMA_BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct"

echo "Activating virtualenv at ${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python3 -u "${REPO_ROOT}/CAI/cai_full_pipeline_common.py" \
  --base-model "${LLAMA_BASE_MODEL}" \
  --base-family "llama" \
  --base-tag "llama" \
  --critic-model "${LLAMA_BASE_MODEL}" \
  --critic-family "llama" \
  --critic-tag "self" \
  --base-generation-batch-size 2 \
  --critic-batch-size 2 \
  --base-generation-max-prompt-length 1792 \
  --critic-max-prompt-length 1792 \
  --no-base-generation-load-in-4bit \
  --no-critic-load-in-4bit \
  --training-load-in-4bit \
  "$@"
