#!/bin/bash
#SBATCH --job-name=cai_full_llama
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --time=24:00:00
#SBATCH --mem=32g
#SBATCH --gres=gpu:1

# Submit from the CAI directory or with an absolute path, for example:
#   sbatch cai_full_pipeline_llama_class.slurm
# Override optional settings at submit time, for example:
#   HF_TOKEN=... sbatch cai_full_pipeline_llama_class.slurm
#   sbatch --qos=high --time=48:00:00 cai_full_pipeline_llama_class.slurm

set -euo pipefail

source ~/.bashrc
REPO_ROOT="/fs/classhomes/mukunds/Tool-Call-Decision-Making"
VENV_DIR="${REPO_ROOT}/.venv"

echo "Activating virtualenv at ${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python3 -u "${REPO_ROOT}/CAI/cai_full_pipeline_common.py" \
  --base-model "meta-llama/Llama-3.2-3B-Instruct" \
  --base-family "llama" \
  --base-tag "llama" \
  --critic-model "Qwen/Qwen3.5-9B" \
  --critic-family "qwen" \
  --critic-tag "qwen3p5_9b" \
  "$@"
