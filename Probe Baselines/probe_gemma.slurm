#!/bin/bash
#SBATCH --job-name=probe_gemma
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --mem=32g
#SBATCH --gres=gpu:1

# Submit from the Probe Baselines directory or with an absolute path, for example:
#   sbatch probe_class.slurm
# Override optional settings at submit time, for example:
#   HF_TOKEN=... sbatch probe_class.slurm
#   sbatch probe_class.slurm --reuse-features
#   sbatch probe_class.slurm --load-in-4bit

set -euo pipefail

source ~/.bashrc
REPO_ROOT="/fs/classhomes/mukunds/Tool-Call-Decision-Making"
VENV_DIR="${REPO_ROOT}/.venv"

echo "Activating virtualenv at ${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python3 -u "${REPO_ROOT}/Probe Baselines/run_probe.py" \
  "google/gemma-3-4b-it" \
  "gemma" \
  "${REPO_ROOT}/Probe Baselines/outputs/gemma3_4b_probe_base" \
  "${REPO_ROOT}/Prompting Baselines/gemma3_4b_zeroshot/samples.jsonl" \
  --dataset-dir "${REPO_ROOT}/local_datasets" \
  "$@"
