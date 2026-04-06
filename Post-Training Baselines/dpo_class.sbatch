#!/bin/bash
#SBATCH --job-name=w2c_dpo
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p logs

MODEL_NAME_OR_PATH="${1:?arg1=model_name_or_path}"
MODEL_FAMILY="${2:?arg2=model_family}"
OUTPUT_DIR="${3:?arg3=output_dir}"
TRAIN_FILE="${4:-}"

cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

bash run_dpo.sh   "${MODEL_NAME_OR_PATH}"   "${MODEL_FAMILY}"   "${OUTPUT_DIR}"   "${TRAIN_FILE}"
