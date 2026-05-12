#!/bin/bash
#SBATCH --job-name=w2c_eval
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p logs

MODEL_NAME_OR_PATH="${1:?arg1=model_name_or_path}"
MODEL_FAMILY="${2:?arg2=model_family}"
NUM_SHOTS="${3:?arg3=num_shots}"
OUTPUT_DIR="${4:?arg4=output_dir}"
FEWSHOT_JSON="${5:-}"

cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python run_eval.py \
  "${MODEL_NAME_OR_PATH}" \
  "${MODEL_FAMILY}" \
  "${NUM_SHOTS}" \
  "${OUTPUT_DIR}" \
  "${FEWSHOT_JSON}"
