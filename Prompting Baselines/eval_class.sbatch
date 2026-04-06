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

# Submit with something like:
# sbatch eval_class.sbatch meta-llama/Llama-3.2-3B-Instruct llama 0 outputs/llama3_2_zeroshot
# sbatch eval_class.sbatch meta-llama/Llama-3.2-3B-Instruct llama 4 outputs/llama3_2_4shot fewshot_examples.json
# sbatch eval_class.sbatch google/gemma-3-4b-it gemma 4 outputs/gemma3_4b_4shot fewshot_examples.json

set -euo pipefail
mkdir -p logs

MODEL_NAME_OR_PATH="${1:?arg1=model_name_or_path}"
MODEL_FAMILY="${2:?arg2=model_family}"
NUM_SHOTS="${3:?arg3=num_shots}"
OUTPUT_DIR="${4:?arg4=output_dir}"
FEWSHOT_JSON="${5:-}"

# Activate your environment here.
# Examples:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your-env
# or
# source ~/venvs/when2call/bin/activate

cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

bash run_eval.sh \
  "${MODEL_NAME_OR_PATH}" \
  "${MODEL_FAMILY}" \
  "${NUM_SHOTS}" \
  "${OUTPUT_DIR}" \
  "${FEWSHOT_JSON}"
