#!/bin/bash
#SBATCH --job-name=cai_sft_data
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gres=gpu:1

# Submit with something like:
# sbatch cai_sft_data_class.slurm google/gemma-3-4b-it gemma outputs/gemma_cai_sft_data generated_datasets/train_pref_cai_sft_source.jsonl

set -euo pipefail
mkdir -p logs

STUDENT_MODEL_NAME_OR_PATH="${1:?arg1=student_model_name_or_path}"
STUDENT_FAMILY="${2:?arg2=student_family}"
OUTPUT_DIR="${3:?arg3=output_dir}"
SOURCE_FILE="${4:?arg4=source_file}"

# Activate your environment here.
# Examples:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your-env
# or
# source ~/venvs/when2call/bin/activate

cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python3 run_cai_sft_data.py \
  "${STUDENT_MODEL_NAME_OR_PATH}" \
  "${STUDENT_FAMILY}" \
  "${OUTPUT_DIR}" \
  "${SOURCE_FILE}"
