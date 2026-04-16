#!/bin/bash
#SBATCH --job-name=cai_dpo_data
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --mem=32g
#SBATCH --gres=gpu:1

# Submit with something like:
# sbatch cai_dpo_data_class.slurm outputs/gemma_cai_sft_self gemma self outputs/gemma_cai_dpo_self generated_datasets/train_pref_cai_dpo_source.jsonl

set -euo pipefail
mkdir -p logs

POLICY_MODEL_NAME_OR_PATH="${1:?arg1=policy_model_name_or_path}"
STUDENT_FAMILY="${2:?arg2=student_family}"
JUDGE_MODE="${3:?arg3=judge_mode}"
OUTPUT_DIR="${4:?arg4=output_dir}"
SOURCE_FILE="${5:?arg5=source_file}"

# Activate your environment here.
# Examples:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your-env
# or
# source ~/venvs/when2call/bin/activate

cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python3 run_cai_dpo_data.py \
  "${POLICY_MODEL_NAME_OR_PATH}" \
  "${STUDENT_FAMILY}" \
  "${JUDGE_MODE}" \
  "${OUTPUT_DIR}" \
  "${SOURCE_FILE}"
