#!/bin/bash
#SBATCH --job-name=cai_split
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g

# Submit with something like:
# sbatch cai_split_class.slurm --allow-hf-fallback

set -euo pipefail
mkdir -p logs

# Activate your environment here.
# Examples:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your-env
# or
# source ~/venvs/when2call/bin/activate

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

python3 run_cai_split.py "$@"
