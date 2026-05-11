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

set -euo pipefail
mkdir -p logs
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

python3 run_cai_split.py "$@"
