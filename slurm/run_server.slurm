#!/bin/bash
#SBATCH --job-name=patient_search_server
#SBATCH --output=./logs/server_%j.out
#SBATCH --error=./logs/server_%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=cai6734

set -euo pipefail

module load conda
mamba activate /blue/cai6734/omerkahveci/conda_env/general

echo "Server node: $(hostname)"
echo "Server started: $(date)"
python -u -m src.hipergator.server
