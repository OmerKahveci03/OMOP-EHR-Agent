#!/bin/bash
#SBATCH --job-name=patient_search
#SBATCH --output=./logs/patient_search_%j.out
#SBATCH --error=./logs/patient_search_%j.err
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=cai6734

set -euo pipefail
module load conda
mamba activate /blue/cai6734/omerkahveci/conda_env/general

python -u -m src.hipergator.patient_search "$@"
