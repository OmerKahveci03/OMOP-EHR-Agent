#!/bin/bash
#SBATCH --job-name=run_analyze_dataset
#SBATCH --output=./logs/run_analyze_dataset_%j.out
#SBATCH --error=./logs/run_analyze_dataset_%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=cai6734

set -euo pipefail

module load conda
mamba activate /blue/cai6734/omerkahveci/conda_env/general

echo "[SLURM] Dataset analysis started at: $(date)"
/usr/bin/time -v python -u -m src.hipergator.data_analysis.analyze_dataset
echo "[SLURM] Dataset analysis finished at: $(date)"
