#!/bin/bash
#SBATCH --job-name=run_preprocess
#SBATCH --output=./logs/run_preprocess_%j.out
#SBATCH --error=./logs/run_preprocess_%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --account=cai6734

# Load required modules
module load conda
mamba activate /blue/cai6734/omerkahveci/conda_env/general

echo "[SLURM] Slurm job started at: $(date)"

/usr/bin/time -v python -u -m src.hipergator.preprocess

echo "[SLURM] Slurm job finished at: $(date)"
