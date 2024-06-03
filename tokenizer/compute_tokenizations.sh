#!/bin/bash
#SBATCH --job-name=compute_stats
#SBATCH --error=slurm_logs/stats_%j.err
#SBATCH --output=slurm_logs/stats_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0

python -u compute_tokenizations.py