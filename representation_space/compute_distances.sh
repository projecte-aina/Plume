#!/bin/bash
#SBATCH --job-name=compute_dist
#SBATCH --error=slurm_logs/compute_dist_%j.err
#SBATCH --output=slurm_logs/compute_dist_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

model_dir=""
name_model=""
save_dir="./results/${name_model}"

python -u compute_distances.py --model_name_or_path $model_dir --output_dir $save_dir