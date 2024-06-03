#!/bin/bash
#SBATCH --job-name=convert_checkpoint
#SBATCH --error=slurm_logs/convert_checkpoint_%j.err
#SBATCH --output=slurm_logs/convert_checkpoint_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0

cd ./parlam_distributed/checkpoint-645000
python -u zero_to_fp32.py . pytorch_model.bin