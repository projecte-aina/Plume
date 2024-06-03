#!/bin/bash
#SBATCH --job-name=extract_representations
#SBATCH --error=slurm_logs/extract_representations_%j.err
#SBATCH --output=slurm_logs/extract_representations_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

# checkpoint config
flores_devtest_dir="./flores200_dataset"
name_model=""
model_dir=""

src_langs=('spa_Latn' 'eng_Latn' 'fra_Latn' 'ita_Latn' 'cat_Latn' 'por_Latn' 'deu_Latn' 'eus_Latn' 'glg_Latn')

# Iterate over each source language
for src in "${src_langs[@]}"; do
    # Create a directory for each language pair
    save_dir="./results/${name_model}/${src}"
    mkdir -p "$save_dir"

    data_path="${flores_devtest_dir}/${src}.devtest"

    python -u extract_representations.py --model_name_or_path $model_dir \
                --data $data_path \
                --output_dir $save_dir \
                --src_lang_code $src
done