#!/bin/bash
#SBATCH --job-name=get_att_matrix
#SBATCH --error=slurm_logs/get_att_matrix_%j.err
#SBATCH --output=slurm_logs/get_att_matrix_%j.out
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

# checkpoint config
checkpoint=360000
name="parlam_distributed"

# Dataset config
dataset="flores_devtest" # flores_devtest
flores_dev_dir="./flores200_dataset/dev"
flores_devtest_dir="./flores200_dataset/devtest"

src_langs=('spa_Latn')
tgt_langs=('spa_Latn' 'eng_Latn')

model_dir=""

# Iterate over each source language
for src in "${src_langs[@]}"; do
    # Iterate over each target language
    for tgt in "${tgt_langs[@]}"; do
        # Check if source and target languages are not the same
        if [ "$src" != "$tgt" ]; then
            # Name of the model output
            name_model="checkpoint-${checkpoint}_${name}"

            # Create a directory for each language pair
            save_dir="./results/${dataset}/${name_model}/${src}-${tgt}"
            mkdir -p "$save_dir"

            if [ "$dataset" == "flores_devtest" ]; then
                data_path="${flores_devtest_dir}/${src}.devtest"
            fi

            # Run the translation generation script
            python -u get_att_matrix.py --src_lang_code $src \
                           --tgt_lang_code $tgt \
                           --model $model_dir \
                           --data $data_path \
                           --output $save_dir \
                           --float16
        fi
    done
done