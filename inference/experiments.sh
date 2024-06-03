#!/bin/bash
#SBATCH --job-name=32experiments
#SBATCH --error=slurm_logs/32experiments_beam_%j.err
#SBATCH --output=slurm_logs/32experiments_beam_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

# checkpoint config
checkpoint=645000
name="parlam_distributed"
vocab_size=32000

# Dataset config
dataset="flores_devtest_beam_5" # flores_devtest
flores_dev_dir="./flores200_dataset/dev"
flores_devtest_dir="./flores200_dataset/devtest"

src_langs=('spa_Latn' 'eng_Latn' 'fra_Latn' 'ita_Latn' 'cat_Latn' 'por_Latn' 'deu_Latn' 'eus_Latn' 'glg_Latn')
tgt_langs=('spa_Latn' 'eng_Latn' 'fra_Latn' 'ita_Latn' 'cat_Latn' 'por_Latn' 'deu_Latn' 'eus_Latn' 'glg_Latn')

model_dir=""
# Name of the model output
name_model="checkpoint-${checkpoint}_${name}"

# Iterate over each source language
for src in "${src_langs[@]}"; do
    # Iterate over each target language
    for tgt in "${tgt_langs[@]}"; do
        # Check if source and target languages are not the same
        if [ "$src" != "$tgt" ]; then
            # Create a directory for each language pair
            save_dir="./translations/${dataset}/${src}-${tgt}"
            mkdir -p "$save_dir"

            data_path="${flores_devtest_dir}/${src}.devtest"
            beam=5

            # Run the translation generation script
            python -u generate.py --src_lang_code $src \
                           --tgt_lang_code $tgt \
                           --vocab_size $vocab_size \
                           --model $model_dir \
                           --data $data_path \
                           --output "${save_dir}/${name_model}.txt" \
                           --beam $beam \
                           --checkpoint hf \
                           --penalty 1.0 \
                           --float16 'no' \
                           --ignore_source 'no'
        fi
    done
done