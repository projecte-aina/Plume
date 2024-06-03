#!/bin/bash
#SBATCH --job-name=head_mask
#SBATCH --error=slurm_logs/head_mask_%j.err
#SBATCH --output=slurm_logs/head_mask_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

ATT_ANALYSIS_FULL_PATH=""

name="parlam_distributed"
checkpoint=645000
model_dir=""
flores_devtest_dir="./flores200_dataset/devtest"

src=('spa_Latn' 'fra_Latn' 'deu_Latn' 'glg_Latn' 'ita_Latn')
tgt=('eng_Latn')

for src_idx in "${!src[@]}"; do
    for tgt_idx in "${!tgt[@]}"; do

        src="${src[$src_idx]}"
        tgt="${tgt[$tgt_idx]}"

        att_matrix_dir="${ATT_ANALYSIS_FULL_PATH}/results/flores_devtest/checkpoint-${checkpoint}_${name}/${src}-${tgt}"
        data_path="${flores_devtest_dir}/${src}.devtest"
        tgt_data_path="${flores_devtest_dir}/${tgt}.devtest"

        save_dir="./results/${name}/${src}-${tgt}"
        mkdir -p "$save_dir"

        python -u remove_by_layer.py    --src_lang_code $src \
                                        --tgt_lang_code $tgt \
                                        --model $model_dir \
                                        --att_matrix_dir $att_matrix_dir \
                                        --data $data_path \
                                        --tgt_data $tgt_data_path \
                                        --output $save_dir
    done
done