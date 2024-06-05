#!/bin/bash
#SBATCH --job-name=overlapping
#SBATCH --error=slurm_logs/overlapping%j.err
#SBATCH --output=slurm_logs/overlapping%j.out
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

lang=(deu eus fra ita glg spa por eng)

compute_overlapping(){
	for src in ${lang[@]}; do
		for tgt in ${lang[@]}; do
			if [ "$src" != "$tgt" ]; then
				python compute_overlapping.py \
				-src ${src} \
				-tgt ${tgt} \
				-t ${tokenizer_path} \
				-f ${flores_path} >> results_overlapping_${len_voc}.txt 
			fi
		done
	done
}

flores="devtest"
flores_path="./flores200_dataset/${flores}"

len_voc="32k"
tokenizer_path=""
compute_overlapping

len_voc="128k"
tokenizer_path=""
compute_overlapping

len_voc="256k"
tokenizer_path=""
compute_overlapping