#!/bin/bash
#SBATCH --job-name=overlapping
#SBATCH --error=slurm_logs/overlapping%j.err
#SBATCH --output=slurm_logs/overlapping%j.out
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=01-23:59:59
#SBATCH --cpus-per-task=20
#SBATCH --qos acc_bscls
#SBATCH -N1
#SBATCH --account bsc88
##SBATCH --qos=acc_bscls

source /gpfs/projects/bsc88/mt_translation/environments/training_hf_venv/bin/activate

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
flores_path="/gpfs/projects/bsc88/mt_translation/data/flores200_dataset/${flores}"

len_voc="32k"
tokenizer_path="/gpfs/projects/bsc88/mt_translation/Parallel_LLM/training/saved_checkpoints/gemma32_distributed/checkpoint-315000"
compute_overlapping

len_voc="128k"
tokenizer_path="/gpfs/projects/bsc88/mt_translation/Parallel_LLM/training/saved_checkpoints/gemma128_distributed/checkpoint-320000"
compute_overlapping

len_voc="256k"
tokenizer_path="/gpfs/projects/bsc88/mt_translation/Parallel_LLM/training/saved_checkpoints/gemma256_distributed/checkpoint-300000"
compute_overlapping