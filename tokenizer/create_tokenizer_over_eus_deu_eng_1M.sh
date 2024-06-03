#!/bin/bash
#SBATCH --job-name=create_tokenizer
#SBATCH --error=slurm_logs/create_tokenizer_%j.err
#SBATCH --output=slurm_logs/create_tokenizer_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0

FILES_DIR=''

# BPE

sampling_name="sampling_over_eus_deu_eng_1M"
tok_type='bpe'
tokenizer_size=32000
python -u build_tokenizer.py --vocab_size $tokenizer_size \
                             --files_directory $FILES_DIR \
                             --output "./tokenizers/$tok_type.$sampling_name/size_$tokenizer_size" \
                             --tokenizer_type $tok_type

sampling_name="sampling_over_eus_deu_eng_1M"
tok_type='bpe'
tokenizer_size=128000
python -u build_tokenizer.py --vocab_size $tokenizer_size \
                             --files_directory $FILES_DIR \
                             --output "./tokenizers/$tok_type.$sampling_name/size_$tokenizer_size" \
                             --tokenizer_type $tok_type

sampling_name="sampling_over_eus_deu_eng_1M"
tok_type='bpe'
tokenizer_size=256000
python -u build_tokenizer.py --vocab_size $tokenizer_size \
                             --files_directory $FILES_DIR \
                             --output "./tokenizers/$tok_type.$sampling_name/size_$tokenizer_size" \
                             --tokenizer_type $tok_type

# UNIGRAM

sampling_name="sampling_over_eus_deu_eng_1M"
tok_type='unigram'
tokenizer_size=32000
python -u build_tokenizer.py --vocab_size $tokenizer_size \
                             --files_directory $FILES_DIR \
                             --output "./tokenizers/$tok_type.$sampling_name/size_$tokenizer_size"

sampling_name="sampling_over_eus_deu_eng_1M"
tok_type='unigram'
tokenizer_size=128000
python -u build_tokenizer.py --vocab_size $tokenizer_size \
                             --files_directory $FILES_DIR \
                             --output "./tokenizers/$tok_type.$sampling_name/size_$tokenizer_size"


sampling_name="sampling_over_eus_deu_eng_1M"
tok_type='unigram'
tokenizer_size=256000
python -u build_tokenizer.py --vocab_size $tokenizer_size \
                             --files_directory $FILES_DIR \
                             --output "./tokenizers/$tok_type.$sampling_name/size_$tokenizer_size"