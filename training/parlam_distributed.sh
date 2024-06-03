#!/bin/bash
#SBATCH --job-name=distributed_parlam_32.LLM.Training
#SBATCH --error=slurm_logs/distributed_parlam_32_%j.err
#SBATCH --output=slurm_logs/distributed_parlam_32_%j.out
#SBATCH --nodes=10
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20

export HF_DATASETS_CACHE=
export HF_HOME=

export GPUS_PER_NODE=4
export NNODES=$SLURM_NNODES
export RANK=$SLURM_PROCID
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

TOKENIZER_PATH=''
VOCAB_SIZE=32000
DATASET_PATH=''
DS_CONFIG=''

export CMD="torchrun \
	--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --rdzv_id=$SLURM_JOB_ID \
 	--rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
	run_clm.py \
	--deepspeed $DS_CONFIG \
	--model_type 'gemma' \
	--config_overrides 'vocab_size=${VOCAB_SIZE},attention_bias=false,attention_dropout=0.0,head_dim=256,hidden_size=2048,initializer_range=0.02,intermediate_size=16384,max_position_embeddings=8192,num_attention_heads=8,num_hidden_layers=18,num_key_value_heads=1,rms_norm_eps=1e-06,rope_theta=10000.0,pad_token_id=3,bos_token_id=0,eos_token_id=1' \
	--tokenizer_name $TOKENIZER_PATH \
	--dataset_name  $DATASET_PATH \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--gradient_accumulation_steps 1 \
	--do_train \
	--do_eval \
	--output_dir ./output/parlam_distributed \
	--num_train_epochs 1 \
	--evaluation_strategy steps \
	--eval_steps 5000 \
	--save_strategy steps \
	--save_steps 5000 \
	--logging_strategy steps \
	--logging_steps 5000 \
	--save_total_limit 1 \
	--load_best_model_at_end \
	--warmup_steps 2000 \
	--learning_rate 3e-04 \
	--cache_dir ./cache
	"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH