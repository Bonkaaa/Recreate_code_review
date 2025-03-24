#!/bin/bash
PROJECT="Code_review_generation"
DataDir="../t5_data"

# TYPE='roberta'
# MODEL='microsoft/codebert-base'
# TOKENIZER='microsoft/codebert-base'
# OUTPUT_DIR="./accel_outputs/codebert"

#TYPE='roberta'
#MODEL='microsoft/graphcodebert-base'
#TOKENIZER='microsoft/graphcodebert-base'
#OUTPUT_DIR=./outputs/graphcodebert

TYPE='codet5'
MODEL='Salesforce/codet5-base'
TOKENIZER='Salesforce/codet5-base'
OUTPUT_DIR=./outputs/codet5

# TYPE='roberta'
# MODEL='microsoft/unixcoder-base'
# TOKENIZER='microsoft/unixcoder-base'
# OUTPUT_DIR=./outputs/unixcoder

CUDA_VISIBLE_DEVICES=1,5 accelerate launch trainer.py \
    --project ${PROJECT} \
    --model_dir ${MODEL} \
    --output_dir=${OUTPUT_DIR} \
    --model_type=${TYPE} \
    --tokenizer_name=${TOKENIZER} \
    --model_name_or_path=${MODEL} \
    --train_data_file=${DataDir}"/train.jsonl" \
    --eval_data_file=${DataDir}"/val.jsonl" \
    --num_train_epochs 2 \
    --do_eval \
    --block_size 256 \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --learning_rate 2e-5 \
    --warmup_steps 1000 \
    --max_grad_norm 1.0 \
    --wandb_name ${MODEL} \
    --num_proc 4