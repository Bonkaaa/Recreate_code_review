#!/bin/bash
PROJECT="Code_review_generation"
DataDir="/kaggle/input/100-datapoint/data"

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
ADAPTER_DIR=/adapter/codet5

# TYPE='roberta'
# MODEL='microsoft/unixcoder-base'
# TOKENIZER='microsoft/unixcoder-base'
# OUTPUT_DIR=./outputs/unixcoder

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py \
    --project ${PROJECT} \
    --model_dir ${MODEL} \
    --output_dir=${OUTPUT_DIR} \
    --adapter_dir=${ADAPTER_DIR} \
    --model_type=${TYPE} \
    --tokenizer_name=${TOKENIZER} \
    --model_name_or_path=${MODEL} \
    --train_data_file=${DataDir}"/train.jsonl" \
    --eval_data_file=${DataDir}"/val.jsonl" \
    --epoch 10 \
    --do_eval \
    --block_size 256 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --warmup_steps 1000 \
    --max_grad_norm 1.0 \
    --wandb_name ${MODEL} \
    --evaluate_during_training \
    --num_proc 4