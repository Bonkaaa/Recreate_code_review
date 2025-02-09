#!/bin/bash
PROJECT="detection"
DataDir="./data_c/"

#TYPE='roberta'
#MODEL='microsoft/codebert-base'
#TOKENIZER='microsoft/codebert-base'
#OUTPUT_DIR="./output/codebert/"

#TYPE='roberta'
#MODEL='microsoft/graphcodebert-base'
#TOKENIZER='microsoft/graphcodebert-base'
#OUTPUT_DIR=./output/graphcodebert/

#TYPE='codet5'
#MODEL='Salesforce/codet5-base'
#TOKENIZER='Salesforce/codet5-base'
#OUTPUT_DIR=./output/codet5/

TYPE='roberta'
MODEL='microsoft/unixcoder-base'
TOKENIZER='microsoft/unixcoder-base'
OUTPUT_DIR=./output/unixcoder/


CUDA_VISIBLE_DEVICES=2 python -m 2_training.args_parse \
    --project ${PROJECT} \
    --model_dir ${MODEL} \
    --output_dir=${OUTPUT_DIR} \
    --model_type=${TYPE} \
    --tokenizer_name=${TOKENIZER} \
    --model_name_or_path=${MODEL} \
    --do_train \
    --do_test \
    --train_data_file=${DataDir}"/train.jsonl" \
    --eval_data_file=${DataDir}"/valid.jsonl" \
    --test_data_file=${DataDir}"/unique_test_data.jsonl" \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --warmup_steps 1000 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fp16 \
    --seed 123456
