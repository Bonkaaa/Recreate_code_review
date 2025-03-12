#!/bin/bash
PROJECT="Automated_code_review"
DataDir="/kaggle/input/dataaa/t5_data"

#TYPE='roberta'
#MODEL='microsoft/codebert-base'
#TOKENIZER='microsoft/codebert-base'
#OUTPUT_DIR="./outputs/codebert/"

#TYPE='roberta'
#MODEL='microsoft/graphcodebert-base'
#TOKENIZER='microsoft/graphcodebert-base'
#OUTPUT_DIR=./outputs/graphcodebert/

 TYPE='codet5'
 MODEL='Salesforce/codet5-base'
 TOKENIZER='Salesforce/codet5-base'
 OUTPUT_DIR=./outputs/codet5/

# TYPE='roberta'
# MODEL='microsoft/unixcoder-base'
# TOKENIZER='microsoft/unixcoder-base'
# OUTPUT_DIR=./outputs/unixcoder/

CUDA_VISIBLE_DEVICES=0,1 python -m test \
    --project ${PROJECT} \
    --model_dir="/kaggle/input/adapter/transformers/default/1/outputs/codet5/checkpoint-best-bleu-score/Code_review_generation/Salesforce/codet5-base" \
    --output_dir=${OUTPUT_DIR} \
    --model_type=${TYPE} \
    --tokenizer_name=${TOKENIZER} \
    --model_name_or_path=${MODEL} \
    --test_data_file="/kaggle/input/test-dataset-only/test.jsonl" \
    --block_size 256 \
    --num_proc 4 \
