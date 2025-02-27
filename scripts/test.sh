#!/bin/bash
PROJECT="repo-specific"
DataDir="/raid/data/dmtran/GovTech/repo-specific/top_5_most"

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

CUDA_VISIBLE_DEVICES=1 python -m test \
    --project ${PROJECT} \
    --model_dir = "C:\neutral_network\outputs\codet5\checkpoint-epoch-10\Code_review_generation\Salesforce\codet5-base" \
    --output_dir=${OUTPUT_DIR} \
    --model_type=${TYPE} \
    --tokenizer_name=${TOKENIZER} \
    --model_name_or_path=${MODEL} \
    --test_data_file= "C:\neutral_network\DATA\Review-Comment-Generation\t5_data\test.jsonl" \
    --block_size 256 \
    --num_proc 4 \
