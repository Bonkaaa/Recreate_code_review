import os
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from args_parse import main as args_parse

def dataset_loader(args, train_data, eval_data, tokenizer):
    # Process the list
    for entry in train_data:
        entry["code_tokens"] = " ".join(entry["code_tokens"])  # Concatenate code tokens
        entry["docstring_tokens"] = " ".join(entry["docstring_tokens"])  # Concatenate docstring tokens

    for entry in eval_data:
        entry["code_tokens"] = " ".join(entry["code_tokens"])  # Concatenate code tokens
        entry["docstring_tokens"] = " ".join(entry["docstring_tokens"])  # Concatenate docstring tokens

    # Convert to Dataset objects
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(eval_data)

    # # Tokenize function
    def tokenize_code(example):
        code = example["code_tokens"]
        return tokenizer(
            code,
            padding='max_length',
            truncation=True,
            max_length=args.block_size
        )

    def tokenize_docstring(example):
        docstring_tokens = example["docstring_tokens"]
        return tokenizer(
            docstring_tokens,
            padding='max_length',
            truncation=True,
            max_length=args.block_size
        )

    # Tokenize datasets
    tokenized_train_path = "./tokenized_dataset/train"
    if os.path.exists(tokenized_train_path):
        tokenized_train_dataset = load_from_disk(tokenized_train_path)
    else:
        tokenized_train_dataset = train_dataset.map(tokenize_code, batched=True, num_proc=args.num_proc)
        tokenized_train_dataset = tokenized_train_dataset.rename_column('input_ids', 'code_ids')
        tokenized_train_dataset = tokenized_train_dataset.rename_column('attention_mask', 'code_attention_mask')
        tokenized_train_dataset = tokenized_train_dataset.map(tokenize_docstring, batched=True, num_proc=args.num_proc)
        tokenized_train_dataset = tokenized_train_dataset.rename_column('input_ids', 'docs_ids')
        tokenized_train_dataset = tokenized_train_dataset.rename_column('attention_mask', 'decoder_attention_mask')
        tokenized_train_dataset = tokenized_train_dataset.remove_columns(['code_tokens', 'docstring_tokens'])
        tokenized_train_dataset = tokenized_train_dataset.rename_column('code_ids', 'input_ids')
        tokenized_train_dataset = tokenized_train_dataset.rename_column('docs_ids', 'decoder_input_ids')
        tokenized_train_dataset = tokenized_train_dataset.rename_column('code_attention_mask', 'attention_mask')
        tokenized_train_dataset.save_to_disk(tokenized_train_path)
    tokenized_train_dataset.set_format("torch")

    tokenized_val_path = "./tokenized_dataset/val"
    if os.path.exists(tokenized_val_path):
        tokenized_val_dataset = load_from_disk(tokenized_val_path)
    else:
        tokenized_val_dataset = val_dataset.map(tokenize_code, batched=True, num_proc=args.num_proc)
        tokenized_val_dataset = tokenized_val_dataset.rename_column('input_ids', 'code_ids')
        tokenized_val_dataset = tokenized_val_dataset.rename_column('attention_mask', 'code_attention_mask')
        tokenized_val_dataset = tokenized_val_dataset.map(tokenize_docstring, batched=True, num_proc=args.num_proc)
        tokenized_val_dataset = tokenized_val_dataset.rename_column('input_ids', 'docs_ids')
        tokenized_val_dataset = tokenized_val_dataset.rename_column('attention_mask', 'decoder_attention_mask')
        tokenized_val_dataset = tokenized_val_dataset.remove_columns(['code_tokens', 'docstring_tokens'])
        tokenized_val_dataset = tokenized_val_dataset.rename_column('code_ids', 'input_ids')
        tokenized_val_dataset = tokenized_val_dataset.rename_column('docs_ids', 'labels')
        tokenized_val_dataset = tokenized_val_dataset.rename_column('code_attention_mask', 'attention_mask')
        tokenized_val_dataset.save_to_disk(tokenized_val_path)
    tokenized_val_dataset.set_format("torch")

    # Combine datasets into DatasetDict
    tokenized_datasets = DatasetDict({
        "train": tokenized_train_dataset,
        "validation": tokenized_val_dataset
    })

    return tokenized_datasets['train'], tokenized_datasets['validation']