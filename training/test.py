import torch
import os
import pandas as pd
from args_parse import main as args_parse
from utils import load_jsonl
from utils import MODEL_CLASSES
from accelerate import Accelerator
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import logging
from datasets import Dataset, load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from bnb_config import get_bnb_config
from peft import get_peft_model
from lora_config import get_lora_config

def test_model(args, model_dir, test_dataloader, model, tokenizer, accelerator):
    """
        Tests the model with the provided test dataset.

        Args:
            model_dir (str): The directory where the model checkpoint is stored.
            test_dataloader (DataLoader): The DataLoader object for the test dataset.
            accelerator (Accelerator): Accelerator object for distributed training.

        Returns:
            list: A list of generated comments for the test dataset.
        """

    # Initialize the accelerator
    args.device = accelerator.device

    # Load the model and tokenizer
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    model = unwrapped_model.from_pretrained(
        pretrained_model_name_or_path=model_dir
    )
    model = accelerator.prepare(model)
    # Initialize the lists to store the generated and actual comments
    all_generated_comments = []
    all_actual_comments = []
    model.eval()

    for batch in test_dataloader:
        in_ids = batch['input_ids'].to(args.device)
        in_masks = batch['code_attention_mask'].to(args.device)
        target_ids = batch['target_ids'].to(args.device)
        with torch.no_grad():
            # Forward pass
            outputs = model(in_ids, attention_mask=in_masks, labels=target_ids)
            if accelerator:
                outputs = accelerator.gather(outputs)
                target_ids = accelerator.gather(target_ids)

            # Decode the generated comments
            generated_comments = tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
            all_generated_comments.extend(generated_comments)

            # Decode the actual comments
            actual_comments = tokenizer.batch_decode(target_ids, skip_special_tokens=True)
            all_actual_comments.extend(actual_comments)

    return all_generated_comments, all_actual_comments

if __name__ == "__main__":
    args = args_parse()
    accelerator = Accelerator()

    # Retrieve the configuration, model, and tokenizer classes based on the model type specified in the arguments
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # Apply QLoRA
    bnb_config = get_bnb_config()

    # Load model with QLoRA
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
    )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    # Load model with LoRA
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    # Load the test dataset
    test_data = load_jsonl(args.test_data_file)[:10] # Load only 10 samples for testing

    if accelerator.is_main_process:
        logging.info(f"Total test data: {len(test_data)}")

    # Process the list
    for entry in test_data:
        entry["code_tokens"] = " ".join(entry["code_tokens"])  # Concatenate code tokens
        entry["docstring_tokens"] = " ".join(entry["docstring_tokens"])  # Concatenate docstring tokens

    test_dataset = Dataset.from_list(test_data)


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
    tokenized_test_path = "./tokenized_dataset/test"
    if os.path.exists(tokenized_test_path):
        tokenized_test_dataset = load_from_disk(tokenized_test_path)
    else:
        tokenized_test_dataset = test_dataset.map(tokenize_code, batched=True, num_proc=args.num_proc)
        tokenized_test_dataset = tokenized_test_dataset.rename_column('input_ids', 'code_ids')
        tokenized_test_dataset = tokenized_test_dataset.rename_column('attention_mask', 'code_attention_mask')
        tokenized_test_dataset = tokenized_test_dataset.map(tokenize_docstring, batched=True, num_proc=args.num_proc)
        tokenized_test_dataset = tokenized_test_dataset.rename_column('input_ids', 'docs_ids')
        tokenized_test_dataset = tokenized_test_dataset.rename_column('attention_mask', 'docs_attention_mask')
        tokenized_test_dataset = tokenized_test_dataset.remove_columns(['code_tokens', 'docstring_tokens'])
        tokenized_test_dataset = tokenized_test_dataset.rename_column('code_ids', 'input_ids')
        tokenized_test_dataset = tokenized_test_dataset.rename_column('docs_ids', 'target_ids')
        tokenized_test_dataset.save_to_disk(tokenized_test_path)
    tokenized_test_dataset.set_format("torch")

    #Combine datasets into DatasetDict
    test_dataset = DatasetDict({'test': tokenized_test_dataset})

    # DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    test_dataloader = DataLoader(
        test_dataset["test"], shuffle=True, batch_size= 4, collate_fn=data_collator
    )

    # Test the model
    all_generated_comments, all_actual_comments = test_model(args, args.model_dir, test_dataloader, model, original_model, tokenizer, accelerator)

    df = pd.DataFrame({"actual_comments": all_actual_comments, "generated_comments": all_generated_comments})
    df.to_csv("test_results.csv")
    print("Test results saved")