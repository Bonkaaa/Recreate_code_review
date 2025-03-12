from utils import *
from args_parse import main as args_parse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, T5ForConditionalGeneration
from datasets import Dataset, DatasetDict, load_from_disk
from evaluating import evaluate
from transformers import DataCollatorWithPadding
from accelerate import Accelerator
from checkpoint import save_checkpoint, load_checkpoint
from lora_config import *
from peft import get_peft_model
import numpy as np
import wandb
from typing import Optional
from accelerate.tracking import GeneralTracker
from pathlib import Path
from dotenv import load_dotenv

dotenv_path = Path('./.env')
load_dotenv(dotenv_path=dotenv_path)

def train(args, train_dataloader, eval_dataloader, model, original_model, tokenizer, accelerator):
    # Setup
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.warmup_steps == 0:
        num_warmup = args.max_steps * args.warmup_ratio
    else:
        num_warmup = args.warmup_steps

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup,
                                                num_training_steps=args.max_steps)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_bleu_score = 0.0
    best_em_score = 0.0
    patience = 0

    args.device = accelerator.device
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )
    # load_checkpoint(args, accelerator, 'checkpoint-best-acc')

    # Train
    if accelerator.is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataloader) * args.train_batch_size}")
        logging.info(f"  Num Epochs = {args.num_train_epochs}")
        logging.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    model.zero_grad()

    step = 0
    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        tr_num = 0
        train_loss = 0

        for _, batch in enumerate(bar):
            with accelerator.accumulate(model):
                in_ids = batch['input_ids'].to(args.device)
                in_masks = batch['code_attention_mask'].to(args.device)
                target_ids = batch['target_ids'].to(args.device)

                model.train()

                outputs = model(input_ids = in_ids, attention_mask = in_masks, labels = target_ids)

                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), target_ids.view(-1))

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_value_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                tr_num += 1
                train_loss += loss.item()
                if avg_loss == 0:
                    avg_loss = tr_loss
                avg_loss = round(train_loss / tr_num, 5)
                bar.set_description(f"Epoch {idx} - Loss: {avg_loss}")

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                # # log after every logging_steps (e.g., 20000)
                # if (step + 1) % args.logging_steps == 0:
                #     avg_loss = round(train_loss / tr_num, 5)
                #     if args.evaluate_during_training:
                #         results = evaluate(args, model, eval_dataloader, tokenizer, criterion, accelerator)
                #         if accelerator.is_main_process:
                #             for key, value in results.items():
                #                 logging.info("  %s = %s", key, round(value, 4))
                #         valid_loss, valid_bleu_score = results.values()
                #
                #         accelerator.log({
                #             'Loss/train-per-1000-steps': avg_loss,
                #             'Loss/valid-per-1000-steps': valid_loss,
                #             'Bleu_score/valid-per-1000-steps': valid_bleu_score,
                #         }, step=step)
                #
                #         # Save model checkpoint
                #         if results['eval_bleu_score'] > best_bleu_score:
                #             best_bleu_score = results['eval_bleu_score']
                #             if accelerator.is_main_process:
                #                 logging.info("  " + "*" * 20)
                #                 logging.info("  Best bleu score:%s", round(best_bleu_score, 4))
                #                 logging.info("  " + "*" * 20)
                #             save_checkpoint(args, accelerator, 'checkpoint-best-bleu-score')

                # increment step within the same epoch
                step += 1

        # log after every epoch
        avg_loss = round(train_loss / tr_num, 5)

        if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
            results = evaluate(args, model, eval_dataloader, tokenizer, criterion, accelerator)
            if accelerator.is_main_process:
                for key, value in results.items():
                    logging.info("  %s = %s", key, round(value, 4))
            valid_loss, valid_bleu_score, valid_em_score = results.values()

            accelerator.log({
                'Loss/train-per-epoch': avg_loss,
                'Loss/valid-per-epoch': valid_loss,
                'Bleu_score/valid-per-1000-steps': valid_bleu_score,
                'EM_score/valid-per-1000-steps': valid_em_score,
            }, step=step)

            # save model checkpoint at ep10
            if idx == 9:
                save_checkpoint(args, model, accelerator, f'checkpoint-epoch-{idx + 1}')

            # Save model checkpoint
            if results['eval_bleu_score'] > best_bleu_score:
                best_bleu_score = results['eval_bleu_score']
                if accelerator.is_main_process:
                    logging.info("  " + "*" * 20)
                    logging.info("  Best Bleu Score:%s", round(best_bleu_score, 4))
                    logging.info("  " + "*" * 20)
                save_checkpoint(args, model, accelerator, 'checkpoint-best-bleu-score')
                patience = 0
            else:
                patience += 1

            if results['eval_EM_score'] > best_em_score:
                best_em_score = results['eval_EM_score']
                if accelerator.is_main_process:
                    logging.info("  " + "*" * 20)
                    logging.info("  Best EM Score:%s", round(best_em_score, 4))
                    logging.info("  " + "*" * 20)
                save_checkpoint(args, model, accelerator, 'checkpoint-best-em-score')
                patience = 0
            else:
                patience += 1

        if patience == args.max_patience:
            if accelerator.is_main_process:
                logging.info(f"Reached max patience ({args.max_patience}). End training now.")
            if best_bleu_score == 0.0:
                save_checkpoint(args, model, accelerator, 'checkpoint-best-bleu-score')
            break

    # Final Evaluation
    results = {}
    if args.do_eval:
        load_checkpoint(args, accelerator, 'checkpoint-best-bleu-score')
        result = evaluate(args, model, eval_dataloader, tokenizer, criterion, accelerator)
        if accelerator.is_main_process:
            logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logging.info(f"  {key} = {str(round(result[key], 4))}")

    accelerator.end_training()

    return results


class MyCustomTracker(GeneralTracker):
    def __init__(self, project_name: str, run_name: str, config: dict = {},entity: str = "manh-td120901-singapore-management-university"):
        self.project_name = project_name
        self.run_name = run_name
        self.entity = entity
        self.config = config
        self.name = "wandb"
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=self.run_name,
            config=self.config
        )

    def store_init_configuration(self, values: dict):
        pass

    def log(self, values: dict, step: Optional[int] = None):
        wandb.log(values, step=step)

def main(args):
    tracker = MyCustomTracker(
        project_name=args.project,
        run_name=args.model_type,
        entity="bonkaa",
        config={ 
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "adam_epsilon": args.adam_epsilon,
            "num_train_epochs": args.num_train_epochs,
            "warmup_steps": args.warmup_steps,
            "warmup_ratio": args.warmup_ratio,
            "max_patience": args.max_patience,
            "max_grad_norm": args.max_grad_norm,
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "fp16": args.fp16,
        }            
    )
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with=tracker)
    accelerator.init_trackers(project_name=args.project)

    seed_torch(args.seed)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if accelerator.is_main_process:
        logging.debug(config_class)
        logging.debug(model_class)
        logging.debug(tokenizer_class)

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 2

    if accelerator.is_main_process:
        logging.debug(config)

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    if tokenizer.pad_token == None:
        tokenizer.pad_token = (tokenizer.eos_token)
        tokenizer.pad_token_id = tokenizer(tokenizer.pad_token, truncation=True)['input_ids'][0]

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path)  # Change this to choose another model for evaluating

    # Apply model with LoRA
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    original_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    if accelerator.is_main_process:
        logging.debug(model)

    # Load data
    train_data = load_jsonl(args.train_data_file)
    eval_data = load_jsonl(args.eval_data_file)
    if accelerator.is_main_process:
        logging.info(f"Total train data: {len(train_data)}")
        logging.info(f"Total validate data: {len(eval_data)}")

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
        tokenized_train_dataset = tokenized_train_dataset.rename_column('attention_mask', 'docs_attention_mask')
        tokenized_train_dataset = tokenized_train_dataset.remove_columns(['code_tokens', 'docstring_tokens'])
        tokenized_train_dataset = tokenized_train_dataset.rename_column('code_ids', 'input_ids')
        tokenized_train_dataset = tokenized_train_dataset.rename_column('docs_ids', 'target_ids')
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
        tokenized_val_dataset = tokenized_val_dataset.rename_column('attention_mask', 'docs_attention_mask')
        tokenized_val_dataset = tokenized_val_dataset.remove_columns(['code_tokens', 'docstring_tokens'])
        tokenized_val_dataset = tokenized_val_dataset.rename_column('code_ids', 'input_ids')
        tokenized_val_dataset = tokenized_val_dataset.rename_column('docs_ids', 'target_ids')
        tokenized_val_dataset.save_to_disk(tokenized_val_path)
    tokenized_val_dataset.set_format("torch")
    
    if accelerator.is_main_process:
        logging.info(tokenized_train_dataset)
        logging.info(tokenized_val_dataset)

    # Combine datasets into DatasetDict
    tokenized_datasets = DatasetDict({
        "train": tokenized_train_dataset,
        "validation": tokenized_val_dataset
    })

    # DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=args.train_batch_size, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=args.eval_batch_size, collate_fn=data_collator
    )

    # Training
    results = train(args, train_dataloader, eval_dataloader, model, original_model, tokenizer, accelerator)

    return results


if __name__ == '__main__':
    args = args_parse()
    args.start_epoch = 0
    args.start_step = 0
    main(args)
