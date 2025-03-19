import numpy as np
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from args_parse import main as args_parse
from metrics import calculate_bleu_score, calculate_exact_match_score
from functools import partial
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import *
from accelerate import Accelerator
from dataset import dataset_loader

def seq2seq_training_ars(args):
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        logging_dir='./logs',
        seed=args.seed,
        run_name=args.wandb_name,
        load_best_model_at_end=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.block_size,
        report_to="wandb",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
    )
    return training_args

def compute_metrics(pred, tokenizer):
    references = pred.label_ids
    generated_texts = pred.predictions

    decoded_references = tokenizer.batch_decode(references, skip_special_tokens=True)
    decoded_generated_texts = tokenizer.batch_decode(generated_texts, skip_special_tokens=True)

    all_bleu_score = []
    all_em_score = []

    for ref, gen in zip(decoded_references, decoded_generated_texts):
        bleu_score = calculate_bleu_score([ref], [gen])
        em_score = calculate_exact_match_score([ref], [gen])['exact_match']
        all_bleu_score.append(bleu_score)
        all_em_score.append(em_score)

    avg_bleu_score = np.mean(all_bleu_score) if all_bleu_score else 0
    avg_em_score = np.mean(all_em_score) if all_em_score else 0

    return {
        "bleu_score": avg_bleu_score,
        "em_score": avg_em_score
    }

def seq2seq_trainer(args, model, training_args, train_dataset, eval_dataset, tokenizer):
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer)
    )
    return trainer

def main(args):

    # Accelerator
    accelerator = Accelerator()

    # Initialize the model
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    if accelerator.is_main_process:
        logging.debug(model)

    # Initialize the tokenizer
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    if accelerator.is_main_process:
        logging.debug(tokenizer)

    # Load data
    train_data = load_jsonl(args.train_data_file)
    eval_data = load_jsonl(args.eval_data_file)

    if accelerator.is_main_process:
        logging.info(f"Total train data: {len(train_data)}")
        logging.info(f"Total validate data: {len(eval_data)}")

    train_dataset, eval_dataset = dataset_loader(args, train_data, eval_data, tokenizer)

    # Prepare accelerator
    model = accelerator.prepare(
        model
    )

    # Load the training arguments
    training_args = seq2seq_training_ars(args)

    # Load the trainer
    trainer = seq2seq_trainer(args, model, training_args, train_dataset, eval_dataset, tokenizer)

    # Train the model
    train_results = trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate(eval_dataset, metric_key_prefix="eval", dataloader_pin_memory=False)

    return train_results, eval_results

if __name__ == "__main__":
    args = args_parse()
    main(args)








