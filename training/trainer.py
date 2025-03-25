import numpy as np
import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from args_parse import main as args_parse
from metrics import calculate_bleu_score, calculate_exact_match_score
from functools import partial
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import *
from accelerate import Accelerator
from dataset import dataset_loader
from CustomSeq2SeqTrainer import CustomSeq2SeqTrainer
from transformers import DataCollatorWithPadding
from bleu_score.bleu import Bleu
from evaluate import load

# Initialize the tokenizer
args = args_parse()
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

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
        # metric_for_best_model="bleu_score",
        # greater_is_better=True,
        label_names=["labels"],
        do_train=True,
    )
    return training_args

# def compute_metrics(eval_pred, tokenizer):
#     predictions, labels = eval_pred
#
#     # print(predictions)
#     # raise SystemExit()
#
#     decoded_references = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     decoded_generated_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#     all_bleu_score = []
#     all_em_score = []
#
#     for ref, gen in zip(decoded_references, decoded_generated_texts):
#         bleu_score = calculate_bleu_score([ref], [gen])
#         em_score = calculate_exact_match_score([ref], [gen])
#         all_bleu_score.append(bleu_score)
#         all_em_score.append(em_score)
#
#     avg_bleu_score = np.mean(all_bleu_score) if all_bleu_score else 0
#     avg_em_score = np.mean(all_em_score) if all_em_score else 0
#
#     return {
#         "bleu_score": avg_bleu_score,
#         "em_score": avg_em_score
#     }

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # Decode the predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Calculate BLEU score
    bleu = Bleu()
    bleu_score = bleu.compute(predictions=decoded_preds,
                                references=[[label] for label in decoded_labels],
                                max_order=4
                            )

    # Calculate exact match score
    exact_match_metric = load("exact_match")
    exact_match_score = exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels, ignore_case=True)

    return {
        "bleu_score": bleu_score['bleu_score'],
        "exact_match": exact_match_score['exact_match']
    }


def seq2seq_trainer(args, model, training_args, train_dataset, eval_dataset, tokenizer, data_collator):
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        preprocess_logits_for_metrics = (lambda logits, labels: logits[0].argmax(dim=-1))
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

    data_collator = DataCollatorWithPadding(tokenizer, max_length=args.block_size)

    # Load data
    train_data = load_jsonl(args.train_data_file)[:20]
    eval_data = load_jsonl(args.eval_data_file)[:20]

    if accelerator.is_main_process:
        logging.info(f"Total train data: {len(train_data)}")
        logging.info(f"Total validate data: {len(eval_data)}")

    train_dataset, eval_dataset = dataset_loader(args, train_data, eval_data, tokenizer)

    # Prepare accelerator
    model = accelerator.prepare(model)

    if accelerator.is_main_process:
        logging.info("***** Model Loaded *****")

    # Load the training arguments
    training_args = seq2seq_training_ars(args)

    # Load the trainer
    trainer = seq2seq_trainer(args, model, training_args, train_dataset, eval_dataset, tokenizer, data_collator)

    # Train the model
    train_results = trainer.train()

    if accelerator.is_main_process:
        logging.info("***** Training Finished *****")

    # Evaluate the model
    eval_results = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")

    if accelerator.is_main_process:
        logging.info("***** Evaluation Finished *****")

    # Save the best model
    best_model_dir = os.path.join(training_args.output_dir, "best_model_checkpoint")
    model = accelerator.unwrap_model(model)
    model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    if accelerator.is_main_process:
        logging.info(f"Best model saved to {best_model_dir}")

    return {"train_results": train_results, "eval_results": eval_results}

if __name__ == "__main__":
    args = args_parse()
    results  = main(args)
    accelerator = Accelerator()
    # Print results
    if accelerator.is_main_process:
        logging.info(f"Eval results: {results['eval_results']}")
        logging.info(f"Train results: {results['train_results']}")









