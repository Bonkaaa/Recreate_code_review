from utils import *
import os
import torch
import numpy as np
from metrics import *
from tqdm.auto import tqdm
from args_parse import *
from transformers import RobertaTokenizer

def evaluate(args, model, eval_dataloader, tokenizer:RobertaTokenizer, criterion, accelerator = None):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    if accelerator and accelerator.is_main_process:
        logging.info("***** Running evaluation *****")
        logging.info(f"  Num examples = {len(eval_dataloader) * args.eval_batch_size}")
        logging.info(f"  Batch size = {args.eval_batch_size}")

    eval_loss = 0.0
    nb_eval_steps = 0
    all_bleu_score = []
    all_em_score = []
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating:", disable=not accelerator.is_local_main_process):
        in_ids = batch['input_ids'].to(args.device)
        in_masks = batch['code_attention_mask'].to(args.device)
        target_ids = batch['target_ids'].to(args.device)
        with torch.no_grad():

            #forward
            outputs = model(in_ids, in_masks, target_ids)

            #Compute loss
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), target_ids.view(-1))
            
            if accelerator:
                loss = accelerator.gather(loss)
                outputs = accelerator.gather(outputs)
                target_ids = accelerator.gather(target_ids)
                
            #decode the generated comments
            generated_comments = tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)  # decode the generated comments   
            actual_comments = tokenizer.batch_decode(target_ids, skip_special_tokens=True)  # decode the actual comments
            
            eval_loss += loss.mean().item()

            # calculate BLEU score
            if accelerator.is_main_process:
                logging.info(f"Generated comments: {generated_comments}")
                logging.info(f"Actual comments: {actual_comments}") 
                
            bleu_score = calculate_bleu_score(actual_comments, generated_comments)
            EM_score = calculate_exact_match_score(actual_comments, generated_comments)
            if accelerator.is_main_process:
                logging.info(f"BLEU score: {bleu_score}")
                logging.info(f"EM score: {EM_score}")

            # add all the score for avg later
            all_bleu_score.append(bleu_score)
            all_em_score.append(EM_score)

        nb_eval_steps += 1

    #calculate avg
    avg_bleu_score = np.mean(all_bleu_score) if all_bleu_score else 0
    avg_EM_score = np.mean(all_em_score) if all_em_score else 0
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_bleu_score": avg_bleu_score,
        "eval_EM_score": avg_EM_score,
    }
    return result