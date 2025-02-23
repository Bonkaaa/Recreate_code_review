from utils import *
import os
import torch
import numpy as np
from metrics import *
from tqdm.auto import tqdm
from args_parse import *


def evaluate(args, model, eval_dataloader, tokenizer, criterion, accelerator=None):
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

            eval_loss += loss.mean().item()

            # generate comments
            generate_ids = accelerator.unwrap_model(model).generate(
                input_ids=in_ids,
                attention_mask=in_masks,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

            # decode generated and actual comments
            generated_comments = tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)
            actual_comments = tokenizer.batch_decode(target_ids, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)

            generated_comments_split = [split_words_and_symbols_for_generated(comment) for comment in generated_comments]   #List[List[str]]
            actual_comments_split = [split_words_and_symbols_for_actuals(comment) for comment in actual_comments]           #List[List[List[str]]]

            # calculate BLEU score
            bleu_score = calculate_bleu_score(actual_comments_split, generated_comments_split)
            EM_score = calculate_exact_match_score(actual_comments, generated_comments)

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