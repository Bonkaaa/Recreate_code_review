import os
import torch
from utils import logging
from args_parse import main as args_parse
from peft import PeftModel

def load_checkpoint(args, model, accelerator, prefix):
    try:
        adapter_dir = f'{args.output_dir}/{prefix}/{args.project}/{args.adapter_dir}'
        if os.path.exists(adapter_dir):
            if accelerator.is_main_process:
                logging.debug(f"Found {adapter_dir}. Load back to model.")
            PeftModel.from_pretrained(model, adapter_dir)
        else:
            if accelerator.is_main_process:
                logging.debug("No adapter to load")
        output_dir = f'{args.output_dir}/{prefix}/{args.project}/{args.model_dir}'
        if os.path.exists(output_dir):
            if accelerator.is_main_process:
                logging.debug(f"Found {output_dir}. Load back to accelerator.")
            accelerator.load_state(output_dir)
        else:
            if accelerator.is_main_process:
                logging.debug("No accelerator to load")
    except Exception as e:
        if accelerator.is_main_process:
            logging.error(f"Load checkpoint failed: {e}")

def save_checkpoint(args, model, accelerator, prefix):
    try:
        output_dir = f'{args.output_dir}/{prefix}/{args.project}/{args.model_dir}'
        adapter_dir = f'{args.output_dir}/{prefix}/{args.project}/{args.adapter_dir}'
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)

        # Save adapter
        local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if local_rank == 0:
            model.module.save_pretrained(output_dir)
        if accelerator.is_main_process:
            logging.info(f"Saving adapter to {adapter_dir}")

        # Save model
        accelerator.wait_for_everyone()
        accelerator.save_state(output_dir)
        if accelerator.is_main_process:
            logging.info(f"Saving model to {output_dir}")
    except Exception as e:
        if accelerator.is_main_process:
            logging.error(f"Save checkpoint failed: {e}")

if __name__ == "__main__":
    args = args_parse()