import os
from utils import logging
from args_parse import main as args_parse

def load_checkpoint(args, accelerator, prefix):
    try:
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
            logging.error(f"Save checkpoint failed: {e}")

def save_checkpoint(args, accelerator, prefix):
    try:
        output_dir = f'{args.output_dir}/{prefix}/{args.project}/{args.model_dir}'
        os.makedirs(output_dir, exist_ok=True)
        accelerator.wait_for_everyone()
        accelerator.save_state(output_dir)
        if accelerator.is_main_process:
            logging.info(f"Saving model to {output_dir}")
    except Exception as e:
        if accelerator.is_main_process:
            logging.error(f"Save checkpoint failed: {e}")

if __name__ == "__main__":
    args = args_parse()