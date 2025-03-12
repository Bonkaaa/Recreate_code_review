import os
from utils import logging
from args_parse import main as args_parse


def load_checkpoint(args, model, original_model, accelerator, prefix):
    try:
        # Load model
        output_dir = f'{args.output_dir}/{prefix}/{args.project}/{args.model_dir}'
        if not os.path.exists(output_dir):
            if accelerator.is_main_process:
                raise FileNotFoundError(f"Model checkpoint not found: {output_dir}")

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.from_pretrained(
            model=original_model,
            model_id=output_dir,
            is_main_process=accelerator.is_main_process
        )

        if accelerator.is_main_process:
            logging.info(f"Found {output_dir}. Load back to accelerator.")

    except Exception as e:
        if accelerator.is_main_process:
            logging.error(f"Load checkpoint failed: {e}")


def save_checkpoint(args, model, accelerator, prefix):
    try:
        output_dir = f'{args.output_dir}/{prefix}/{args.project}/{args.model_dir}'
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )

        if accelerator.is_main_process:
            logging.info(f"Saving model to {output_dir}")
    except Exception as e:
        if accelerator.is_main_process:
            logging.error(f"Save checkpoint failed: {e}")

if __name__ == "__main__":
    args = args_parse()