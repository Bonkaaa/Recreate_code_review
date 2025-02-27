import torch
from args_parse import main as args_parse
from utils import MODEL_CLASSES
from accelerate import Accelerator
from transformers import T5ForConditionalGeneration, T5Tokenizer



def test_model(args,model_dir, test_dataloader, accelerator):
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

    # Retrieve the configuration, model, and tokenizer classes based on the model type specified in the arguments
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Load the model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model = accelerator.prepare(model)
    accelerator.load_state(model_dir)

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model.eval()

    # Prepare the dataloader
    test_dataloader = accelerator.prepare(test_dataloader)

    all_generated_comments = []
    all_actual_comments = []

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
    pass
