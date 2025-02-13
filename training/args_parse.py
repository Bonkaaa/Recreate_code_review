import argparse
from utils import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', type=str, required=True, help="using dataset from this project.")
    parser.add_argument('--test_project', type=str, required=False, help="test setup name.")
    parser.add_argument('--model_dir', type=str, required=True, help="directory to store the model weights.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--run_dir', type=str, default="runs", help="parent directory to store run stats.")

    parser.add_argument("--max_source_length", default=400, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--train_data_file", default=None, type=str, help="The input training data file (a text file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--finetune_data_file", default=None, type=str,
                        help="The input training data file (a text file).")

    parser.add_argument("--model_type", default="bert", type=str, help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_fine_tuning", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    parser.add_argument("--class_weight", action='store_true',
                        help="Use class weight in the regular cross entropy loss.")
    parser.add_argument("--vul_weight", default=1.0, type=float,
                        help="Weight for the vulnerable class in the regular cross entropy loss.")

    parser.add_argument("--train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float, help="Linear warmup ratio over all steps.")
    parser.add_argument("--test_size", default=0.05, type=float, help="Linear warmup ratio over all steps.")

    parser.add_argument('--logging_steps', type=int, default=20000, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--max-patience', type=int, default=-1, help="Max iterations for model with no improvement.")
    parser.add_argument('--save_prediction_filename', type=str, default="predictions.txt")
    parser.add_argument('--wandb_name', type=str, default="WanDB run")

    parser.add_argument("--num_proc", type=int, default=1, help="Number of Processes")

    args = parser.parse_args()
    if not PRODUCT:
        logging.debug(json.dumps(vars(args), indent=4))
    return args


if __name__ == "__main__":
    args = main()
    logging.debug(json.dumps(vars(args), indent=4))