import logging
import os
import json
import random
import numpy as np
import torch
from datetime import datetime
from transformers import (BertConfig, BertForMaskedLM, BertTokenizer)
from transformers import (T5Config, T5ForConditionalGeneration)
from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

MODEL_CLASSES = {
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

PRODUCT = True
LOGS_DIR = "./logs"
os.makedirs(LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(LOGS_DIR, 'train.log')
logging.basicConfig(
    force=True,
    level=logging.INFO if PRODUCT else logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)


def load_jsonl(file_path):
    """
    Load a JSON Lines (JSONL) file and return its content as a list of dictionaries.

    :param file_path: Path to the JSON Lines file.
    :return: List of dictionaries (or other JSON-parsable objects).
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file if line.strip()]


def dump_jsonl(data, file_path):
    """
    Dump a list of Python dictionaries or lists to a JSON Lines file.

    :param data: List of Python dictionaries or lists to save as JSON Lines.
    :param file_path: Path to save the JSON Lines file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_cuda_devices():
    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        return

    num_devices = torch.cuda.device_count()
    print(f"Number of available CUDA devices: {num_devices}")

    for device_idx in range(num_devices):
        device_name = torch.cuda.get_device_name(device_idx)
        print(f"Device Index: {device_idx}, Device Name: {device_name}")


def is_sorted_by_date(dict_list, date_format="%Y-%m-%dT%H:%M:%SZ"):
    """
    Checks if a list of dictionaries is sorted by the 'date' field in ascending order.

    Args:
        dict_list (list): A list of dictionaries containing a 'date' field.
        date_field (str): The key for the date field in the dictionaries.
        date_format (str): The format of the date strings in the 'date' field.

    Returns:
        bool: True if the list is sorted by date, False otherwise.
    """
    for i in range(len(dict_list) - 1):
        current_date = datetime.strptime(dict_list[i]["date"], date_format)
        next_date = datetime.strptime(dict_list[i + 1]["date"], date_format)

        if current_date > next_date:
            return False
    return True


def is_data_sorted(train_dataset: list, test_dataset: list, date_format="%Y-%m-%dT%H:%M:%SZ") -> bool:
    if not is_sorted_by_date(train_dataset, date_format):
        logging.error("Train set is not sorted by date")
        return False

    if not is_sorted_by_date(test_dataset, date_format):
        logging.error("Test set is not sorted by date")
        return False

    train_last_date = datetime.strptime(train_dataset[-1]["date"], date_format)
    test_first_date = datetime.strptime(test_dataset[0]["date"], date_format)

    if train_last_date > test_first_date:
        logging.error("Test set contains data prior to Train set")
        return False

    logging.info("There is no data leakage in train set")
    return True


if __name__ == "__main__":
    check_cuda_devices()