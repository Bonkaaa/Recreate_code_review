from torch.utils.data import Dataset
from utils import *


# CustomDataset
class CustomDataset(Dataset):
    def __init__(self, path, tokenizer, code_lines_len, comments_len):
        """
        Initializes Dataset class

        :param path: where data locates
        :param tokenizer: Transformer tokenizer
        :param code_lines_len: max length of codes
        :param comments_len: max length of comments\
        """

        self.data = load_jsonl(path)
        self.code_lines = self.data['Code_tokens'].tolist()
        self.comments = self.data['Docstring_tokens'].tolist()
        self.tokenizer = tokenizer
        self.code_lines_len = code_lines_len
        self.comments_len = comments_len

    def __len__(self):  # return the length
        return len(self.code_lines)

    def __getitem__(self, idx):
        code = str(self.code_lines[idx])
        comment = str(self.comments[idx])

        # Tokenizer source and target
        source_encoding = self.tokenizer.encode(
            code,
            max_length=self.code_lines_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        target_encoding = self.tokenizer.encode(
            comment,
            max_length=self.comments_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        source_ids = source_encoding['input_ids'].squeeze()
        source_mask = source_encoding['attention_mask'].squeeze()
        target_ids = target_encoding['input_ids'].squeeze()

        return {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,
        }

