from typing import List

import torch
import numpy as np
from transformers import BertTokenizer


class DatasetForBert(torch.utils.data.Dataset):

    def __init__(self, text_list: List[str], label_list: List[str], config_labels: dict, config_tokenizer: dict):
        bert_tokenizer = BertTokenizer.from_pretrained(config_tokenizer["bert_tokenizer_name"])
        self.labels = [config_labels[label] for label in label_list]
        self.texts = [bert_tokenizer(text,
                                     padding=config_tokenizer["padding"],
                                     max_length=config_tokenizer["max_length"],
                                     truncation=config_tokenizer["truncation"],
                                     return_tensors=config_tokenizer["return_tensors"]) for text in text_list]

    def __len__(self):
        return len(self.labels)

    def get_labels(self, idx):
        return np.array(self.labels[idx])

    def get_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        texts = self.get_texts(idx)
        y = self.get_labels(idx)

        return texts, y
