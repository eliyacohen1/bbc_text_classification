from typing import List

import sys

sys.path.insert(0, "..")

from src.dataset.dataset_interface import DatasetInterface
from src.models.tokenizer_model.tokenizer_model_interface import TokenizerModelInterface


class DatasetForBert(DatasetInterface):

    def __init__(self, text_list: List[str], label_list: List[str], config_labels: dict,
                 tokenizer: TokenizerModelInterface):
        self.labels = [config_labels[label] for label in label_list]
        self.texts = [tokenizer.tokenize_text(text) for text in text_list]
