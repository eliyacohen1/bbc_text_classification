from transformers import BertTokenizer

from .tokenizer_model_interface import TokenizerModelInterface


class BertTokenizerModel(TokenizerModelInterface):

    def __init__(self, model_name: str, padding: str, max_length: int, truncation: str, return_tensors: str):
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        self.padding = padding
        self.max_length = max_length
        self.truncation = truncation
        self.return_tensors = return_tensors

    def tokenize_text(self, text: str):
        return self.bert_tokenizer(text, padding=self.padding,
                                   max_length=self.max_length,
                                   truncation=self.truncation,
                                   return_tensors=self.return_tensors)
