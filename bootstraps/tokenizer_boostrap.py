import sys
from src.models.tokenizer_model.bert_tokenizer import BertTokenizerModel

sys.path.insert(0, "..")
from src.models.tokenizer_model.tokenizer_model_interface import TokenizerModelInterface


def tokenizer_bootstrap(config: dict) -> TokenizerModelInterface:
    if config["model_name"] == "bert":
        tokenizer_config = config["tokenizer_config"]
        return BertTokenizerModel(tokenizer_config["model_name"], tokenizer_config["padding"],
                                  tokenizer_config["max_length"], tokenizer_config["truncation"],
                                  tokenizer_config["return_tensors"])
