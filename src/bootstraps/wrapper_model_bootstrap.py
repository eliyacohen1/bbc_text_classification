import sys

sys.path.insert(0, "..")

from exceptions import NoSuchModelWrapper
from model import WrapperClassificationBertModel
from model.classification_model import BertClassifier


def wrapper_model_bootstrap(config: dict):
    if config['model_name'] == "bert":
        bert_config = config["bert_settings"]
        model = BertClassifier(pretrained_name=bert_config['model_name'],
                               layers=bert_config["layers"],
                               num_classes=bert_config["num_classes"],
                               activation_func=bert_config["activation_func"],
                               dropout=bert_config["dropout"])
        return model
    return NoSuchModelWrapper
