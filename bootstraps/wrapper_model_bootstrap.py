import sys

sys.path.insert(0, "..")
from src.models.embedding_model.bert_embbeding import BertEmbedding
from src.models.classification_model.mlp_classifier import MLPClassifier
from src.models.bert_text_classification import BertTextClassification


def wrapper_model_bootstrap(config: dict):
    if config['embedding'] == "bert":
        bert_config = config["bert_settings"]
        embed_model = BertEmbedding(pretrained_name=bert_config['model_name'])
    else:
        raise ValueError("Not Such Embedding Model")
    if config["classifier"] == "mlp":
        mlp_config = config["mlp_settings"]
        cls_model = MLPClassifier(layers=mlp_config["layers"],
                                  num_classes=mlp_config["num_classes"],
                                  activation_func=mlp_config["activation_func"],
                                  dropout=mlp_config["dropout"])
    else:
        raise ValueError("Not Such CLS Model")

    return BertTextClassification(embed_model, cls_model)
