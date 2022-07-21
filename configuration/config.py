import torch
from torch import nn

MODELS_FOLDERS = "torch_models/{}_classifier.pth"

config = {
    "embedding": "bert",
    "classifier": "mlp",
    "model_name": "bert",
    "bert_settings": {
        "model_name": 'bert-base-cased'
    },
    "mlp_settings": {
        "layers": [768, 50, 10],
        "dropout": 0.5,
        "num_classes": 5,
        "activation_func": nn.ReLU()
    },
    "tokenizer_config": {
        "model_name": 'bert-base-cased',
        "padding": "max_length",
        "max_length": 512,
        "truncation": True,
        "return_tensors": "pt",
    },
    "train_settings": {
        "batch_size": 5,
        "epochs": 2,
        "lr": 1e-6,
        "optimizer": torch.optim.Adam,
        "weight_decay": 0.0
    },
    "labels": {'business': 0,
               'entertainment': 1,
               'sport': 2,
               'tech': 3,
               'politics': 4
               },

}
