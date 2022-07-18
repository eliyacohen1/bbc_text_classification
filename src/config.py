from torch import nn

config = {
    "model_name": "bert",
    "bert_settings": {
        "layers": [768, 256, 50],
        "dropout": 0.5,
        "model_name": 'bert-base-uncased',
        "num_classes": 5,
        "activation_func": nn.ReLU()
    },
    "train_settings": {
        "batch_size": 50
    },
    "tokenizer_config": {
        "padding": "max_length",
        "max_length": 512,
        "truncation": True,
        "return_tensors": "pt",
        "bert_tokenizer_name": 'bert-base-uncased',
    },
    "labels": {'business': 0,
               'entertainment': 1,
               'sport': 2,
               'tech': 3,
               'politics': 4
               },

}
