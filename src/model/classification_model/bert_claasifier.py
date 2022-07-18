from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):

    def __init__(self, pretrained_name: str, layers: list, num_classes: int, activation_func=nn.ReLU(), dropout=0.3):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained_name)
        self.model_layers = [nn.Linear(layers[0], layers[1])]
        self.activation_func = activation_func
        self.dropout = nn.Dropout(dropout)

        for layer_idx in range(2, len(layers) - 1):
            self.model_layers.append(self.activation_func)
            self.model_layers.append(self.dropout)
            self.model_layers.append(nn.Linear(layers[layer_idx - 1], layers[layer_idx]))
        self.model_layers.append(nn.Linear(layers[-2], num_classes))

        self.fc_module = nn.Sequential(*self.model_layers)

    def forward(self, input_id, mask):
        _, pooled = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        out = self.fc_module(pooled)

        return out
