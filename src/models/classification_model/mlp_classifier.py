from torch import nn

from .classifier_model_interface import ClassifierModelInterface


class MLPClassifier(ClassifierModelInterface):

    def __init__(self, layers: list, num_classes: int, activation_func=nn.ReLU(), dropout=0.3):
        super(MLPClassifier, self).__init__()

        self.model_layers = [nn.Linear(layers[0], layers[1])]
        self.activation_func = activation_func
        self.dropout = nn.Dropout(dropout)

        for layer_idx in range(2, len(layers) - 1):
            self.model_layers.append(self.activation_func)
            self.model_layers.append(self.dropout)
            self.model_layers.append(nn.Linear(layers[layer_idx - 1], layers[layer_idx]))
        self.model_layers.append(nn.Linear(layers[-2], num_classes))

        self.fc_module = nn.Sequential(*self.model_layers)

    def forward(self, embedding_input):
        out = self.fc_module(embedding_input)
        return out
