from typing import Union
import numpy as np
import torch

from .classification_model.classifier_model_interface import ClassifierModelInterface
from .embedding_model.bert_embbeding import EmbeddingModelInterface
from .text_classification_interface import TextClassificationInterface


class BertTextClassification(TextClassificationInterface):
    def __init__(self, embed_model: EmbeddingModelInterface, cls_model: ClassifierModelInterface):
        super(BertTextClassification, self).__init__()

        self.embed_model = embed_model
        self.cls_model = cls_model

    def forward(self, inputs: Union[torch.Tensor, np.ndarray], mask: Union[torch.Tensor, np.ndarray]):
        embedding = self.embed_model(inputs, mask)
        output = self.cls_model(embedding)
        return output
