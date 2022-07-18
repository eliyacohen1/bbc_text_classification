from typing import Union

import numpy as np
from torch import Tensor
from torch import nn
from .wrapper_model_interface import WrapperModelInterface


class WrapperClassificationBertModel(WrapperModelInterface):
    def __init__(self, bert_cls_model: nn.Module):
        self.bert_cls_model = bert_cls_model

    def predict(self, inputs: Union[Tensor, np.ndarray], **kwargs) -> Union[Tensor, np.ndarray]:
        output = self.bert_cls_model(inputs, kwargs["mask"])
        return output
