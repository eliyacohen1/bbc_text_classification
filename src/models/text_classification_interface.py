import abc
from typing import Union

import torch
from torch import nn
import numpy as np


class TextClassificationInterface(nn.Module):

    @abc.abstractmethod
    def forward(self, inputs: Union[torch.Tensor, np.ndarray], mask: Union[torch.Tensor, np.ndarray]):
        raise NotImplementedError
