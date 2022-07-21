import abc
from typing import Union

import torch
from torch import nn
import numpy as np


class EmbeddingModelInterface(nn.Module):

    @abc.abstractmethod
    def forward(self, tokenize_text: Union[torch.Tensor, np.ndarray], mask: Union[torch.Tensor, np.ndarray]):
        raise NotImplementedError
