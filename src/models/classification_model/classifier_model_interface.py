import abc
from typing import Union

import torch
from torch import nn
import numpy as np


class ClassifierModelInterface(nn.Module):

    @abc.abstractmethod
    def forward(self, embedding_input: Union[torch.Tensor, np.ndarray]):
        raise NotImplementedError
