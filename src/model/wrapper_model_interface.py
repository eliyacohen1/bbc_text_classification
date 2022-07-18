import abc
from typing import Union

import numpy as np
from torch import Tensor


class WrapperModelInterface:

    @abc.abstractmethod
    def predict(self, inputs, **kwargs) -> Union[Tensor, np.ndarray]:
        raise NotImplementedError
