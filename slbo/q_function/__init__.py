import abc
from typing import Union
import lunzi.nn as nn


class BaseQFunction(abc.ABC):
    @abc.abstractmethod
    def get_q(self, states, values):
        pass


BaseNNQFunction = Union[BaseQFunction, nn.Module]
