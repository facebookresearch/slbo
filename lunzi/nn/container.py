# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Dict, Any
from .module import Module
from .parameter import Parameter

_dict_methods = ['__setitem__', '__getitem__', '__delitem__', '__len__', '__iter__', '__contains__',
                 'update', 'keys', 'values', 'items', 'clear', 'pop']


class ModuleDict(Module, dict):  # use dict for auto-complete
    """
        Essentially this exposes some methods of `Module._modules`.
    """
    def __init__(self, modules: Dict[Any, Module] = None):
        super().__init__()
        for method in _dict_methods:
            setattr(self, method, getattr(self._modules, method))
        if modules is not None:
            self.update(modules)

    def forward(self):
        raise RuntimeError("ModuleDict is not callable")


# Do we need a factory for it?
class ParameterDict(Module, dict):
    def __init__(self, parameters: Dict[Any, Parameter] = None):
        super().__init__()
        for method in _dict_methods:
            setattr(self, method, getattr(self._modules, method))
        if parameters is not None:
            self.update(parameters)

    def forward(self):
        raise RuntimeError("ParameterDict is not callable")
