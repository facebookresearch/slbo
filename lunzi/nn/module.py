# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Dict, Any, Callable, List
from collections import Counter
import tensorflow as tf
import numpy as np

from lunzi import Tensor
from lunzi.Logger import logger
from .parameter import Parameter


class Module(object):
    """
        A front-end for TensorFlow, heavily inspired by PyTorch's design and implementation.

        Deepcopy is not supported since I didn't find a good way to duplicate `tf.Variables` and `tf.variable_scope`.
    """

    # To generate unique name scope
    # The only reason we keep variable scope here is that we want the variables have meaning names,
    # since the internal operations always look messy, I put no hope maintaining their names,
    # So let's just do it for variables.
    prefix_count = Counter()

    @staticmethod
    def _create_uid(prefix: str) -> str:
        scope = tf.get_variable_scope().name + '/'
        uid = Module.prefix_count[scope + prefix]
        Module.prefix_count[scope + prefix] += 1
        if uid == 0:
            return prefix
        return f'{prefix}_{uid}'

    def __init__(self):
        scope = Module._create_uid(self.__class__.__name__)
        with tf.variable_scope(scope, reuse=False) as self._scope:
            pass
        # Since we only plan to support Python 3.6+, in which dict is already ordered, we don't use OrderedDict here.
        self._parameters: Dict[Any, Parameter] = {}
        self._modules: Dict[Any, Module] = {}
        self._callables: Dict[Any, Callable] = {}

    def forward(self, *args: List[Any], **kwargs: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def fast(self, *args, **kwargs):
        pass

    def __setattr__(self, key, value):
        # dynamically maintain sub modules.
        modules = self.__dict__.get('_modules')
        if isinstance(value, Parameter):
            self._parameters[key] = value
        if isinstance(value, Module):
            assert modules is not None, 'Call `super().__init__` before assigning modules'
            modules[key] = value
        else:
            if modules and key in modules:
                del modules[key]
        object.__setattr__(self, key, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def register_callable(self, key, callable):
        self._callables[key] = callable

    def eval(self, fetch: str, **feed: Dict[str, np.ndarray]):
        cache_key = f'[{" ".join(feed.keys())}] => [{fetch}]'
        if cache_key not in self._callables:
            logger.info('[%s] is making TensorFlow callables, key = %s', self.__class__.__name__, cache_key)
            feed_ops = []
            for key in feed.keys():
                feed_ops.append(self.__dict__['op_' + key])
            if isinstance(fetch, str):
                fetch_ops = [self.__dict__['op_' + key] for key in fetch.split(' ')]
                if len(fetch_ops) == 1:
                    fetch_ops = fetch_ops[0]
            else:
                fetch_ops = fetch
            self.register_callable(cache_key, tf.get_default_session().make_callable(fetch_ops, feed_ops))
        return self._callables[cache_key](*feed.values())

    def parameters(self, trainable=True, non_trainable=False, recursive=True, out=None) -> List[Parameter]:
        """
            We don't introduce `buffers` here. PyTorch has it since it doesn't have non-trainable Parameter.
            A tensor in `buffers` is essentially a non-trainable Parameter (part of state_dict but isn't
            optimized over).
        """
        if out is None:
            out = []
        for param in self._parameters.values():
            if param.trainable and trainable or not param.trainable and non_trainable:
                out.append(param)
        if recursive:
            for module in self._modules.values():
                module.parameters(trainable=trainable, non_trainable=non_trainable, recursive=True, out=out)
        # probably we don't need to sort since we're using `OrderedDict`
        return out

    @property
    def scope(self) -> tf.variable_scope:
        return tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE)

    def extra_repr(self) -> str:
        return ''

    def named_modules(self) -> dict:
        return self._modules

    def __repr__(self):
        def dfs(node, prefix):
            root_info = node.__class__.__name__
            modules = node.named_modules()
            if not modules:
                return root_info + f'({node.extra_repr()})'

            root_info += '(\n'
            for key, module in modules.items():
                module_repr = dfs(module, prefix + '    ')
                root_info += f'{prefix}    ({key}): {module_repr}\n'
            root_info += prefix + ')'
            return root_info
        return dfs(self, '')

    def state_dict(self, recursive=True):
        """
            A better option is to find all parameters and then sess.run(state) but I assume this can't be the
            bottleneck.
        """
        state = {}
        for key, parameter in self._parameters.items():
            # although we can use `.numpy()` here, for safety I'd use `.eval()`
            state[key] = parameter.eval()
        if recursive:
            for key, module in self._modules.items():
                state[key] = module.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[Any, Any], recursive=True, strict=True):
        for key, parameter in self._parameters.items():
            if key in state_dict:
                parameter.load(state_dict[key])
                parameter.invalidate()
            else:
                assert not strict, f'Missing Parameter {key} in state_dict'
        if recursive:
            for key, module in self._modules.items():
                if key in state_dict:
                    module.load_state_dict(state_dict[key], recursive=recursive, strict=strict)
                else:
                    assert not strict, f'Missing Module {key} in state_dict.'

    def apply(self, fn):
        for module in self._modules.values():
            module.apply(fn)
        fn(self)
        return self
