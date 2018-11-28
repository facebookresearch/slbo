# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Callable, List, Union
import inspect
from functools import wraps
import tensorflow as tf
import numpy as np
from lunzi import Tensor

from .parameter import Parameter


def make_method(feed: str = None, fetch: str = ''):
    """
        The following code:

            @make_method('. w', fetch='d)
            def func(a, c):
                pass

        will be converted to

            def func(a, c, fetch='d'):
                return self.eval(fetch, a=a, w=c)

        Note that `func(1, c=2, b=1)` is also supported. This is
        useful when writing PyTorch-like object method.

    """

    def decorator(func: Callable):
        arg_names = inspect.signature(func).parameters.keys()
        arg_map = {}
        if feed is None:
            arg_map = {op_name: op_name for op_name in arg_names if op_name != 'self'}
        else:
            feeds = ['-'] + feed.split(' ')  # ignore first `self`
            for op_name, arg_name in zip(feeds, arg_names):
                if op_name == '.':
                    arg_map[op_name] = op_name
                elif op_name != '-':  # deprecated
                    arg_map[op_name] = arg_name

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            cur_fetch = kwargs.pop('fetch', fetch)
            call_args = inspect.getcallargs(func, self, *args, **kwargs)
            feed_dict = {op_name: call_args[arg_name] for op_name, arg_name in arg_map.items()}
            return self.eval(cur_fetch, **feed_dict)

        return wrapper

    return decorator


def n_parameters(params: List[Parameter]) -> int:
    return sum([np.prod(p.shape) for p in params])


def parameters_to_vector(parameters: List[Union[Parameter, Tensor]]) -> Tensor:
    return tf.concat([param.reshape([-1]) for param in parameters], axis=0)


def vector_to_parameters(vec: Tensor, parameters: List[Parameter]) -> List[Tensor]:
    params: List[Tensor] = []
    start = 0
    for p in parameters:
        end = start + np.prod(p.shape)
        params.append(vec[start:end].reshape(p.shape))
        start = end
    return params


def hessian_vec_prod(ys: Tensor, xs: List[Parameter], vs: Tensor) -> Tensor:
    grad = parameters_to_vector(tf.gradients(ys, xs))
    aux = (grad * vs).reduce_sum()
    return parameters_to_vector(tf.gradients(aux, xs))


# credit to https://github.com/renmengye/tensorflow-forward-ad/issues/2#issue-234418055
def jacobian_vec_prod(ys: Tensor, xs: List[Parameter], vs: Tensor) -> Tensor:
    u = tf.zeros_like(ys)  # dummy variable
    grad = tf.gradients(ys, xs, grad_ys=u)
    return tf.gradients(grad, u, grad_ys=vs)


