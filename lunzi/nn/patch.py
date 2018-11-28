# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import tensorflow as tf

from lunzi.Logger import logger


def find_monkey_patch_keys(avoid_set=None):
    if avoid_set is None:
        avoid_set = {"shape"}  # tf.shape conflicts with Tensor.shape
    patched = []
    for key, value in tf.__dict__.items():
        if not callable(value) or key in avoid_set:
            continue
        doc = value.__doc__
        if doc is None:
            continue
        loc = doc.find('Args:\n')
        if loc == -1:
            continue

        # Am I doing NLP?
        # It seems that PyTorch has better doc. They always write `x (Tensor): ...` which is much easier to parse.
        first_arg_doc = doc[loc + 6:].split('\n')[0].split(': ')[1]
        if first_arg_doc.startswith('A `Tensor`') or first_arg_doc.startswith('`Tensor`') or key.startswith('reduce_'):
            patched.append(key)
    logger.warning(f'Monkey patched TensorFlow: {patched}')
    return patched


def monkey_patch(avoid_set=None):
    logger.warning('Monkey patching TensorFlow...')

    patched = ['abs', 'acos', 'acosh', 'add', 'angle', 'argmax', 'argmin', 'asin', 'asinh', 'atan', 'atan2', 'atanh',
            'betainc', 'cast', 'ceil', 'check_numerics', 'clip_by_average_norm', 'clip_by_norm', 'clip_by_value',
            'complex', 'conj', 'cos', 'cosh', 'cross', 'cumprod', 'cumsum', 'dequantize', 'diag', 'digamma', 'div',
            'equal', 'erf', 'erfc', 'exp', 'expand_dims', 'expm1', 'fill', 'floor', 'floor_div', 'floordiv', 'floormod',
            'gather', 'gather_nd', 'greater', 'greater_equal', 'hessians', 'identity', 'igamma', 'igammac', 'imag',
            'is_finite', 'is_inf', 'is_nan', 'less', 'less_equal', 'lgamma', 'log', 'log1p', 'logical_and',
            'logical_not', 'logical_or', 'matmul', 'maximum', 'meshgrid', 'minimum', 'mod', 'multiply', 'negative',
            'norm', 'not_equal', 'one_hot', 'ones_like', 'pad', 'polygamma', 'pow', 'quantize', 'real', 'realdiv',
            'reciprocal', 'reduce_all', 'reduce_any', 'reduce_logsumexp', 'reduce_max', 'reduce_mean', 'reduce_min',
            'reduce_prod', 'reduce_sum', 'reshape', 'reverse', 'rint', 'round', 'rsqrt', 'scatter_nd', 'sign', 'sin',
            'sinh', 'size', 'slice', 'sqrt', 'square', 'squeeze', 'stop_gradient', 'subtract', 'tan', 'tensordot',
            'tile', 'to_bfloat16', 'to_complex128', 'to_complex64', 'to_double', 'to_float', 'to_int32', 'to_int64',
            'transpose', 'truediv', 'truncatediv', 'truncatemod', 'unique', 'where', 'zeros_like', 'zeta']
    alias = {
        'mul': 'multiply',
        'sub': 'subtract',
    }

    # use the code below for more ops
    # patched = find_monkey_patch_keys(avoid_set)

    for key, method in list(zip(patched, patched)) + list(alias.items()):
        value = tf.__dict__[method]
        setattr(tf.Tensor, key, value)
        setattr(tf.Variable, key, value)


monkey_patch()
