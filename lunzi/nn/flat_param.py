# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import tensorflow as tf
from lunzi.Logger import logger
from .module import Module
from .utils import make_method, n_parameters, parameters_to_vector, vector_to_parameters


class FlatParam(Module):
    def __init__(self, parameters):
        super().__init__()
        self.params = parameters
        self.op_feed_flat, self.op_set_flat, self.op_get_flat = \
            self.enable_flat()

    def enable_flat(self):
        params = self.params
        logger.info('Enabling flattening... %s', [p.name for p in params])
        n_params = n_parameters(params)
        feed_flat = tf.placeholder(tf.float32, [n_params])
        get_flat = parameters_to_vector(params)
        set_flat = tf.group(*[tf.assign(param, value) for param, value in
                            zip(params, vector_to_parameters(feed_flat, params))])
        return feed_flat, set_flat, get_flat

    def forward(self):
        return self.op_get_flat

    @make_method(feed='feed_flat', fetch='set_flat')
    def set_flat(self, feed_flat):
        pass

    @make_method(fetch='get_flat')
    def get_flat(self):
        pass
