# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from lunzi import Tensor
from .module import Module


class PointwiseLoss(Module):
    def __init__(self, size_average=True, reduce=True):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce

    def pointwise(self, output: Tensor, target: Tensor):
        raise NotImplementedError

    def forward(self, output: Tensor, target: Tensor, input: Tensor = None):
        loss = self.pointwise(output, target)
        if self.reduce and len(loss.shape) > 1:
            if self.size_average:
                loss = loss.reduce_mean(axis=1)
            else:
                loss = loss.reduce_sum(axis=1)
        return loss


class L1Loss(PointwiseLoss):
    def pointwise(self, output: Tensor, target: Tensor):
        return output.sub(target).abs()


class L2Loss(PointwiseLoss):
    def pointwise(self, output: Tensor, target: Tensor):
        return output.sub(target).pow(2)

    def forward(self, output: Tensor, target: Tensor, input: Tensor = None):
        return super().forward(output, target).sqrt()


class MSELoss(PointwiseLoss):
    def pointwise(self, output: Tensor, target: Tensor):
        return output.sub(target).pow(2)
