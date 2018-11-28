# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
class AverageMeter(object):
    sum: float
    count: float

    def __init__(self, discount=1.):
        self.discount = discount
        self.reset()

    def update(self, value, count=1):
        self.sum = self.sum * self.discount + value * count
        self.count = self.count * self.discount + count
        return self.get()

    def get(self):
        return self.sum / (self.count + 1.e-8)

    def reset(self):
        self.sum = 0.
        self.count = 0.
