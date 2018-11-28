# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np


def gaussian_kl(mean_1: np.ndarray, std_1: np.ndarray, mean_2: np.ndarray, std_2: np.ndarray) -> np.ndarray:
    eps = 1e-20
    std_1 = np.maximum(std_1, eps)
    std_2 = np.maximum(std_2, eps)
    return np.log(std_2 / std_1) + (std_1**2 + (mean_1 - mean_2)**2) / std_2**2 / 2. - 0.5
