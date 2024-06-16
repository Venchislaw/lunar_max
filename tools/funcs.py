import numpy as np


def sigmoid(linear_pred):
    return 1 / (1 + np.exp(-linear_pred))
