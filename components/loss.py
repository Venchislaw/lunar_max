import numpy as np


# Regression:

def numpyrize(x1, x2):
    if type(x1) is list:
        x1 = np.array(x1)
    if type(x2) is list:
        x2 = np.array(x2)
    return x1, x2


def mean_squared_error(y_true, y_pred):
    y_true, y_pred = numpyrize(y_true, y_pred)
    return np.sum((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = numpyrize(y_true, y_pred)
    return np.sum(np.abs(y_true - y_pred))


# Classification:

def binary_crossentropy(y_true, y_pred):
    y_true, y_pred = numpyrize(y_true, y_pred)

    term_0 = (1 - y_true) * np.log(1 - y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)

    return -np.mean(term_0 + term_1, axis=0)
