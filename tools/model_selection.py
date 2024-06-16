from Main.tools.utils import shuffle as shfl
import numpy as np


def train_test_split(X, y, train_size, shuffle=False):
    if shuffle:
        X, y = shfl(X, y)

    n = len(X)
    X_train, y_train = X[:int(n*train_size)], y[:int(n*train_size)]
    X_test, y_test = X[int(n*train_size):], y[int(n*train_size):]

    return X_train, X_test, y_train, y_test
