import numpy as np


class Model:
    def __init__(self):
        self.learning_rate = 0.001
        self.weights = None
        self.bias = 0
        self.history = []

    def fit(self, X, y, iterations, learning_rate=None):
        if not self.weights:
            m_samples, n_features = X.shape
            self.weights = np.zeros((n_features, 1))

        if learning_rate: self.learning_rate = learning_rate
        return m_samples, n_features
